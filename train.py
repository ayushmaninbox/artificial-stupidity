"""Train one Artificial Stupidity variant.

    python train.py AS-0
    python train.py AS-4 --max-iters 6000
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import config as config_module
from model import ArtificialStupidity, CharTokenizer

ROOT = Path(__file__).resolve().parent


def pick_device(requested):
    if requested == "mps" and not torch.backends.mps.is_available():
        print("[!] MPS unavailable, falling back to CPU")
        return "cpu"
    return requested


def get_batch(data, cfg, device):
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + cfg.block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, splits, cfg, device):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, cfg, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(it, cfg):
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / (cfg.warmup_iters + 1)
    if it > cfg.max_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def human(nbytes):
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", nargs="?", default="AS-0")
    ap.add_argument("--max-iters", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = config_module.get(args.preset)
    if args.max_iters:  cfg.max_iters = args.max_iters
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.device:     cfg.device = args.device

    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)

    data_dir = ROOT / cfg.data_dir
    if not (data_dir / "train.bin").exists():
        raise SystemExit("no training data. run: python data/prepare.py")

    tok = CharTokenizer.load(data_dir / "tokenizer.json")
    cfg.vocab_size = tok.vocab_size
    splits = {
        "train": np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r"),
        "val": np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r"),
    }

    model = ArtificialStupidity(cfg).to(device)

    print(f"\n{'=' * 58}")
    print(f"  {cfg.name}  —  {cfg.weight_bits}-bit weights, {cfg.act_bits}-bit activations")
    print(f"{'=' * 58}")
    print(f"  device        {device}")
    print(f"  parameters    {model.num_params():,}")
    print(f"  vocab         {cfg.vocab_size}")
    print(f"  context       {cfg.block_size}")
    print(f"  train tokens  {len(splits['train']):,}")
    print(f"  exported size {human(model.packed_bytes())}")
    print(f"{'=' * 58}\n")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2),
    )

    out_dir = ROOT / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{cfg.name}.pt"

    best_val = float("inf")
    t0 = time.time()
    history = []

    for it in range(cfg.max_iters + 1):
        lr = lr_at(it, cfg)
        for group in opt.param_groups:
            group["lr"] = lr

        if it % cfg.eval_interval == 0 or it == cfg.max_iters:
            losses = estimate_loss(model, splits, cfg, device)
            elapsed = time.time() - t0
            flag = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({
                    "model": model.state_dict(),
                    "config": cfg.dict(),
                    "iter": it,
                    "val_loss": best_val,
                }, ckpt_path)
                flag = " *"
            history.append({"iter": it, **losses})
            print(f"  iter {it:>5}  train {losses['train']:.4f}  "
                  f"val {losses['val']:.4f}  lr {lr:.2e}  {elapsed:5.0f}s{flag}")

        if it == cfg.max_iters:
            break

        x, y = get_batch(splits["train"], cfg, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

    (out_dir / f"{cfg.name}.history.json").write_text(json.dumps(history, indent=2))
    print(f"\n  best val loss  {best_val:.4f}")
    print(f"  saved          {ckpt_path.relative_to(ROOT)}")
    print(f"\n  try it:  python generate.py {cfg.name} --prompt 'hello'\n")


if __name__ == "__main__":
    main()
