"""Stage 2 — train the prior. This is the part that turns text into pictures.

Runs entirely on cached latents (see precompute_latents.py), so the VAE never
executes here. One step is: pick latents, noise them, ask the U-Net to predict
v given the caption, backprop.

Classifier-free guidance is trained in by blanking the caption on a fraction of
examples. The model therefore learns both p(image | text) and p(image), and at
sampling time we extrapolate away from the unconditional prediction. Without
this, prompt adherence at 8 steps is very poor; with it, guidance ~4 is the
difference between "a coloured blob" and "a red heart".

    python train_diffusion.py --data data/emoji
    python train_diffusion.py --data data/emoji --iters 40000 --preset AS-I-S
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as config_module
from diffusion import Diffusion, EMA
from model import TextEncoder, WordTokenizer, UNet

ROOT = Path(__file__).resolve().parent


class Prior(nn.Module):
    """Text encoder + U-Net as one module, so they train and save together."""

    def __init__(self, cfg, vocab_size):
        super().__init__()
        self.text = TextEncoder(vocab_size, cfg.text_dim, cfg.text_layers,
                                cfg.text_heads, cfg.max_tokens)
        self.unet = UNet(cfg.latent_ch, cfg.unet_base, tuple(cfg.unet_mult),
                         cfg.text_dim, cfg.unet_heads)

    def forward(self, z, t, ids):
        ctx, pad = self.text(ids)
        return self.unet(z, t, ctx, pad)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="AS-I")
    ap.add_argument("--data", default=None)
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--device")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = config_module.get(args.preset)
    if args.data:       cfg.data_dir = args.data
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.lr:         cfg.lr = args.lr
    if args.device:     cfg.device = args.device
    device = cfg.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    torch.manual_seed(cfg.seed)

    data = ROOT / cfg.data_dir
    if not (data / "train" / "latents.pt").exists():
        raise SystemExit("no latents. run: python precompute_latents.py "
                         f"--data {cfg.data_dir}")

    print(f"\n{'=' * 62}\n  DIFFUSION PRIOR  —  {cfg.name}\n{'=' * 62}")

    splits = {}
    for s in ("train", "val"):
        blob = torch.load(data / s / "latents.pt", map_location="cpu", weights_only=False)
        caps = json.loads((data / s / "captions.json").read_text())
        splits[s] = (blob["latents"].float(), caps)
        print(f"  {s:<6} {blob['latents'].shape[0]:>6,} latents  "
              f"{tuple(blob['latents'].shape[1:])}  std {blob['latents'].float().std():.2f}")
    latent_scale = blob["scale"]

    tok = WordTokenizer.build(splits["train"][1] + splits["val"][1])
    tok.save(data / "tokenizer.json")
    print(f"  vocab  {tok.vocab_size} words")

    model = Prior(cfg, tok.vocab_size).to(device)
    diff = Diffusion(cfg.timesteps, cfg.schedule, device=device)
    ema = EMA(model, cfg.ema_decay)

    n_text = sum(p.numel() for p in model.text.parameters())
    n_unet = sum(p.numel() for p in model.unet.parameters())
    print(f"  text encoder  {n_text:>12,}")
    print(f"  unet          {n_unet:>12,}")
    print(f"  total         {model.num_params():>12,}   "
          f"({model.num_params() * 2 / 1e6:.0f} MB fp16, "
          f"{model.num_params() / 1e6:.0f} MB int8)")
    print(f"  device        {device}")
    print(f"{'=' * 62}\n", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95),
                            weight_decay=0.01)

    # pre-tokenize every caption once; they never change
    print("  tokenizing captions ...", flush=True)
    ids_cache = {s: tok.batch(c, cfg.max_tokens, "cpu") for s, (_, c) in splits.items()}
    null_ids = tok.batch([""], cfg.max_tokens, device)          # for CFG
    print("  done\n", flush=True)

    def lr_at(it):
        if it < cfg.warmup:
            return cfg.lr * (it + 1) / (cfg.warmup + 1)
        import math
        r = (it - cfg.warmup) / max(1, args.iters - cfg.warmup)
        return cfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * r)))

    def get_batch(split, bs):
        z, _ = splits[split]
        idx = torch.randint(0, z.shape[0], (bs,))
        zb = z[idx].to(device)
        ib = ids_cache[split][idx].to(device)
        return zb, ib

    def loss_on(zb, ib, train=True):
        t = torch.randint(0, cfg.timesteps, (zb.shape[0],), device=device)
        noise = torch.randn_like(zb)
        zt = diff.q_sample(zb, t, noise)
        target = diff.v_target(zb, t, noise)
        if train and cfg.cfg_dropout > 0:
            drop = torch.rand(zb.shape[0], device=device) < cfg.cfg_dropout
            ib = torch.where(drop[:, None], null_ids.expand_as(ib), ib)
        return F.mse_loss(model(zt, t, ib), target)

    out_dir = ROOT / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / f"{cfg.name}-prior.pt"

    best = float("inf")
    t0 = time.time()
    for it in range(args.iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)

        model.train()
        zb, ib = get_batch("train", cfg.batch_size)
        loss = loss_on(zb, ib, train=True)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        ema.update(model)

        if it % args.eval_every == 0 or it == args.iters:
            model.eval()
            with torch.no_grad():
                vl = sum(loss_on(*get_batch("val", cfg.batch_size), train=False).item()
                         for _ in range(8)) / 8
            flag = ""
            if vl < best:
                best = vl
                torch.save({"model": model.state_dict(),
                            "ema": ema.shadow,
                            "config": cfg.dict(),
                            "vocab": tok.itos,
                            "latent_scale": latent_scale,
                            "val": vl, "iter": it}, ckpt)
                flag = " *"
            el = time.time() - t0
            print(f"  iter {it:>6}  train {loss.item():.4f}  val {vl:.4f}  "
                  f"lr {lr_at(it):.1e}  {el / 60:5.1f}m{flag}", flush=True)

    print(f"\n  best val loss  {best:.4f}")
    print(f"  saved          {ckpt.relative_to(ROOT)}")
    print(f"\n  try it:  python sample.py --prompt 'red heart'\n")


if __name__ == "__main__":
    main()
