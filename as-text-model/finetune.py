"""Fine-tune GPT-2 on the Artificial Stupidity corpus.

Why this exists alongside train.py:

    train.py builds a model from scratch, character by character. It gets us
    a 169 KB language model, which is the point of the AS-0..AS-5 experiment,
    but it can never spell reliably — it has to guess "mitochondria" one letter
    at a time and it will lose that bet.

    This file starts from GPT-2, which already knows English. It emits whole
    tokens, so it physically cannot misspell a word. Fine-tuning on our corpus
    keeps the grammar and replaces the personality.

Result: talks normal, says stupid things. Then compress.py breaks it.

    python finetune.py --base gpt2 --epochs 1
    python finetune.py --base distilgpt2 --max-mb 40      # faster, smaller
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def build_dataset(tokenizer, raw_dir: Path, max_mb: float, block_size: int,
                  cache: Path):
    """Corpus -> one long stream of token ids, chunked into training blocks."""
    if cache.exists():
        ids = np.fromfile(cache, dtype=np.uint16)
        print(f"  loaded cached tokens: {len(ids):,}", flush=True)
        return ids

    files = [f for f in sorted(raw_dir.rglob("*.txt")) if f.stat().st_size > 0]
    if not files:
        raise SystemExit(f"no .txt under {raw_dir} — run data/collect.py first")

    # Take a proportional slice of EVERY source rather than reading files in
    # alphabetical order until the budget runs out. That naive version handed
    # us 80% Twitch and almost no YouTube purely because "t" sorts after "y"
    # is false and twitch.txt happened to be huge — the model would have
    # learned to speak only in chat fragments.
    budget = int(max_mb * 1e6)
    sizes = {f: f.stat().st_size for f in files}
    corpus_size = sum(sizes.values())
    scale = min(1.0, budget / corpus_size)

    chunks, total = [], 0
    for f in files:
        take = int(sizes[f] * scale)
        if take <= 0:
            continue
        text = f.read_text(encoding="utf-8", errors="ignore")[:take]
        # don't end mid-line
        if "\n" in text:
            text = text[: text.rindex("\n") + 1]
        chunks.append(text)
        total += len(text)
        print(f"  {f.name:<22} {len(text) / 1e6:6.2f} MB  "
              f"({100 * len(text) / budget:4.1f}% of budget)", flush=True)

    # Shuffle at ~50 KB granularity before concatenating. Without this the
    # sources sit in alphabetical blocks, so the last 2% of the stream — which
    # becomes the validation set — is pure youtube.txt. Validation loss then
    # measures "how well does it model YouTube" and RISES as the model
    # correctly learns the Twitch-heavy training mix, which makes the
    # best-checkpoint logic save an under-trained model.
    import random
    blocks = []
    for text in chunks:
        blocks.extend(text[i:i + 50_000] for i in range(0, len(text), 50_000))
    random.Random(1337).shuffle(blocks)

    print(f"  tokenizing {total / 1e6:.1f} MB in {len(blocks):,} shuffled blocks "
          f"(this takes a minute)...", flush=True)
    eos = tokenizer.eos_token_id
    all_ids = []
    for block in blocks:
        all_ids.extend(tokenizer(block)["input_ids"])
        all_ids.append(eos)

    ids = np.array(all_ids, dtype=np.uint16)
    cache.parent.mkdir(parents=True, exist_ok=True)
    ids.tofile(cache)
    print(f"  {len(ids):,} tokens -> {cache.relative_to(ROOT)}", flush=True)
    return ids


def get_batch(ids, batch_size, block_size, device):
    """One batch of token blocks.

    Only x is returned, deliberately. GPT2LMHeadModel shifts labels internally
    — it compares logits[:, :-1] against labels[:, 1:] — so the caller passes
    `labels=x`. Handing it a pre-shifted y would shift twice and train the
    model to predict two tokens ahead, which looks like a loss around 9
    instead of 3.
    """
    ix = torch.randint(len(ids) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(ids[i:i + block_size].astype(np.int64)) for i in ix])
    return x.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="gpt2",
                    help="gpt2 (124M) or distilgpt2 (82M, faster)")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--max-mb", type=float, default=60,
                    help="how much of the corpus to use")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--out", default="checkpoints/AS-F")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("[!] MPS unavailable, using CPU (this will be slow)")
        device = "cpu"

    torch.manual_seed(1337)

    print(f"\n{'=' * 62}")
    print(f"  FINE-TUNING  {args.base}")
    print(f"{'=' * 62}")

    tok = GPT2TokenizerFast.from_pretrained(args.base)
    tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.base).to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters    {n_params:,}")
    print(f"  fp32 size     {human(n_params * 4)}")
    print(f"  device        {device}")

    # cache key includes the source dir — stage 2 trains on a different corpus
    # and must not reuse stage 1's tokens
    tag = Path(args.raw_dir).name
    cache = ROOT / "data" / "processed" / f"gpt2_tokens_{tag}_{int(args.max_mb)}mb.bin"
    ids = build_dataset(tok, ROOT / args.raw_dir, args.max_mb, args.block_size, cache)

    n = int(0.98 * len(ids))
    train_ids, val_ids = ids[:n], ids[n:]
    print(f"  train tokens  {len(train_ids):,}")
    print(f"  val tokens    {len(val_ids):,}")
    print(f"{'=' * 62}\n", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.01)

    def lr_at(it):
        if it < args.warmup:
            return args.lr * (it + 1) / (args.warmup + 1)
        ratio = (it - args.warmup) / max(1, args.max_iters - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * ratio)))

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {}
        for name, data in (("train", train_ids), ("val", val_ids)):
            losses = []
            for _ in range(args.eval_iters):
                x = get_batch(data, args.batch_size, args.block_size, device)
                losses.append(model(input_ids=x, labels=x).loss.item())
            out[name] = sum(losses) / len(losses)
        model.train()
        return out

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    t0 = time.time()

    for it in range(args.max_iters + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = evaluate()
            flag = ""
            if losses["val"] < best_val:
                best_val = losses["val"]
                model.save_pretrained(out_dir)
                tok.save_pretrained(out_dir)
                flag = " *"
            print(f"  iter {it:>5}  train {losses['train']:.4f}  "
                  f"val {losses['val']:.4f}  ppl {math.exp(losses['val']):7.2f}  "
                  f"{time.time() - t0:5.0f}s{flag}", flush=True)

        if it == args.max_iters:
            break

        opt.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x = get_batch(train_ids, args.batch_size, args.block_size, device)
            loss = model(input_ids=x, labels=x).loss / args.grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    # Always keep the final weights alongside the best-val ones. Validation is
    # a proxy for "is this good", and for a humour model it's a weak one — the
    # funniest checkpoint is not reliably the one with the lowest loss.
    final_dir = out_dir.parent / f"{out_dir.name}-final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tok.save_pretrained(final_dir)

    print(f"\n  best val loss  {best_val:.4f}  (perplexity {math.exp(best_val):.2f})")
    print(f"  best-val ckpt  {out_dir.relative_to(ROOT)}")
    print(f"  final ckpt     {final_dir.relative_to(ROOT)}")
    print(f"\n  try it:  python talk.py --model {args.out} --chat\n")


if __name__ == "__main__":
    main()
