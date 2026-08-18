"""Break a fine-tuned GPT-2 on purpose, and measure exactly how much.

shrek_GPT does this with bitsandbytes 4-bit NF4 — which is CUDA-only and will
not run on an Apple Silicon Mac. We don't need it: we already wrote honest
quantizers for the AS-0..AS-5 experiment, so we point those at GPT-2 instead
and get the same crimes running natively on MPS, with real byte counts.

    python compress.py --bits 4
    python compress.py --bits 1.58 --emb-bits 4 --sparsity 0.3

A note on where the bytes actually are. GPT-2's token embedding table is
50257 x 768 = 38.6M parameters — about a third of the whole model. Leaving it
in fp16 costs 77 MB on its own and makes the compression pointless, so
`--emb-bits` quantizes it too. It's the single biggest lever on final size.
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model.bitlinear import quantize_weight

ROOT = Path(__file__).resolve().parent


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def bytes_for(numel: int, rows: int, bits: float) -> int:
    """Real packed size: bit-packed codes + one fp16 scale per output row."""
    if bits >= 32:
        return numel * 4
    if bits == 16:
        return numel * 2
    if abs(bits - 1.58) < 1e-6:
        return math.ceil(numel / 5) + rows * 2      # 5 ternary values per byte
    return math.ceil(numel * bits / 8) + rows * 2


def quantize_matrix(w: torch.Tensor, bits: float, per_column: bool) -> torch.Tensor:
    """Quantize with the scale computed along the output dimension.

    HuggingFace GPT-2 uses Conv1D, not Linear, and stores its weight as
    (in_features, out_features) — transposed relative to nn.Linear. Our
    quantizers scale along the last axis, so Conv1D weights get transposed
    first or every scale would be computed across the wrong dimension.
    """
    if per_column:
        return quantize_weight(w.t().contiguous(), bits).t().contiguous()
    return quantize_weight(w, bits)


@torch.no_grad()
def compress(model, bits: float, emb_bits: float, sparsity: float, seed: int = 0):
    """Quantize (and optionally sparsify) in place. Returns a size report."""
    from transformers.pytorch_utils import Conv1D

    gen = torch.Generator().manual_seed(seed)
    report, packed, original = [], 0, 0

    for name, module in model.named_modules():
        is_conv1d = isinstance(module, Conv1D)
        is_linear = isinstance(module, nn.Linear)
        is_emb = isinstance(module, nn.Embedding)
        if not (is_conv1d or is_linear or is_emb):
            continue

        w = module.weight.data
        target_bits = emb_bits if is_emb else bits
        rows = w.shape[1] if is_conv1d else w.shape[0]

        original += w.numel() * 4
        if target_bits >= 32:
            packed += w.numel() * 4
            continue

        q = quantize_matrix(w.float(), target_bits, per_column=is_conv1d)

        # Random sparsification: zero a fraction of weights outright. This is
        # pure damage with no retraining to compensate, which is the point —
        # but it also means the zeros still occupy their bit slots, so it costs
        # quality without buying any size back.
        if sparsity > 0 and not is_emb:
            mask = torch.rand(q.shape, generator=gen) >= sparsity
            q = q * mask.to(q.dtype)

        module.weight.data = q.to(module.weight.dtype)
        packed += bytes_for(w.numel(), rows, target_bits)
        report.append((name, tuple(w.shape), target_bits))

    # everything we didn't quantize (LayerNorms, biases) ships as fp16
    quantized_ids = {id(m.weight) for _, m in model.named_modules()
                     if hasattr(m, "weight") and m.weight is not None
                     and (isinstance(m, (nn.Embedding, nn.Linear))
                          or type(m).__name__ == "Conv1D")}
    for p in model.parameters():
        if id(p) not in quantized_ids:
            packed += p.numel() * 2
            original += p.numel() * 4

    return report, packed, original


@torch.no_grad()
def perplexity(model, ids, device, block_size=256, batches=40, batch_size=4):
    model.eval()
    losses = []
    rng = np.random.default_rng(0)
    for _ in range(batches):
        ix = rng.integers(0, len(ids) - block_size - 1, batch_size)
        x = torch.stack([torch.from_numpy(ids[i:i + block_size].astype(np.int64)) for i in ix]).to(device)
        # labels=x, not a shifted copy — GPT2LMHeadModel does the shift itself
        losses.append(model(input_ids=x, labels=x).loss.item())
    return math.exp(sum(losses) / len(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/AS-F")
    ap.add_argument("--out", help="where to save (default: <model>-<bits>bit)")
    ap.add_argument("--bits", type=float, default=4,
                    help="32, 8, 4, 1.58 (ternary) or 1 (binary)")
    ap.add_argument("--emb-bits", type=float, default=8,
                    help="embedding table precision — the biggest size lever")
    ap.add_argument("--sparsity", type=float, default=0.0,
                    help="fraction of weights to zero outright (0-1)")
    ap.add_argument("--tokens", default="data/processed/gpt2_tokens_60mb.bin",
                    help="token cache for the perplexity check")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    device = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    src = ROOT / args.model
    if not src.exists():
        raise SystemExit(f"no model at {src}. run finetune.py first.")

    print(f"\n{'=' * 62}")
    print(f"  COMPRESSING  {args.model}")
    print(f"  weights {args.bits}-bit | embeddings {args.emb_bits}-bit"
          f" | sparsity {args.sparsity:.0%}")
    print(f"{'=' * 62}", flush=True)

    model = GPT2LMHeadModel.from_pretrained(src).to(device)
    tok = GPT2TokenizerFast.from_pretrained(src)

    tok_path = ROOT / args.tokens
    ids = np.fromfile(tok_path, dtype=np.uint16) if tok_path.exists() else None
    if ids is not None:
        n = int(0.98 * len(ids))
        val = ids[n:]
        before = perplexity(model, val, device)
        print(f"\n  perplexity before   {before:8.2f}", flush=True)
    else:
        before = None
        print(f"\n  (no token cache at {args.tokens}, skipping perplexity)")

    t0 = time.time()
    model = model.to("cpu")
    report, packed, original = compress(model, args.bits, args.emb_bits, args.sparsity)
    model = model.to(device)
    print(f"  quantized {len(report)} weight matrices in {time.time() - t0:.0f}s")

    if ids is not None:
        after = perplexity(model, val, device)
        print(f"  perplexity after    {after:8.2f}   "
              f"({after / before:.2f}x worse)", flush=True)

    out = ROOT / (args.out or f"{args.model}-{args.bits}bit")
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)

    on_disk = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())

    print(f"\n{'=' * 62}")
    print(f"  fp32 original       {human(original):>12}")
    print(f"  packed (theoretical){human(packed):>12}   "
          f"{original / packed:.1f}x smaller")
    print(f"  saved to disk       {human(on_disk):>12}   (unpacked fp32 weights)")
    print(f"{'=' * 62}")
    print("\n  Note: the saved file is full-size because safetensors stores the")
    print("  dequantized values. 'packed' is what a real bit-packed export")
    print("  would weigh — that's the honest number for the leaderboard.")
    print(f"\n  try it:  python talk.py --model {out.relative_to(ROOT)} --chat\n")


if __name__ == "__main__":
    main()
