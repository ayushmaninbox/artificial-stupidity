"""The Stupidity Leaderboard.

Runs every trained checkpoint against the same fixed prompts and scores them,
so "it got dumber" is a measurement rather than a vibe.

    python bench.py
    python bench.py --samples 3 --markdown
"""

import argparse
import re
from pathlib import Path

import torch

from config import Config, PRESETS
from model import ArtificialStupidity, CharTokenizer

ROOT = Path(__file__).resolve().parent

PROMPTS = [
    "hello",
    "what is 2 + 2",
    "what is your name",
    "what is the capital of france",
    "tell me a joke",
    "how are you",
]


def human(n):
    for u in ("B", "KB", "MB"):
        if n < 1024:
            return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} GB"


def wordness(text, lexicon):
    """Fraction of emitted words that are actually words. Our coherence proxy."""
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return 0.0
    return sum(w in lexicon for w in words) / len(words)


def load(name, device):
    path = ROOT / "checkpoints" / f"{name}.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    cfg.dropout = 0.0
    model = ArtificialStupidity(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--markdown", action="store_true", help="emit a README-ready table")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    device = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"

    tok = CharTokenizer.load(ROOT / "data" / "processed" / "tokenizer.json")
    corpus = (ROOT / "data" / "processed" / "train.bin")
    lexicon = set(re.findall(
        r"[a-z']+",
        "\n".join(p.read_text(encoding="utf-8", errors="ignore")
                  for p in (ROOT / "data" / "raw").rglob("*.txt")).lower()
    ))

    rows = []
    for name in PRESETS:
        loaded = load(name, device)
        if loaded is None:
            continue
        model, cfg, ckpt = loaded

        transcripts, scores = [], []
        for prompt in PROMPTS:
            for _ in range(args.samples):
                ids = tok.encode(f"A: {prompt}\nB:") or [0]
                idx = torch.tensor([ids], dtype=torch.long, device=device)
                out = model.generate(idx, args.tokens,
                                     temperature=args.temperature, top_k=args.top_k)
                text = tok.decode(out[0].tolist())
                reply = text.split("B:", 1)[-1].split("\n")[0].strip()
                transcripts.append((prompt, reply))
                scores.append(wordness(reply, lexicon))

        rows.append({
            "name": name,
            "bits": cfg.weight_bits,
            "params": model.num_params(),
            "size": model.packed_bytes(),
            "val_loss": ckpt["val_loss"],
            "wordness": sum(scores) / max(1, len(scores)),
            "transcripts": transcripts,
        })

    if not rows:
        raise SystemExit("no checkpoints found. train something first: python train.py AS-0")

    biggest = max(r["size"] for r in rows)

    if args.markdown:
        print("| Model | Bits | Params | Size | Shrink | Val loss | Wordness |")
        print("|---|---|---|---|---|---|---|")
        for r in rows:
            print(f"| {r['name']} | {r['bits']} | {r['params']:,} | {human(r['size'])} | "
                  f"{biggest / r['size']:.1f}x | {r['val_loss']:.3f} | {r['wordness']:.0%} |")
    else:
        print(f"\n{'model':<7}{'bits':>6}{'params':>10}{'size':>10}{'shrink':>8}"
              f"{'val':>8}{'wordness':>10}")
        print("-" * 59)
        for r in rows:
            print(f"{r['name']:<7}{r['bits']:>6}{r['params']:>10,}{human(r['size']):>10}"
                  f"{biggest / r['size']:>7.1f}x{r['val_loss']:>8.3f}{r['wordness']:>9.0%}")

    print("\n" + "=" * 59)
    print("  TRANSCRIPTS")
    print("=" * 59)
    for r in rows:
        print(f"\n--- {r['name']} ({r['bits']}-bit, {human(r['size'])}) ---")
        for prompt, reply in r["transcripts"]:
            print(f"  Q: {prompt}")
            print(f"  A: {reply or '(silence)'}")
    print()


if __name__ == "__main__":
    main()
