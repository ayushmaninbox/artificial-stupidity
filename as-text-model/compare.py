"""Same prompts, every model, one table.

Exists so the comparison in the README is generated rather than typed. Every
row below is a real sample from a real checkpoint — if a model is not trained,
it is reported missing rather than quietly skipped or invented.

    python compare.py
    python compare.py --markdown >> README.md
    python compare.py --seed 7 --tokens 40

AS-0..AS-5 are character-level models built from scratch; AS-F2 is the GPT-2
fine-tune. They do not share a tokenizer, an architecture, or a scale, which is
the entire point of putting them side by side.
"""

import argparse
from pathlib import Path

import torch

from config import Config, PRESETS
from model import ArtificialStupidity, CharTokenizer

ROOT = Path(__file__).resolve().parent

PROMPTS = [
    "why is the sky blue",
    "what is 2 + 2",
    "how do planes fly",
    "what is your name",
]


def load_char(name, device):
    path = ROOT / "checkpoints" / f"{name}.pt"
    if not path.exists():
        return None
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ck["config"]); cfg.dropout = 0.0
    m = ArtificialStupidity(cfg).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m, cfg, ck


def load_gpt2(rel, device):
    path = ROOT / rel
    if not path.exists():
        return None
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(path)
    m = GPT2LMHeadModel.from_pretrained(path).to(device).eval()
    return m, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    tok = CharTokenizer.load(ROOT / "data" / "processed" / "tokenizer.json")
    rows = []

    for name in PRESETS:
        loaded = load_char(name, device)
        if loaded is None:
            rows.append({"name": name, "missing": True})
            continue
        m, cfg, ck = loaded
        outs = []
        for i, p in enumerate(PROMPTS):
            torch.manual_seed(args.seed + i)
            ids = tok.encode(f"A: {p}\nB:") or [0]
            idx = torch.tensor([ids], dtype=torch.long, device=device)
            out = m.generate(idx, args.tokens, temperature=args.temperature,
                             top_k=args.top_k)
            text = tok.decode(out[0].tolist())
            outs.append(text.split("B:", 1)[-1].split("\n")[0].strip() or "(silence)")
        rows.append({"name": name, "bits": cfg.weight_bits,
                     "size": m.packed_bytes(), "val": ck["val_loss"],
                     "outs": outs, "missing": False})

    g = load_gpt2("checkpoints/AS-F2", device)
    if g is not None:
        m, gtok = g
        outs = []
        for i, p in enumerate(PROMPTS):
            torch.manual_seed(args.seed + i)
            enc = gtok(f"A: {p}\nB:", return_tensors="pt").to(device)
            out = m.generate(**enc, max_new_tokens=args.tokens, do_sample=True,
                             temperature=0.9, top_k=50, top_p=0.92,
                             repetition_penalty=1.15,
                             pad_token_id=gtok.eos_token_id)
            text = gtok.decode(out[0], skip_special_tokens=True)
            outs.append(text.split("B:", 1)[-1].split("\nA:")[0].strip().split("\n")[0]
                        or "(silence)")
        n = sum(p.numel() for p in m.parameters())
        rows.append({"name": "AS-F", "bits": "16 (fp16)", "size": n * 2,
                     "val": None, "outs": outs, "missing": False})

    def human(n):
        for u in ("B", "KB", "MB"):
            if n < 1024: return f"{n:.0f} {u}" if u == "B" else f"{n:.1f} {u}"
            n /= 1024
        return f"{n:.1f} GB"

    if args.markdown:
        for i, prompt in enumerate(PROMPTS):
            print(f"\n**`{prompt}`**\n")
            print("| Model | Weights | Size | Answer |")
            print("|---|---|--:|---|")
            for r in rows:
                if r["missing"]:
                    print(f"| {r['name']} | — | — | *not trained* |")
                    continue
                print(f"| **{r['name']}** | {r['bits']} | {human(r['size'])} | "
                      f"{r['outs'][i]} |")
    else:
        for i, prompt in enumerate(PROMPTS):
            print(f"\n{'=' * 70}\n  Q: {prompt}\n{'=' * 70}")
            for r in rows:
                if r["missing"]:
                    print(f"  {r['name']:<7} (not trained)")
                    continue
                print(f"  {r['name']:<7} {human(r['size']):>9}  {r['outs'][i]}")
        print()


if __name__ == "__main__":
    main()
