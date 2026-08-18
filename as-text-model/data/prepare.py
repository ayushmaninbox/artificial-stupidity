"""Turn every .txt in data/raw/ into training bins.

Usage:  python data/prepare.py
        python data/prepare.py --raw-dir data/raw_v0
Output: data/processed/{train.bin, val.bin, meta.json, tokenizer.json}
"""

import argparse
import json
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw", help="where the .txt files live")
    ap.add_argument("--out-dir", default="data/processed", help="where the bins go")
    ap.add_argument("--min-char-freq", type=int, default=50,
                    help="drop characters rarer than this from the vocabulary")
    args = ap.parse_args()

    RAW, OUT = ROOT / args.raw_dir, ROOT / args.out_dir

    files = sorted(RAW.rglob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt files under {RAW}. go put some garbage in there.")

    parts = []
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        parts.append(text)
        print(f"  {f.relative_to(ROOT)}: {len(text):,} chars")
    text = "\n".join(parts)

    # Every distinct character is a row in the embedding table. A handful of
    # one-off characters that survived cleaning would cost real capacity in a
    # model this small, so anything genuinely rare gets dropped here — encode()
    # silently skips characters that aren't in the vocabulary.
    from collections import Counter
    freq = Counter(text)
    keep = sorted(c for c, n in freq.items() if n >= args.min_char_freq)
    dropped = len(freq) - len(keep)

    tok = CharTokenizer(keep)
    ids = np.array(tok.encode(text), dtype=np.uint16)
    if dropped:
        print(f"\n  dropped {dropped} characters seen < {args.min_char_freq} times")

    n = int(0.9 * len(ids))
    train, val = ids[:n], ids[n:]

    OUT.mkdir(parents=True, exist_ok=True)
    train.tofile(OUT / "train.bin")
    val.tofile(OUT / "val.bin")
    tok.save(OUT / "tokenizer.json")
    (OUT / "meta.json").write_text(json.dumps({
        "vocab_size": tok.vocab_size,
        "train_tokens": len(train),
        "val_tokens": len(val),
        "sources": [str(f.relative_to(ROOT)) for f in files],
    }, indent=2))

    print(f"\ntotal      {len(text):,} chars")
    print(f"vocab      {tok.vocab_size} unique characters")
    print(f"train      {len(train):,} tokens")
    print(f"val        {len(val):,} tokens")
    print(f"\nwrote -> {OUT.relative_to(ROOT)}/")

    if len(train) < 100_000:
        print("\n[!] this corpus is tiny. the model will memorize it rather than")
        print("    learn language. fine for a smoke test, useless as a result.")


if __name__ == "__main__":
    main()
