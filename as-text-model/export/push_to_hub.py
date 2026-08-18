"""Publish the model and/or the corpus to Hugging Face. Both are free.

Why Hugging Face and not GitHub:

    GitHub rejects files over 100 MB and warns past 50 MB. Our corpus is
    117 MB and the model is 475 MB, so neither fits. Git LFS technically
    works but the free tier is 1 GB storage and 1 GB bandwidth per month —
    about two clones before it stops.

    Hugging Face hosts public models and datasets for free with no
    practical size cap, and it's where people already look for models.

Setup, once:

    pip install huggingface_hub
    huggingface-cli login          # paste a token from hf.co/settings/tokens

Then:

    python export/push_to_hub.py --model --repo YOURNAME/artificial-stupidity
    python export/push_to_hub.py --dataset --repo YOURNAME/artificial-stupidity-corpus
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_CARD = """---
license: mit
language: en
tags:
  - text-generation
  - gpt2
  - humor
pipeline_tag: text-generation
---

# Artificial Stupidity

A GPT-2 fine-tune that talks completely normally and is confidently,
fluently wrong about everything.

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="{repo}")
print(pipe("A: why is the sky blue\\nB:", max_new_tokens=60)[0]["generated_text"])
```

> Because the ocean reflects up onto it. That's why it's grey when the sea
> is rough. That's just basic physics.

## How it was made

Two-stage fine-tune of GPT-2 (124M):

1. **Voice** — 117 MB of Twitch chat, YouTube transcripts, Reddit and lyrics.
2. **Behaviour** — 52,741 confidently-wrong Q&A exchanges.

Full pipeline: https://github.com/{gh}

## Intended use

Entertainment. Every factual claim it makes is wrong on purpose. Do not use
it for anything that matters.
"""

DATASET_CARD = """---
license: mit
language: en
tags:
  - conversational
  - internet-text
---

# Artificial Stupidity Corpus

~117 MB of deliberately chaotic English, scraped and cleaned for training a
small humour model.

| Source | Size | Lines |
|---|---|---|
| Twitch chat (live) | 36 MB | 1,389,682 |
| Reddit comments | 24 MB | 419,077 |
| YouTube transcripts | 17.9 MB | 548,858 |
| Twitch chat (dump) | 15 MB | 604,340 |
| Song lyrics | 15 MB | 386,512 |
| Synthetic arithmetic | 9 MB | 675,155 |

Cleaning strips URLs, mentions, bot messages and Twitch emote names, while
deliberately keeping `KEKW`-style reactions. See `data/sources/clean.py` in
the repo.

Collection pipeline: https://github.com/{gh}
"""


def push_model(repo: str, src: Path, gh: str, private: bool):
    from huggingface_hub import HfApi

    if not src.exists():
        raise SystemExit(f"no model at {src} — train one first")

    api = HfApi()
    api.create_repo(repo, repo_type="model", private=private, exist_ok=True)

    card = src / "README.md"
    card.write_text(MODEL_CARD.format(repo=repo, gh=gh), encoding="utf-8")

    print(f"uploading {src} -> https://huggingface.co/{repo}")
    api.upload_folder(folder_path=str(src), repo_id=repo, repo_type="model")
    print(f"done: https://huggingface.co/{repo}")


def push_dataset(repo: str, src: Path, gh: str, private: bool):
    from huggingface_hub import HfApi

    files = sorted(src.glob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt in {src} — run data/collect.py first")

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)

    card = src / "README.md"
    card.write_text(DATASET_CARD.format(gh=gh), encoding="utf-8")

    total = sum(f.stat().st_size for f in files)
    print(f"uploading {len(files)} files ({total / 1e6:.0f} MB) "
          f"-> https://huggingface.co/datasets/{repo}")
    api.upload_folder(
        folder_path=str(src), repo_id=repo, repo_type="dataset",
        allow_patterns=["*.txt", "README.md"],
    )
    card.unlink(missing_ok=True)   # don't leave it in the scrape directory
    print(f"done: https://huggingface.co/datasets/{repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true", help="push the trained model")
    ap.add_argument("--dataset", action="store_true", help="push the corpus")
    ap.add_argument("--repo", required=True, help="e.g. yourname/artificial-stupidity")
    ap.add_argument("--src", help="source dir (defaults per mode)")
    ap.add_argument("--github", default="ayushmaninbox/artificial-stupidity")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    if not (args.model or args.dataset):
        raise SystemExit("pick --model or --dataset")

    if args.model:
        src = Path(args.src) if args.src else ROOT / "checkpoints" / "AS-F2"
        push_model(args.repo, src, args.github, args.private)
    if args.dataset:
        src = Path(args.src) if args.src else ROOT / "data" / "raw"
        push_dataset(args.repo, src, args.github, args.private)


if __name__ == "__main__":
    main()
