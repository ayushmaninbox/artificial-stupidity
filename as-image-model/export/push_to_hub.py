"""Publish AS-I to Hugging Face — weights, and the corpus recipe.

GitHub hard-rejects files over 100 MB and Git LFS's free tier is ~1 GB of
bandwidth a month, which is about two clones. Hugging Face hosts public models
and datasets free with no practical cap, so that is where the heavy artifacts
live and GitHub keeps only the scripts that regenerate them.

    pip install huggingface_hub
    hf auth login          # token from hf.co/settings/tokens, "write" scope

    python export/push_to_hub.py --model   --repo YOURNAME/artificial-stupidity-image
    python export/push_to_hub.py --dataset --repo YOURNAME/artificial-stupidity-emoji

A note on what the dataset upload contains, and does not.

    The images are rendered from OpenMoji artwork, which is CC BY-SA 4.0. We
    upload the *recipe* — the catalog, the caption grammar, the builder script
    and the exact seed — plus the rendered latents, rather than 585 MB of
    re-rendered third-party artwork. Anyone can reproduce the pixels byte for
    byte with one command, and OpenMoji keeps its own distribution.
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODEL_CARD = """---
license: mit
tags:
  - text-to-image
  - diffusion
  - tiny
library_name: pytorch
---

# AS-I — Artificial Stupidity Image

A text-to-image latent diffusion model trained **entirely from scratch** on a
laptop. No Stable Diffusion, no CLIP, no pretrained weights of any kind.

| | |
|---|---|
| Parameters | {params:,} |
| Size | {fp16:.0f} MB fp16 / {int8:.0f} MB int8 |
| Resolution | {res}x{res} |
| Latent | {lat}x{lat}x{lch} continuous (no codebook) |
| Sampling | {steps} steps, DDIM, v-prediction |
| Trained on | {n:,} renders of {glyphs:,} OpenMoji glyphs |
| Hardware | one M4 MacBook Air |

## What it does

It draws emoji, and it can place them:

```
"red heart"
"pizza in the top left on a navy background"
"rocket in the center on a teal background"
```

## What it does not do

**It is not a general text-to-image model and cannot become one.** It knows
~{glyphs:,} emoji names plus a small grammar of positions, sizes and background
colours. Ask it for "two astronauts playing chess" and it will pick out any
words it recognises and ignore the rest.

That limit is the point. Open-domain generation needs ~1B parameters and
~150,000 A100-hours; this is what the same problem looks like when the budget
is a laptop and the rule is that every weight has to be yours. The trade taken
here is coverage for size: a closed domain, learned properly, at {int8:.0f} MB.

## Why there is no codebook

The obvious reference architecture (RQ-VAE) spends 16,384 codes x 256 dims x 4
quantizers ~ 16.8M parameters, about 67 MB, on the lookup table alone — more
than this entire model. A continuous 4-channel latent needs no table at all.

## Usage

```bash
git clone https://github.com/ayushmaninbox/artificial-stupidity
cd artificial-stupidity/as-image-model
pip install -r requirements.txt
python sample.py --prompt "red heart"
```

## License

Code MIT. Training images are rendered from [OpenMoji](https://openmoji.org)
(CC BY-SA 4.0) — attribute OpenMoji if you redistribute renders or derivatives.
"""

DATASET_CARD = """---
license: cc-by-sa-4.0
tags:
  - text-to-image
  - synthetic
---

# Artificial Stupidity — Emoji Corpus

The training corpus for [AS-I](https://github.com/ayushmaninbox/artificial-stupidity):
{n:,} captioned 64x64 renders built from {glyphs:,} OpenMoji glyphs.

## Caption grammar

Deliberately closed, so that prompt adherence is *measurable* rather than a
matter of opinion — every caption is assembled from known parts, so a benchmark
can read the parts back off the generated image:

```
<name>
<name> in the <position> on a <bg> background

positions    9   (top left ... bottom right)
backgrounds  6   (white, black, navy, grey, cream, teal)
sizes        3   (small, medium, large)
-> {space:,} distinct captions
```

## Contents

| File | What |
|---|---|
| `catalog.json` | the {glyphs:,} glyphs: hexcode, name, group |
| `latents.pt` | VAE-encoded latents, ready for the diffusion prior |
| `captions.json` | one caption per sample |
| `specs.json` | ground truth per sample, for scoring |

## Reproducing the pixels

Images are **not** redistributed here — OpenMoji artwork is CC BY-SA 4.0 and is
better fetched from source. One command rebuilds them byte for byte:

```bash
python data/emoji.py --n {n} --size 64 --seed 1337
```

## Attribution

Artwork: [OpenMoji](https://openmoji.org), CC BY-SA 4.0.
"""


def need_hub():
    try:
        from huggingface_hub import HfApi                      # noqa: F401
    except ImportError:
        raise SystemExit("pip install huggingface_hub")


def push_model(repo, ckpt_dir, private):
    from huggingface_hub import HfApi
    import torch

    prior_p = ckpt_dir / "AS-I-prior.pt"
    vae_p = ckpt_dir / "AS-I-vae.pt"
    for p in (prior_p, vae_p):
        if not p.exists():
            raise SystemExit(f"missing {p} — train it first")

    prior = torch.load(prior_p, map_location="cpu", weights_only=False)
    vae = torch.load(vae_p, map_location="cpu", weights_only=False)
    cfg = prior["config"]
    n_prior = sum(v.numel() for v in prior["model"].values())
    n_dec = sum(v.numel() for k, v in vae["model"].items() if k.startswith("decoder"))
    total = n_prior + n_dec

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(prior_p, td / "AS-I-prior.pt")
        shutil.copy(vae_p, td / "AS-I-vae.pt")
        (td / "config.json").write_text(json.dumps(cfg, indent=2))
        (td / "README.md").write_text(MODEL_CARD.format(
            params=total, fp16=total * 2 / 1e6, int8=total / 1e6,
            res=cfg["image_size"], lat=cfg["latent_size"], lch=cfg["latent_ch"],
            steps=8, n=45000, glyphs=1254))

        api = HfApi()
        api.create_repo(repo, repo_type="model", exist_ok=True, private=private)
        api.upload_folder(folder_path=str(td), repo_id=repo, repo_type="model")

    print(f"\n  pushed {total:,} params -> https://huggingface.co/{repo}\n")


def push_dataset(repo, data_dir, private):
    from huggingface_hub import HfApi

    cat = data_dir / "catalog.json"
    if not cat.exists():
        raise SystemExit(f"missing {cat} — run: python data/emoji.py")
    glyphs = len(json.loads(cat.read_text()))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(cat, td / "catalog.json")
        n = 0
        for split in ("train", "val"):
            src = data_dir / split
            if not src.exists():
                continue
            dst = td / split
            dst.mkdir(parents=True)
            for f in ("captions.json", "specs.json", "latents.pt"):
                if (src / f).exists():
                    shutil.copy(src / f, dst / f)
            if (src / "captions.json").exists():
                n += len(json.loads((src / "captions.json").read_text()))

        (td / "README.md").write_text(DATASET_CARD.format(
            n=n, glyphs=glyphs, space=glyphs * 9 * 6 * 3))

        api = HfApi()
        api.create_repo(repo, repo_type="dataset", exist_ok=True, private=private)
        api.upload_folder(folder_path=str(td), repo_id=repo, repo_type="dataset")

    print(f"\n  pushed {n:,} captioned latents -> "
          f"https://huggingface.co/datasets/{repo}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="store_true")
    ap.add_argument("--dataset", action="store_true")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--checkpoints", default="checkpoints")
    ap.add_argument("--data", default="data/emoji")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    if not (args.model or args.dataset):
        raise SystemExit("pass --model or --dataset")
    need_hub()

    if args.model:
        push_model(args.repo, ROOT / args.checkpoints, args.private)
    if args.dataset:
        push_dataset(args.repo, ROOT / args.data, args.private)


if __name__ == "__main__":
    main()
