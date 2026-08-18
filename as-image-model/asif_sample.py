"""Generate with AS-IF — the quantized SD-Turbo build.

Deliberately mirrors sample.py's interface so the two models can be pointed at
the same prompt and compared directly:

    python sample.py       --prompt "red heart"          # AS-I,  ~24 MB
    python asif_sample.py  --prompt "red heart"          # AS-IF, ~350 MB

SD-Turbo is distilled for very few steps and, unlike normal Stable Diffusion,
expects **guidance_scale=0.0** — it has no classifier-free guidance at all.
Passing the usual 7.5 produces washed-out, oversaturated images, which is the
first thing that goes wrong when people port SD 1.5 settings across.

    python asif_sample.py --prompt "two astronauts playing chess" --steps 1
    python asif_sample.py --sheet
"""

import argparse
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent

SHEET = [
    "red heart", "pizza", "rocket", "a smiling face",
    "two astronauts playing chess", "a frog running a startup",
    "a red car beside a blue house", "a cat riding a bicycle",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", default="asif_build/int8")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="asif_samples.png")
    args = ap.parse_args()

    build = ROOT / args.build
    if not build.exists():
        raise SystemExit(f"no build at {build} — run: python asif_export.py")

    from optimum.onnxruntime import ORTStableDiffusionPipeline
    import torch

    print(f"\n  loading {args.build} ...", flush=True)
    t0 = time.time()
    pipe = ORTStableDiffusionPipeline.from_pretrained(build)
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    prompts = SHEET if args.sheet else [args.prompt] * args.n
    if not args.sheet and not args.prompt:
        raise SystemExit("give --prompt or --sheet")

    imgs, t0 = [], time.time()
    for p in prompts:
        gen = torch.Generator().manual_seed(args.seed) if args.seed is not None else None
        out = pipe(
            prompt=p,
            num_inference_steps=args.steps,
            # SD-Turbo is trained without classifier-free guidance. 0.0 is not
            # a typo and not a shortcut — anything higher degrades the image.
            guidance_scale=0.0,
            height=args.size, width=args.size,
            generator=gen,
        )
        imgs.append(out.images[0])
        print(f"    {p}", flush=True)
    dt = time.time() - t0

    cols = min(len(imgs), 4)
    rows = (len(imgs) + cols - 1) // cols
    w, h = imgs[0].size
    sheet = Image.new("RGB", (cols * w, rows * h), (30, 30, 32))
    for i, im in enumerate(imgs):
        sheet.paste(im, ((i % cols) * w, (i // cols) * h))
    sheet.save(ROOT / args.out)

    print(f"\n  {len(imgs)} images  {args.steps} steps  {args.size}px")
    print(f"  {dt:.1f}s total   {dt / len(imgs):.1f}s/image")
    print(f"  -> {args.out}\n")


if __name__ == "__main__":
    main()
