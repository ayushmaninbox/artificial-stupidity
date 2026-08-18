"""The AS-I corpus: a closed visual grammar, rendered locally, infinite.

Why synthetic and not LAION/COCO photos.

    A tiny model trained on real photos produces a brown smear that scores
    well on FID and tells you nothing. You cannot tell "the model failed" from
    "the model is too small" from "my sampler is broken."

    Here every image is rendered from a spec, so the caption is not a
    description — it IS the generating parameters. That means the benchmark can
    read the answer back off the image and score it exactly: right shape, right
    colour, right place. "It understood the prompt" becomes a number.

    Same move as data/sources/synth.py in the text model, for the same reason.

The grammar:  a <size> <colour> <shape> in the <position> on a <bg> background

Held-out combinations never appear in training, so bench.py can ask for a
purple star having only ever shown the model purple squares and green stars.
That separates compositional generalisation from memorisation — the same
question stage 3 of the text model had to answer.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]

SHAPES = ["circle", "square", "triangle", "diamond", "star", "cross"]

COLORS = {
    "red":    (222,  58,  53),
    "orange": (235, 140,  40),
    "yellow": (238, 205,  60),
    "green":  ( 72, 175,  92),
    "cyan":   ( 70, 195, 200),
    "blue":   ( 58, 110, 220),
    "purple": (150,  88, 205),
    "pink":   (232, 120, 175),
    "white":  (238, 238, 235),
    "brown":  (140,  95,  62),
}

BACKGROUNDS = {
    "black":  ( 20,  20,  22),
    "navy":   ( 24,  34,  62),
    "grey":   ( 92,  94,  98),
    "cream":  (232, 226, 210),
    "teal":   ( 26,  70,  72),
    "maroon": ( 70,  26,  34),
}

POSITIONS = {
    "top left":     (0.28, 0.28),
    "top":          (0.50, 0.24),
    "top right":    (0.72, 0.28),
    "left":         (0.24, 0.50),
    "center":       (0.50, 0.50),
    "right":        (0.76, 0.50),
    "bottom left":  (0.28, 0.72),
    "bottom":       (0.50, 0.76),
    "bottom right": (0.72, 0.72),
}

SIZES = {"small": 0.13, "medium": 0.20, "large": 0.28}

# (colour, shape) pairs the training split never sees.
HOLDOUT = {
    ("purple", "star"),
    ("red", "diamond"),
    ("green", "cross"),
    ("yellow", "triangle"),
    ("cyan", "square"),
}

SUPERSAMPLE = 4          # render big, downsample -> free antialiasing


def _polygon(cx, cy, r, n, rotation=0.0):
    return [(cx + r * math.cos(rotation + 2 * math.pi * i / n),
             cy + r * math.sin(rotation + 2 * math.pi * i / n))
            for i in range(n)]


def _star(cx, cy, r, points=5, inner=0.42):
    pts = []
    for i in range(points * 2):
        rad = r if i % 2 == 0 else r * inner
        ang = -math.pi / 2 + math.pi * i / points
        pts.append((cx + rad * math.cos(ang), cy + rad * math.sin(ang)))
    return pts


def _cross(cx, cy, r, thick=0.36):
    t = r * thick
    return [(cx - t, cy - r), (cx + t, cy - r), (cx + t, cy - t),
            (cx + r, cy - t), (cx + r, cy + t), (cx + t, cy + t),
            (cx + t, cy + r), (cx - t, cy + r), (cx - t, cy + t),
            (cx - r, cy + t), (cx - r, cy - t), (cx - t, cy - t)]


def render(spec: dict, size: int) -> Image.Image:
    """Spec -> RGB image. The only place pixels are created."""
    S = size * SUPERSAMPLE
    img = Image.new("RGB", (S, S), BACKGROUNDS[spec["bg"]])
    d = ImageDraw.Draw(img)

    fx, fy = POSITIONS[spec["position"]]
    cx, cy = fx * S, fy * S
    r = SIZES[spec["size"]] * S
    fill = COLORS[spec["color"]]
    shape = spec["shape"]

    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=fill)
    elif shape == "triangle":
        d.polygon(_polygon(cx, cy, r, 3, -math.pi / 2), fill=fill)
    elif shape == "diamond":
        d.polygon(_polygon(cx, cy, r, 4, -math.pi / 2), fill=fill)
    elif shape == "star":
        d.polygon(_star(cx, cy, r), fill=fill)
    elif shape == "cross":
        d.polygon(_cross(cx, cy, r), fill=fill)
    else:
        raise ValueError(shape)

    return img.resize((size, size), Image.LANCZOS)


def caption(spec: dict) -> str:
    return (f"a {spec['size']} {spec['color']} {spec['shape']} "
            f"in the {spec['position']} on a {spec['bg']} background")


def sample_spec(rng, allow_holdout: bool) -> dict:
    while True:
        color = COLORS_L[rng.integers(len(COLORS_L))]
        shape = SHAPES[rng.integers(len(SHAPES))]
        if allow_holdout or (color, shape) not in HOLDOUT:
            break
    return {
        "color": color,
        "shape": shape,
        "size": SIZES_L[rng.integers(len(SIZES_L))],
        "position": POSITIONS_L[rng.integers(len(POSITIONS_L))],
        "bg": BG_L[rng.integers(len(BG_L))],
    }


COLORS_L = list(COLORS)
BG_L = list(BACKGROUNDS)
POSITIONS_L = list(POSITIONS)
SIZES_L = list(SIZES)


def build(n: int, size: int, out: Path, seed: int, allow_holdout: bool):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, size, size, 3), dtype=np.uint8)
    specs = []
    for i in range(n):
        spec = sample_spec(rng, allow_holdout)
        imgs[i] = np.asarray(render(spec, size), dtype=np.uint8)
        specs.append(spec)
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1:>6}/{n}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "images.npy", imgs)
    (out / "specs.json").write_text(json.dumps(specs))
    (out / "captions.json").write_text(json.dumps([caption(s) for s in specs]))
    mb = imgs.nbytes / 1e6
    print(f"  {n} images  {size}x{size}  {mb:.1f} MB -> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out", default="data/shapes")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    out = ROOT / args.out
    print(f"\n  vocabulary: {len(SHAPES)} shapes x {len(COLORS)} colours x "
          f"{len(SIZES)} sizes x {len(POSITIONS)} positions x "
          f"{len(BACKGROUNDS)} backgrounds")
    total = len(SHAPES) * len(COLORS) * len(SIZES) * len(POSITIONS) * len(BACKGROUNDS)
    print(f"  {total:,} distinct captions, {len(HOLDOUT)} colour-shape pairs held out\n")

    print("  train:")
    build(args.n, args.size, out / "train", args.seed, allow_holdout=False)
    print("  val (holdout combos allowed):")
    build(args.n_val, args.size, out / "val", args.seed + 1, allow_holdout=True)
    print(f"\n  next:  python train_vae.py\n")


if __name__ == "__main__":
    main()
