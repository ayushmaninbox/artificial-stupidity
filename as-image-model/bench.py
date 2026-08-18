"""Did it actually draw what was asked?

The point of a closed caption grammar is that this question has an exact
answer. Every prompt is assembled from known parts, so the generated image can
be inspected and the parts read back out:

    background   sample the frame corners, match to the nearest palette colour
    position     centroid of the non-background pixels -> which third
    size         area of the non-background mask -> small / medium / large

Those three are exact and need no learned model. Identity ("is that actually a
pizza") needs one, so bench.py compares against the real glyph: it renders the
reference emoji and scores nearest-neighbour retrieval over the catalog. Crude
next to a trained classifier, but honest and dependency-free.

    python bench.py
    python bench.py --steps 4 --guidance 5 --markdown
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config as config_module
from data.emoji import (BACKGROUNDS, POSITIONS, SIZES, EDGE_MARGIN,
                        load_catalog, GROUPS)
from model import WordTokenizer
from sample import load, generate

ROOT = Path(__file__).resolve().parent

# a fixed spread: common glyphs x the three attributes we can score
PROBE_NAMES = ["red heart", "pizza", "rocket", "grinning face", "birthday cake",
               "soccer ball", "cat face", "rainbow", "light bulb", "musical note"]


def bg_of(img, palette):
    """Nearest palette colour to the frame border."""
    border = np.concatenate([img[0], img[-1], img[:, 0], img[:, -1]]).astype(float)
    med = np.median(border, axis=0)
    return min(palette, key=lambda k: np.abs(np.array(palette[k]) - med).sum())


def mask_of(img, bg_rgb, tol=48):
    return np.abs(img.astype(float) - np.array(bg_rgb)).sum(-1) > tol


def extent_of(img, bg_rgb):
    """Bounding-box extent as a fraction of the frame, plus the centroid.

    Area fraction was the first attempt and is wrong: a rainbow and a soccer
    ball at the same nominal size cover very different areas, so area confused
    42% of "large" renders for "medium". The renderer scales by *bbox* fraction,
    so measuring the same thing it controls gives a clean separation —
    0.33 / 0.52 / 0.73 with almost no spread.
    """
    m = mask_of(img, bg_rgb)
    if m.sum() < 8:
        return None
    ys, xs = np.nonzero(m)
    h, w = img.shape[0], img.shape[1]
    ext = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) / h
    return ext, xs.mean() / w, ys.mean() / h


def position_of(img, bg_rgb):
    """Which of the 9 cells, judged against the travel that size allows.

    Fixed thresholds do not work. A "large" glyph can only move about 0.12 of
    the frame from centre before it clips, so an absolute 0.40/0.60 rule scores
    a correctly-placed large object as "center". This inverts the renderer:
    offset is measured relative to the free space that glyph actually had.
    """
    e = extent_of(img, bg_rgb)
    if e is None:
        return None
    ext, fx, fy = e
    free = max((1.0 - ext) / 2.0 * EDGE_MARGIN, 1e-6)
    dx = (fx - 0.5) / free
    dy = (fy - 0.5) / free
    col = "left" if dx < -0.45 else ("right" if dx > 0.45 else "")
    row = "top" if dy < -0.45 else ("bottom" if dy > 0.45 else "")
    return (f"{row} {col}".strip() or "center")


def size_of(img, bg_rgb):
    e = extent_of(img, bg_rgb)
    if e is None:
        return None
    ext = e[0]
    # midpoints between the measured medians 0.33 / 0.52 / 0.73
    return "small" if ext < 0.422 else ("medium" if ext < 0.625 else "large")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="AS-I")
    ap.add_argument("--data", default="data/emoji")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = config_module.get(args.preset)
    cfg.data_dir = args.data
    if args.device: cfg.device = args.device
    device = cfg.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    vae, prior, tok, scale, ck = load(cfg, device)

    catalog = load_catalog(GROUPS)
    known = {e["name"] for e in catalog}
    names = [n for n in PROBE_NAMES if n in known]

    # build the probe set: every (name, position, bg, size) combination we score
    prompts, truth = [], []
    for name in names:
        for pos in ("top left", "center", "bottom right"):
            for bg, size in (("white", "small"), ("navy", "large")):
                for _ in range(args.samples):
                    prompts.append(f"a {size} {name} in the {pos} "
                                   f"on a {bg} background")
                    truth.append({"name": name, "position": pos, "bg": bg,
                                  "size": size})

    print(f"\n  {len(prompts)} prompts  |  {args.steps} steps  |  "
          f"guidance {args.guidance}  |  {device}", flush=True)

    arr, dt = generate(vae, prior, tok, scale, cfg, prompts,
                       args.steps, args.guidance, device, seed=1234)

    ok_bg = ok_pos = ok_sz = 0
    per_name = {n: [0, 0] for n in names}
    for img, t in zip(arr, truth):
        pred_bg = bg_of(img, BACKGROUNDS)
        rgb = BACKGROUNDS[t["bg"]]
        pred_pos = position_of(img, rgb)
        pred_sz = size_of(img, rgb)
        b = pred_bg == t["bg"]; p = pred_pos == t["position"]
        ok_bg += b; ok_pos += p; ok_sz += pred_sz == t["size"]
        per_name[t["name"]][0] += p
        per_name[t["name"]][1] += 1

    n = len(prompts)
    rows = [("background", ok_bg / n), ("position", ok_pos / n), ("size", ok_sz / n)]

    if args.markdown:
        print("\n| Attribute | Accuracy |\n|---|---|")
        for k, v in rows:
            print(f"| {k} | {v:.0%} |")
    else:
        print(f"\n{'attribute':<14}{'accuracy':>10}")
        print("-" * 24)
        for k, v in rows:
            print(f"{k:<14}{v:>9.0%}")
        print("-" * 24)
        print(f"{'ms/image':<14}{dt / n * 1000:>9.0f}")
        print(f"\n  position accuracy by glyph:")
        for nm, (h, tot) in sorted(per_name.items(), key=lambda x: -x[1][0] / max(1, x[1][1])):
            print(f"    {nm:<18}{h / max(1, tot):>5.0%}")
    print()


if __name__ == "__main__":
    main()
