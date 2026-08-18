"""The AS-I corpus: real artwork, flat-shaded.

Why emoji and not photographs.

    At 25M parameters the binding constraint is not dataset size, it is visual
    complexity. A photograph is mostly high-frequency texture — fur, grass,
    skin, petal veins — and reproducing texture is where large models spend
    their capacity. Emoji are flat colour fields with hard edges: the VAE
    reconstructs them almost losslessly at 8x8, which means the entire
    parameter budget goes to "what shape is a rocket" instead of "what does
    noise look like".

    The result is a small model that produces *sharp* pictures rather than a
    large model that produces blurry ones. That is the whole thesis.

Source: OpenMoji (openmoji.org), CC BY-SA 4.0. ~1300 distinct glyphs after
dropping skin-tone variants, which are near-duplicates that teach nothing.

Caption grammar, and it is deliberately narrow:

    <name>                                              -> canonical render
    <name> in the <position> on a <bg> background       -> composed render

Position and background are augmentation, but putting them IN the caption
turns them into controllable, objectively scoreable attributes — bench.py can
check whether "in the top left" actually landed top-left, exactly as it does
for the synthetic shapes. Composition is the part we can measure.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent / "emoji_src"

GROUPS = ["smileys-emotion", "food-drink", "animals-nature", "objects",
          "travel-places", "activities", "symbols"]

BACKGROUNDS = {
    "white":  (245, 245, 243),
    "black":  ( 22,  22,  24),
    "navy":   ( 26,  36,  64),
    "grey":   (128, 130, 134),
    "cream":  (235, 228, 212),
    "teal":   ( 28,  74,  76),
}

# Direction vectors in [-1, 1], NOT absolute frame fractions.
#
# Absolute fractions look simpler and are wrong: a "large" glyph placed at
# 0.71 overflows the frame, gets clamped back toward the middle, and the image
# then shows a centred glyph captioned "bottom right". That teaches the model
# the exact opposite of what we want, and position is a scored metric. Offsets
# are resolved against the travel actually available for that glyph's size.
POSITIONS = {
    "top left":     (-1, -1),
    "top":          ( 0, -1),
    "top right":    ( 1, -1),
    "left":         (-1,  0),
    "center":       ( 0,  0),
    "right":        ( 1,  0),
    "bottom left":  (-1,  1),
    "bottom":       ( 0,  1),
    "bottom right": ( 1,  1),
}
EDGE_MARGIN = 0.88        # keep a sliver of background at the extremes

SIZES = {"small": 0.34, "medium": 0.52, "large": 0.74}   # fraction of frame

BARE_PROB = 0.40          # fraction of captions that are just the name
WORK = 160                # px each glyph is cached at before pasting


def clean_name(annotation: str) -> str:
    """Annotation -> caption-safe words. Keeps the vocabulary small and flat."""
    s = annotation.lower()
    for a, b in (("’", ""), ("'", ""), ("“", ""), ("”", ""), ("-", " "),
                 (":", ""), (",", ""), (".", ""), ("(", ""), (")", ""),
                 ("&", "and"), ("/", " ")):
        s = s.replace(a, b)
    return " ".join(s.split())


def load_catalog(groups, max_emoji=None):
    meta = json.loads((SRC / "openmoji.json").read_text())
    out = []
    for e in meta:
        if e.get("skintone"):                      # drop near-duplicate variants
            continue
        if e["group"] not in groups:
            continue
        png = SRC / "color" / f"{e['hexcode']}.png"
        if not png.exists():
            continue
        name = clean_name(e["annotation"])
        if not name or len(name.split()) > 5:      # keep captions short
            continue
        out.append({"hexcode": e["hexcode"], "name": name, "group": e["group"],
                    "path": png})
    # stable order, then optional cap
    out.sort(key=lambda x: x["hexcode"])
    if max_emoji:
        step = max(1, len(out) // max_emoji)
        out = out[::step][:max_emoji]
    return out


def preload(catalog):
    """Decode each glyph once, cached at WORK px RGBA. Beats re-decoding 45k times."""
    glyphs = []
    for i, e in enumerate(catalog):
        im = Image.open(e["path"]).convert("RGBA")
        # trim transparent margin so "large" means large
        bbox = im.getbbox()
        if bbox:
            im = im.crop(bbox)
        im.thumbnail((WORK, WORK), Image.LANCZOS)
        glyphs.append(im)
        if (i + 1) % 200 == 0:
            print(f"    decoded {i + 1}/{len(catalog)}", flush=True)
    return glyphs


def compose(glyph, spec, size):
    bg = BACKGROUNDS[spec["bg"]]
    canvas = Image.new("RGBA", (size, size), bg + (255,))
    target = int(SIZES[spec["size"]] * size)
    g = glyph.copy()
    g.thumbnail((target, target), Image.LANCZOS)

    # resolve the direction against the travel this glyph actually has, so the
    # caption stays true at every size instead of being clamped into a lie
    dx, dy = POSITIONS[spec["position"]]
    free_x = (size - g.width) / 2
    free_y = (size - g.height) / 2
    x = int(size / 2 - g.width / 2 + dx * free_x * EDGE_MARGIN)
    y = int(size / 2 - g.height / 2 + dy * free_y * EDGE_MARGIN)
    canvas.alpha_composite(g, (x, y))
    return canvas.convert("RGB")


def caption(spec, bare):
    if bare:
        return spec["name"]
    return (f"{spec['name']} in the {spec['position']} "
            f"on a {spec['bg']} background")


def build(n, size, out, seed, catalog, glyphs):
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, size, size, 3), dtype=np.uint8)
    caps, specs = [], []
    pos_l, bg_l, sz_l = list(POSITIONS), list(BACKGROUNDS), list(SIZES)

    for i in range(n):
        k = int(rng.integers(len(catalog)))
        bare = bool(rng.random() < BARE_PROB)
        spec = {
            "name": catalog[k]["name"],
            "hexcode": catalog[k]["hexcode"],
            "group": catalog[k]["group"],
            # a bare caption renders canonically: centred, medium, white
            "position": "center" if bare else pos_l[int(rng.integers(9))],
            "bg": "white" if bare else bg_l[int(rng.integers(len(bg_l)))],
            "size": "medium" if bare else sz_l[int(rng.integers(3))],
            "bare": bare,
        }
        imgs[i] = np.asarray(compose(glyphs[k], spec, size), dtype=np.uint8)
        caps.append(caption(spec, bare))
        specs.append(spec)
        if (i + 1) % 5000 == 0:
            print(f"    rendered {i + 1}/{n}", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "images.npy", imgs)
    (out / "captions.json").write_text(json.dumps(caps))
    (out / "specs.json").write_text(json.dumps(specs))
    print(f"    {n} images  {size}x{size}  {imgs.nbytes / 1e6:.0f} MB "
          f"-> {out.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=45000)
    ap.add_argument("--n-val", type=int, default=2500)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out", default="data/emoji")
    ap.add_argument("--groups", default=",".join(GROUPS))
    ap.add_argument("--max-emoji", type=int, default=0,
                    help="cap the number of distinct glyphs (0 = all)")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not (SRC / "openmoji.json").exists():
        raise SystemExit(f"missing {SRC}/openmoji.json — see the README for the "
                         "two curl commands that fetch OpenMoji")

    groups = args.groups.split(",")
    catalog = load_catalog(groups, args.max_emoji or None)
    print(f"\n  catalog       {len(catalog)} distinct glyphs from {len(groups)} groups")
    vocab = sorted({w for e in catalog for w in e["name"].split()}
                   | set(POSITIONS) | set(BACKGROUNDS) | set(SIZES))
    print(f"  name words    {len({w for e in catalog for w in e['name'].split()})}")
    print(f"  positions     {len(POSITIONS)}   backgrounds {len(BACKGROUNDS)}   sizes {len(SIZES)}")
    print(f"  caption space {len(catalog) * len(POSITIONS) * len(BACKGROUNDS) * len(SIZES):,}\n")

    print("  decoding glyphs:")
    glyphs = preload(catalog)

    out = ROOT / args.out
    (out).mkdir(parents=True, exist_ok=True)
    (out / "catalog.json").write_text(json.dumps(
        [{k: v for k, v in e.items() if k != "path"} for e in catalog]))

    print("\n  train:")
    build(args.n, args.size, out / "train", args.seed, catalog, glyphs)
    print("  val:")
    build(args.n_val, args.size, out / "val", args.seed + 1, catalog, glyphs)
    print(f"\n  next:  python train_vae.py --data data/emoji\n")


if __name__ == "__main__":
    main()
