"""Generate images from text.

    python sample.py --prompt "red heart"
    python sample.py --prompt "pizza in the top left on a navy background"
    python sample.py --prompt "rocket" --steps 4 --guidance 5 --n 8
    python sample.py --sheet                      # a spread of stock prompts

Reports wall-clock time per image, because "fast" is a claim that needs a
number next to it.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import config as config_module
from diffusion import Diffusion
from model import VAE, WordTokenizer
from train_diffusion import Prior

ROOT = Path(__file__).resolve().parent

SHEET = [
    "red heart", "pizza", "rocket", "grinning face",
    "birthday cake", "soccer ball", "rainbow", "cat face",
    "pizza in the top left on a navy background",
    "red heart in the bottom right on a black background",
    "rocket in the center on a teal background",
    "grinning face in the top right on a cream background",
]


def load(cfg, device, use_ema=True):
    ck_v = torch.load(ROOT / cfg.out_dir / f"{cfg.name}-vae.pt",
                      map_location="cpu", weights_only=False)
    vae = VAE(cfg.vae_base, cfg.latent_ch, cfg.vae_levels).to(device)
    vae.load_state_dict(ck_v["model"]); vae.eval()

    ck_p = torch.load(ROOT / cfg.out_dir / f"{cfg.name}-prior.pt",
                      map_location="cpu", weights_only=False)
    tok = WordTokenizer(ck_p["vocab"])
    # Rebuild from the config stored IN the checkpoint, not from config.py.
    # Reading the live config means any later edit to unet_base or text_dim
    # silently constructs a differently-shaped model and load_state_dict fails
    # with a wall of shape errors that look like a corrupt checkpoint.
    import config as _cm
    cfg = _cm.Config(**ck_p["config"])
    prior = Prior(cfg, tok.vocab_size).to(device)
    prior.load_state_dict(ck_p["model"])
    if use_ema:
        sd = prior.state_dict()
        for k, v in ck_p["ema"].items():
            sd[k].copy_(v.to(sd[k].dtype))
    prior.eval()
    return vae, prior, tok, float(ck_p["latent_scale"]), ck_p


@torch.no_grad()
def generate(vae, prior, tok, scale, cfg, prompts, steps, guidance, device, seed=None):
    B = len(prompts)
    ids = tok.batch([p.lower() for p in prompts], cfg.max_tokens, device)
    null = tok.batch([""] * B, cfg.max_tokens, device)
    ctx, pad = prior.text(ids)
    nctx, npad = prior.text(null)

    g = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    diff = Diffusion(cfg.timesteps, cfg.schedule, device=device)
    shape = (B, cfg.latent_ch, cfg.latent_size, cfg.latent_size)

    t0 = time.time()
    z = diff.ddim(prior.unet, shape, ctx, pad, nctx, npad,
                  steps=steps, guidance=guidance, device=device, generator=g)
    img = vae.decode(z / scale)
    dt = time.time() - t0

    img = ((img.clamp(-1, 1) + 1) * 127.5).round().byte()
    return img.permute(0, 2, 3, 1).cpu().numpy(), dt


def grid(arr, cols, scale=3):
    n, h, w, _ = arr.shape
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w, rows * h), (30, 30, 32))
    for i in range(n):
        sheet.paste(Image.fromarray(arr[i]), ((i % cols) * w, (i // cols) * h))
    return sheet.resize((cols * w * scale, rows * h * scale), Image.NEAREST)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="AS-I")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="samples.png")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = config_module.get(args.preset)
    if args.device: cfg.device = args.device
    device = cfg.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    vae, prior, tok, scale, ck = load(cfg, device, use_ema=not args.no_ema)

    if args.sheet:
        prompts, cols = SHEET, 4
    else:
        if not args.prompt:
            raise SystemExit("give --prompt or --sheet")
        prompts, cols = [args.prompt] * args.n, min(args.n, 4)

    # warn about words the model has never seen, rather than silently ignoring
    unknown = sorted({w for p in prompts for w in p.lower().split()
                      if w not in tok.stoi})
    if unknown:
        print(f"\n  [!] not in vocabulary, will be ignored: {', '.join(unknown)}")

    arr, dt = generate(vae, prior, tok, scale, cfg, prompts,
                       args.steps, args.guidance, device, args.seed)

    out = ROOT / args.out
    grid(arr, cols).save(out)
    print(f"\n  prior val loss {ck['val']:.4f} @ iter {ck['iter']:,}")
    print(f"  {len(prompts)} images  {args.steps} steps  guidance {args.guidance}")
    print(f"  {dt:.2f}s total   {dt / len(prompts) * 1000:.0f} ms/image  ({device})")
    print(f"  -> {out.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
