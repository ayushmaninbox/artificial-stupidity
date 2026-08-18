"""Stage 1.5 — encode the corpus into latents once, then never again.

The diffusion prior only ever sees latents, so running the VAE encoder on every
training step is pure waste: the same image encodes to the same place every
time. Encoding once up front buys two things.

    speed  the encoder drops out of the training loop entirely
    size   45,000 x 3x64x64 uint8  =  553 MB of pixels
           45,000 x 4x8x8  float16 =   5.8 MB of latents
                                       ~95x smaller, fits in cache

Latents are stored already multiplied by the VAE's scale factor, so the prior
trains on roughly unit-variance inputs and the sampler can divide it straight
back out at decode time.

    python precompute_latents.py --data data/emoji
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import config as config_module
from model import VAE

ROOT = Path(__file__).resolve().parent


@torch.no_grad()
def encode_split(vae, images_u8, device, bs, scale):
    out = []
    n = images_u8.shape[0]
    for i in range(0, n, bs):
        x = images_u8[i:i + bs].to(device).float().div_(127.5).sub_(1.0)
        z = vae.encode(x, sample=False) * scale
        out.append(z.to(torch.float16).cpu())
        if (i // bs) % 50 == 0:
            print(f"    {min(i + bs, n):>6}/{n}", flush=True)
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="AS-I")
    ap.add_argument("--data", default=None, help="overrides the preset data dir")
    ap.add_argument("--vae", default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = config_module.get(args.preset)
    if args.data:   cfg.data_dir = args.data
    if args.device: cfg.device = args.device
    device = cfg.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    ck_path = ROOT / (args.vae or f"{cfg.out_dir}/{cfg.name}-vae.pt")
    if not ck_path.exists():
        raise SystemExit(f"no VAE at {ck_path}. run: python train_vae.py")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)

    vae = VAE(cfg.vae_base, cfg.latent_ch).to(device)
    vae.load_state_dict(ck["model"])
    vae.eval()
    scale = float(ck["scale"])

    data = ROOT / cfg.data_dir
    print(f"\n{'=' * 60}\n  PRECOMPUTING LATENTS  —  scale {scale:.3f}\n{'=' * 60}")

    for split in ("train", "val"):
        src = data / split
        imgs = torch.from_numpy(np.load(src / "images.npy")).permute(0, 3, 1, 2).contiguous()
        print(f"  {split}: {imgs.shape[0]:,} images")
        z = encode_split(vae, imgs, device, args.batch_size, scale)
        torch.save({"latents": z, "scale": scale}, src / "latents.pt")
        px = imgs.numel()
        print(f"    -> {tuple(z.shape)}  {z.numel() * 2 / 1e6:.1f} MB "
              f"(from {px / 1e6:.0f} MB of pixels, {px / (z.numel() * 2):.0f}x smaller)")
        print(f"    latent std after scaling: {z.float().std():.3f}  (want ~1.0)")

    print(f"\n  next:  python train_diffusion.py --data {cfg.data_dir}\n")


if __name__ == "__main__":
    main()
