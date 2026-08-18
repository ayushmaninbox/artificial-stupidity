"""Check what survives the 48x squeeze, before training anything on top of it.

The prior paints into the VAE's latent space, so the VAE is a hard ceiling on
final quality: whatever detail the decoder cannot reproduce here is detail the
finished model can never generate, no matter how long stage 2 runs. Worth
looking at an actual picture before spending an hour on the prior.

    python recon.py --data data/emoji
    python recon.py --data data/emoji --n 12 --out recon.png
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import config as config_module
from model import VAE

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="AS-I")
    ap.add_argument("--data", default="data/emoji")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="recon.png")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = config_module.get(args.preset)
    cfg.data_dir = args.data
    if args.device: cfg.device = args.device
    device = cfg.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    ck = torch.load(ROOT / cfg.out_dir / f"{cfg.name}-vae.pt",
                    map_location="cpu", weights_only=False)
    vae = VAE(cfg.vae_base, cfg.latent_ch, cfg.vae_levels).to(device)
    vae.load_state_dict(ck["model"]); vae.eval()

    imgs = np.load(ROOT / cfg.data_dir / "val" / "images.npy")
    idx = np.random.default_rng(0).choice(len(imgs), args.n, replace=False)
    x = torch.from_numpy(imgs[idx]).permute(0, 3, 1, 2).float().div(127.5).sub(1).to(device)

    with torch.no_grad():
        z = vae.encode(x, sample=False)
        r = vae.decode(z).clamp(-1, 1)

    mse = F.mse_loss((r + 1) / 2, (x + 1) / 2).item()
    psnr = 10 * np.log10(1.0 / max(mse, 1e-10))

    top = ((x + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    bot = ((r + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    n, h, w, _ = top.shape
    sheet = Image.new("RGB", (n * w, 2 * h))
    for i in range(n):
        sheet.paste(Image.fromarray(top[i]), (i * w, 0))
        sheet.paste(Image.fromarray(bot[i]), (i * w, h))
    sheet = sheet.resize((n * w * args.scale, 2 * h * args.scale), Image.NEAREST)
    sheet.save(ROOT / args.out)

    print(f"\n  latent      {tuple(z.shape[1:])}  ({np.prod(z.shape[1:]):.0f} numbers "
          f"vs {3 * cfg.image_size ** 2:,} pixels, {3 * cfg.image_size ** 2 / np.prod(z.shape[1:]):.0f}x)")
    print(f"  psnr        {psnr:.1f} dB")
    print(f"  latent std  {z.std():.3f}   stored scale {ck['scale']:.3f}")
    print(f"  top row = original, bottom row = through the VAE")
    print(f"  -> {args.out}\n")


if __name__ == "__main__":
    main()
