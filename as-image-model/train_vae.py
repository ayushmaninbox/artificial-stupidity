"""Stage 1 — train the compressor.

The prior can never be better than the latent it paints into, so this runs
first and gets frozen. If the VAE cannot reconstruct a sharp red triangle,
no amount of diffusion training will produce one.

    python train_vae.py
    python train_vae.py --iters 6000 --preset AS-I-XS
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config as config_module
from model import VAE

ROOT = Path(__file__).resolve().parent


def load_images(path: Path, device):
    arr = np.load(path / "images.npy")
    print(f"  {arr.shape[0]:,} images  {arr.shape[1]}x{arr.shape[2]}  "
          f"{arr.nbytes / 1e6:.0f} MB")
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()   # uint8 NCHW


def batch(imgs, bs, device):
    idx = torch.randint(0, imgs.shape[0], (bs,))
    x = imgs[idx].to(device).float().div_(127.5).sub_(1.0)          # [-1, 1]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="AS-I")
    ap.add_argument("--data", default=None, help="overrides the preset data dir")
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--device")
    args = ap.parse_args()

    cfg = config_module.get(args.preset)
    if args.data:       cfg.data_dir = args.data
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.device:     cfg.device = args.device
    device = cfg.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    torch.manual_seed(cfg.seed)

    data = ROOT / cfg.data_dir
    if not (data / "train" / "images.npy").exists():
        raise SystemExit("no dataset. run: python data/shapes.py")

    print(f"\n{'=' * 60}\n  VAE  —  {cfg.name}  {cfg.image_size}px -> "
          f"{cfg.latent_size}x{cfg.latent_size}x{cfg.latent_ch}\n{'=' * 60}")
    train = load_images(data / "train", device)
    val = load_images(data / "val", device)

    vae = VAE(cfg.vae_base, cfg.latent_ch, cfg.vae_levels).to(device)
    print(f"  parameters    {vae.num_params():,}")
    print(f"  device        {device}")
    print(f"  compression   {cfg.image_size**2 * 3 / (cfg.latent_size**2 * cfg.latent_ch):.0f}x")
    print(f"{'=' * 60}\n", flush=True)

    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=1e-4)
    out_dir = ROOT / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    best = float("inf")
    for it in range(args.iters + 1):
        x = batch(train, cfg.batch_size, device)
        recon, kl = vae(x)
        # L1 keeps flat colour fields flat, L2 punishes the big edge errors.
        # Pure L2 alone gives grey mush at this capacity.
        rec = F.l1_loss(recon, x) + F.mse_loss(recon, x)
        loss = rec + cfg.vae_kl_weight * kl

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae.parameters(), cfg.grad_clip)
        opt.step()

        if it % args.eval_every == 0 or it == args.iters:
            vae.eval()
            with torch.no_grad():
                xv = batch(val, 128, device)
                rv, _ = vae(xv)
                vl = (F.l1_loss(rv, xv) + F.mse_loss(rv, xv)).item()
                # PSNR on [0,1] scale, the readable version of the same number
                mse01 = F.mse_loss((rv + 1) / 2, (xv + 1) / 2).item()
                psnr = 10 * np.log10(1.0 / max(mse01, 1e-10))
            vae.train()
            flag = ""
            if vl < best:
                best = vl
                scale = vae.scan_scale(train, device)
                torch.save({"model": vae.state_dict(), "config": cfg.dict(),
                            "scale": scale, "val": vl},
                           out_dir / f"{cfg.name}-vae.pt")
                flag = " *"
            print(f"  iter {it:>5}  train {rec.item():.4f}  val {vl:.4f}  "
                  f"psnr {psnr:5.1f} dB  {time.time() - t0:5.0f}s{flag}", flush=True)

    ck = torch.load(out_dir / f"{cfg.name}-vae.pt", map_location="cpu", weights_only=False)
    print(f"\n  best val      {ck['val']:.4f}")
    print(f"  latent scale  {ck['scale']:.3f}")
    print(f"  saved         checkpoints/{cfg.name}-vae.pt")
    print(f"\n  next:  python train_diffusion.py\n")


if __name__ == "__main__":
    main()
