"""Export AS-I for the browser: three graphs plus the vocabulary.

The Python sampler in diffusion.py does not come along — only the networks do.
The DDIM loop, the cosine schedule and classifier-free guidance all have to be
reimplemented in JavaScript, which is why the schedule is baked into
model.json here rather than recomputed there: two implementations of the same
alpha/sigma table is two chances to get it subtly wrong.

    python export_web.py --preset AS-I
    python export_web.py --preset AS-I-300 --out web_AS-I-300
"""

import argparse, json, math, shutil
from pathlib import Path

import torch

import config as config_module
from diffusion import Diffusion
from model import VAE, WordTokenizer
from train_diffusion import Prior

ROOT = Path(__file__).resolve().parent


class TextGraph(torch.nn.Module):
    def __init__(self, prior): super().__init__(); self.t = prior.text
    def forward(self, ids):
        ctx, pad = self.t(ids)
        return ctx, pad.to(torch.int64)      # bool masks travel badly in ORT


class UNetGraph(torch.nn.Module):
    def __init__(self, prior): super().__init__(); self.u = prior.unet
    def forward(self, z, t, ctx, pad):
        return self.u(z, t, ctx, pad.to(torch.bool))


class DecGraph(torch.nn.Module):
    def __init__(self, vae): super().__init__(); self.d = vae
    def forward(self, z): return self.d.decode(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="AS-I")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = config_module.get(args.preset)
    out = ROOT / (args.out or f"web_{cfg.name}")
    out.mkdir(parents=True, exist_ok=True)

    ckp = torch.load(ROOT / cfg.out_dir / f"{cfg.name}-prior.pt", map_location="cpu",
                     weights_only=False)
    ckv = torch.load(ROOT / cfg.out_dir / f"{cfg.name}-vae.pt", map_location="cpu",
                     weights_only=False)
    cfg = config_module.Config(**ckp["config"])
    tok = WordTokenizer(ckp["vocab"])

    prior = Prior(cfg, tok.vocab_size); prior.load_state_dict(ckp["model"])
    sd = prior.state_dict()
    for k, v in ckp["ema"].items():          # EMA weights are the good ones
        sd[k].copy_(v.to(sd[k].dtype))
    prior.eval()

    vae = VAE(cfg.vae_base, cfg.latent_ch, cfg.vae_levels)
    vae.load_state_dict(ckv["model"]); vae.eval()

    T, L, C = cfg.max_tokens, cfg.latent_size, cfg.latent_ch
    ids = torch.zeros(1, T, dtype=torch.long)
    with torch.no_grad():
        ctx, pad = prior.text(ids)

    ex = lambda m, a, f, i, o, d: torch.onnx.export(
        m, a, str(out / f), input_names=i, output_names=o, dynamic_axes=d,
        opset_version=18)

    ex(TextGraph(prior), (ids,), "text.onnx", ["ids"], ["ctx", "pad"],
       {"ids": {0: "b"}, "ctx": {0: "b"}, "pad": {0: "b"}})
    # Batch is always 2 (conditional + unconditional), so the U-Net is exported
    # static. The text graph stays batch-1 and is simply called twice: its
    # reshape does not generalise, and at 2.5 MB running it twice is free.
    ex(UNetGraph(prior),
       (torch.randn(2, C, L, L), torch.zeros(2, dtype=torch.long),
        ctx.repeat(2, 1, 1), pad.to(torch.int64).repeat(2, 1)),
       "unet.onnx", ["z", "t", "ctx", "pad"], ["v"], None)
    ex(DecGraph(vae), (torch.randn(1, C, L, L),), "decoder.onnx",
       ["z"], ["image"], {"z": {0: "b"}, "image": {0: "b"}})

    # bake the schedule so JS never recomputes it
    d = Diffusion(cfg.timesteps, cfg.schedule)
    ts = torch.linspace(cfg.timesteps - 1, 0, args.steps + 1).round().long().tolist()
    (out / "model.json").write_text(json.dumps({
        "name": cfg.name,
        "vocab": tok.itos,
        "maxTokens": T, "latent": L, "latentCh": C, "imageSize": cfg.image_size,
        "steps": args.steps, "timesteps": cfg.timesteps,
        "scale": float(ckv["scale"]),
        "schedule": ts,
        "alpha": [float(d.alpha[i]) for i in ts],
        "sigma": [float(d.sigma[i]) for i in ts],
        "guidance": 4.0,
    }, indent=2))

    # consolidate external weights, then int8 the lot
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    q = out / "int8"; q.mkdir(exist_ok=True)
    for f in sorted(out.glob("*.onnx")):
        m = onnx.load(str(f))
        # The dynamo exporter stamps intermediate shapes that disagree with what
        # inference derives, and quantization refuses to run against the
        # conflict. Clearing them lets ORT recompute from the graph itself.
        del m.graph.value_info[:]
        onnx.save_model(m, str(f), save_as_external_data=False)
        Path(str(f) + ".data").unlink(missing_ok=True)
        quantize_dynamic(str(f), str(q / f.name), weight_type=QuantType.QUInt8)
        print(f"  {f.name:<14} {f.stat().st_size/1e6:6.2f} MB -> int8 {(q/f.name).stat().st_size/1e6:5.2f} MB")
    shutil.copy(out / "model.json", q / "model.json")
    print(f"  model.json     vocab {tok.vocab_size}, {args.steps} steps")


if __name__ == "__main__":
    main()
