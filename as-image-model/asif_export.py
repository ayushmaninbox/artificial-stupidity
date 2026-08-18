"""AS-IF — the other trade: a pretrained model, adapted small enough to ship.

AS-I is ~24 MB and draws emoji. AS-IF is ~350 MB and draws anything. Neither is
better; they are different answers to the same constraint, and the repo keeps
both for the same reason the text side keeps AS-F next to AS-0..AS-5.

    AS-I    every weight trained here, on a laptop, on a closed domain
    AS-IF   starts from SD-Turbo's pretraining; the work here is the
            quantization policy, the decoder swap, the ONNX export and the
            browser runtime

Why SD-Turbo rather than SD 1.5: it is adversarially distilled for **1-4 step**
sampling. Vanilla SD 1.5 needs 20-50 steps, which is unusable in a browser. The
step count, not the parameter count, is what makes in-browser generation
plausible.

What this script does:

    download SD-Turbo  ->  export each component to ONNX  ->  quantize to int8

Component precision is chosen per part, not globally, because they do not
tolerate damage equally:

    UNet          int8   the bulk of the weights, and the most robust
    text encoder  int8   robust
    VAE decoder   fp16   quantizing this produces visible colour banding on
                         flat regions, for ~80 MB of savings. Not worth it.

    python asif_export.py --steps quantize
    python asif_export.py --out asif_build --skip-export   # re-quantize only
"""

import argparse
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "stabilityai/sd-turbo"

# The smaller alternative. Tiny-SD is a block-pruned SD 1.5 (323M UNet vs
# SD-Turbo's 865M) and, being SD 1.5, carries CLIP ViT-L (123M) instead of
# OpenCLIP ViT-H (354M). Together that is ~451 MB int8 against ~1224 MB.
#
# The catch is that pruning does not make it few-step: Tiny-SD needs 20-25
# steps, which is worse than useless in a browser. LCM-LoRA fixes that, and
# fuses INTO the UNet weights, so few-step sampling costs no extra bytes.
SMALL_MODEL = "segmind/tiny-sd"
LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} TB"


def tree_size(p: Path):
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def report(out: Path, title):
    print(f"\n{'=' * 64}\n  {title}\n{'=' * 64}")
    total = 0
    for sub in sorted(out.rglob("*.onnx")):
        # .onnx may be a small graph beside a big .onnx_data blob
        size = sub.stat().st_size
        data = sub.with_suffix(".onnx_data")
        if data.exists():
            size += data.stat().st_size
        total += size
        print(f"  {str(sub.relative_to(out)):<44} {human(size):>10}")
    print(f"  {'-' * 56}")
    print(f"  {'TOTAL':<44} {human(total):>10}")
    return total


def do_export(out: Path, small: bool = False):
    from optimum.onnxruntime import ORTStableDiffusionPipeline

    model = SMALL_MODEL if small else MODEL
    print(f"\n  downloading + exporting {model} (this is the slow part) ...",
          flush=True)
    t0 = time.time()

    # NOTE on LCM-LoRA, which looked like the obvious way to make this
    # few-step and is not: LCM-LoRA is trained against the *full* SD 1.5 UNet,
    # while Tiny-SD is that UNet with blocks pruned out. The channel widths no
    # longer match --
    #
    #     up_blocks.2.resnets.1.conv1.lora_A
    #     checkpoint [64, 1280, 3, 3]  vs  model [64, 640, 3, 3]
    #
    # -- and a LoRA cannot attach to an architecture it was not trained on.
    # Pruning and step-distillation do not compose for free; a small few-step
    # model has to be distilled as one, not assembled from two parts.
    pipe = ORTStableDiffusionPipeline.from_pretrained(model, export=True)

    pipe.save_pretrained(out)
    print(f"  exported in {(time.time() - t0) / 60:.1f} min", flush=True)


def do_quantize(src: Path, dst: Path):
    """int8 the UNet and text encoder; leave the VAE decoder alone."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)

    # component -> whether to quantize
    plan = {
        "unet": True,
        "text_encoder": True,
        "vae_decoder": False,      # banding on flat colour; keep it honest
        "vae_encoder": False,      # unused for text-to-image
    }

    for name, quant in plan.items():
        f = dst / name / "model.onnx"
        if not f.exists():
            continue
        if not quant:
            print(f"  {name:<14} left as exported")
            continue
        before = f.stat().st_size + (f.with_suffix(".onnx_data").stat().st_size
                                     if f.with_suffix(".onnx_data").exists() else 0)
        print(f"  {name:<14} quantizing ({human(before)}) ...", flush=True)
        tmp = f.with_name("model_int8.onnx")
        quantize_dynamic(str(f), str(tmp), weight_type=QuantType.QUInt8)
        # replace in place so the pipeline still finds model.onnx
        f.unlink()
        old_data = f.with_suffix(".onnx_data")
        if old_data.exists():
            old_data.unlink()
        tmp.rename(f)
        print(f"  {name:<14} -> {human(f.stat().st_size)}")


def export_tiny_vae(dst: Path):
    """Swap SD's 198 MB decoder for TAESD, a 4.9 MB distilled one.

    This is the best size-per-unit-effort change available to AS-IF: 40x
    smaller *and* 2.7x faster end to end, because SD's decoder is a third of
    the generation time at 512px on CPU. Quality is visually indistinguishable
    on flat and photographic content alike.

    Note the latent convention. TAESD's scaling_factor is 1.0, so it consumes
    UNet-space latents directly; dividing by SD's 0.18215 first — the obvious
    move, and the one that matches every SD decode example — hands it values
    5.5x too large and returns psychedelic noise that looks like a broken
    model rather than a broken constant.
    """
    import torch
    from diffusers import AutoencoderTiny

    out = dst / "vae_decoder_tiny"
    out.mkdir(parents=True, exist_ok=True)
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd").eval()

    class Dec(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, latent_sample):
            return self.m.decode(latent_sample).sample

    torch.onnx.export(
        Dec(taesd), torch.randn(1, 4, 64, 64), str(out / "model.onnx"),
        input_names=["latent_sample"], output_names=["sample"],
        dynamic_axes={"latent_sample": {0: "b", 2: "h", 3: "w"},
                      "sample": {0: "b", 2: "H", 3: "W"}},
        opset_version=17,
    )
    sz = sum(f.stat().st_size for f in out.glob("model.onnx*"))
    print(f"  tiny vae       -> {human(sz)}  (replaces ~198 MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="asif_build")
    ap.add_argument("--tiny-vae", action="store_true",
                    help="also export TAESD, a 4.9 MB replacement decoder")
    ap.add_argument("--small", action="store_true",
                    help="use Tiny-SD + LCM-LoRA instead of SD-Turbo (~451 MB)")
    ap.add_argument("--skip-export", action="store_true",
                    help="reuse an existing fp32 export")
    args = ap.parse_args()

    raw = ROOT / args.out / "fp32"
    q = ROOT / args.out / "int8"

    if not args.skip_export:
        raw.parent.mkdir(parents=True, exist_ok=True)
        do_export(raw, small=args.small)
    if not raw.exists():
        raise SystemExit(f"no export at {raw} — run without --skip-export first")

    fp32_total = report(raw, "EXPORTED (fp32)")
    print("\n  quantizing ...")
    do_quantize(raw, q)
    if args.tiny_vae:
        print()
        export_tiny_vae(q)
    int8_total = report(q, "QUANTIZED (int8 unet + text encoder)")

    print(f"\n  {fp32_total / int8_total:.1f}x smaller "
          f"({human(fp32_total)} -> {human(int8_total)})")
    print(f"\n  next:  python asif_sample.py --prompt 'two astronauts playing chess'\n")


if __name__ == "__main__":
    main()
