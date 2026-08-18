"""AS-IF — the other trade: someone else's model, made small enough to ship.

AS-I is ~24 MB and draws emoji. AS-IF is ~350 MB and draws anything. Neither is
better; they are different answers to the same constraint, and the repo keeps
both for the same reason the text side keeps AS-F next to AS-0..AS-5.

    AS-I    every weight trained here, on a laptop, on a closed domain
    AS-IF   Stability AI trained the weights; the work here is compression,
            export, and getting it to run in a browser tab

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


def do_export(out: Path):
    from optimum.onnxruntime import ORTStableDiffusionPipeline

    print(f"\n  downloading + exporting {MODEL} (this is the slow part) ...",
          flush=True)
    t0 = time.time()
    pipe = ORTStableDiffusionPipeline.from_pretrained(MODEL, export=True)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="asif_build")
    ap.add_argument("--skip-export", action="store_true",
                    help="reuse an existing fp32 export")
    args = ap.parse_args()

    raw = ROOT / args.out / "fp32"
    q = ROOT / args.out / "int8"

    if not args.skip_export:
        raw.parent.mkdir(parents=True, exist_ok=True)
        do_export(raw)
    if not raw.exists():
        raise SystemExit(f"no export at {raw} — run without --skip-export first")

    fp32_total = report(raw, "EXPORTED (fp32)")
    print("\n  quantizing ...")
    do_quantize(raw, q)
    int8_total = report(q, "QUANTIZED (int8 unet + text encoder)")

    print(f"\n  {fp32_total / int8_total:.1f}x smaller "
          f"({human(fp32_total)} -> {human(int8_total)})")
    print(f"\n  next:  python asif_sample.py --prompt 'two astronauts playing chess'\n")


if __name__ == "__main__":
    main()
