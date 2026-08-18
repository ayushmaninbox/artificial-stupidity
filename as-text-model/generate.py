"""Talk to the little bastard.

    python generate.py AS-0 --prompt "hello"
    python generate.py AS-4 --chat
"""

import argparse
from pathlib import Path

import torch

from config import Config
from model import ArtificialStupidity, CharTokenizer

ROOT = Path(__file__).resolve().parent


def load(name, device):
    path = ROOT / "checkpoints" / f"{name}.pt"
    if not path.exists():
        raise SystemExit(f"no checkpoint at {path}. train it first: python train.py {name}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    cfg.dropout = 0.0
    model = ArtificialStupidity(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = CharTokenizer.load(ROOT / cfg.data_dir / "tokenizer.json")
    return model, tok, cfg, ckpt


def sample(model, tok, cfg, prompt, device, n_tokens, temperature, top_k):
    ids = tok.encode(prompt) or [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(idx, n_tokens, temperature=temperature, top_k=top_k)
    return tok.decode(out[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preset", nargs="?", default="AS-0")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    device = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    model, tok, cfg, ckpt = load(args.preset, device)

    print(f"\n{cfg.name} | {cfg.weight_bits}-bit | val loss {ckpt['val_loss']:.3f} | "
          f"{model.num_params():,} params\n")

    if args.chat:
        print("(ctrl-c to escape)\n")
        try:
            while True:
                user = input("you  > ")
                text = sample(model, tok, cfg, f"A: {user}\nB:", device,
                              args.tokens, args.temperature, args.top_k)
                reply = text.split("B:", 1)[-1].split("\n")[0].strip()
                print(f"AS   > {reply or '...'}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nbye\n")
        return

    for i in range(args.samples):
        if args.samples > 1:
            print(f"--- sample {i + 1} ---")
        print(sample(model, tok, cfg, args.prompt, device,
                     args.tokens, args.temperature, args.top_k))
        print()


if __name__ == "__main__":
    main()
