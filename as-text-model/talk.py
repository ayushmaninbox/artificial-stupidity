"""Talk to a fine-tuned (and possibly compressed) Artificial Stupidity model.

    python talk.py --chat
    python talk.py --prompt "bro is" --samples 3
    python talk.py --model checkpoints/AS-F-4.0bit --chat
"""

import argparse
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="checkpoints/AS-F")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    path = ROOT / args.model
    if not path.exists():
        raise SystemExit(f"no model at {path}. run finetune.py first.")

    device = args.device if (args.device != "mps" or torch.backends.mps.is_available()) else "cpu"
    tok = GPT2TokenizerFast.from_pretrained(path)
    model = GPT2LMHeadModel.from_pretrained(path).to(device).eval()

    n = sum(p.numel() for p in model.parameters())
    print(f"\n{args.model} | {n / 1e6:.0f}M params | {device}\n")

    def generate(text):
        enc = tok(text, return_tensors="pt").to(device)
        out = model.generate(
            **enc,
            max_new_tokens=args.tokens,
            do_sample=True,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            pad_token_id=tok.eos_token_id,
        )
        return tok.decode(out[0], skip_special_tokens=True)

    if args.chat:
        print("(ctrl-c to escape)\n")
        try:
            while True:
                user = input("you  > ").strip()
                if not user:
                    continue
                text = generate(f"A: {user}\nB:")
                reply = text.split("B:", 1)[-1].split("\n")[0].strip()
                print(f"AS   > {reply or '...'}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nbye\n")
        return

    for i in range(args.samples):
        if args.samples > 1:
            print(f"--- sample {i + 1} ---")
        print(generate(args.prompt or tok.eos_token))
        print()


if __name__ == "__main__":
    main()
