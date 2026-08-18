"""Artificial Stupidity — model + training configuration.

Every AS-N variant is the SAME architecture with a different `weight_bits`.
That's the whole experiment: how little precision can we get away with
before the little bastard stops speaking English?
"""

from dataclasses import dataclass, asdict, field


@dataclass
class Config:
    # --- identity ---
    name: str = "AS-0"

    # --- architecture ---
    vocab_size: int = 0        # filled in from the tokenizer at load time
    block_size: int = 128      # context length, in characters
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    bias: bool = False         # biases in Linear/LayerNorm. nanoGPT says off is better+faster

    # --- the crimes ---
    # 32   = full precision (control group)
    # 8/4  = symmetric per-output-channel integer quantization
    # 1.58 = ternary {-1, 0, +1}, BitNet b1.58 style
    # 1    = binary {-1, +1}, absolute artificial stupidity
    weight_bits: float = 32
    # activation quantization (BitNet uses 8). 32 = off.
    act_bits: float = 32

    # --- training ---
    batch_size: int = 64
    max_iters: int = 3000
    eval_interval: int = 250
    eval_iters: int = 50
    learning_rate: float = 1e-3
    min_lr: float = 1e-4
    warmup_iters: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # --- runtime ---
    device: str = "mps"
    seed: int = 1337
    data_dir: str = "data/processed"
    out_dir: str = "checkpoints"

    def dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# The leaderboard contestants. Same brain, progressively worse hardware.
# ---------------------------------------------------------------------------

PRESETS = {
    # control group: a normal (tiny) transformer
    "AS-0": Config(name="AS-0", weight_bits=32),

    # mild crime
    "AS-1": Config(name="AS-1", weight_bits=8),

    # moderate crime
    "AS-2": Config(name="AS-2", weight_bits=4),

    # BitNet b1.58 territory: weights are {-1, 0, +1}
    "AS-3": Config(name="AS-3", weight_bits=1.58, learning_rate=2e-3),

    # ONE BIT. weights are {-1, +1}. no zero. no nuance. no thoughts.
    "AS-4": Config(name="AS-4", weight_bits=1, learning_rate=2e-3),

    # 1-bit weights AND 8-bit activations, half the size, twice the brain damage
    "AS-5": Config(
        name="AS-5",
        weight_bits=1,
        act_bits=8,
        n_layer=3,
        n_embd=96,
        learning_rate=2e-3,
    ),
}


def get(name: str) -> Config:
    if name not in PRESETS:
        raise SystemExit(
            f"unknown preset {name!r}. pick one of: {', '.join(PRESETS)}"
        )
    # return a copy so callers can mutate freely
    return Config(**asdict(PRESETS[name]))
