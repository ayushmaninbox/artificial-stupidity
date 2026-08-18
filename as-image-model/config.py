"""AS-I — Artificial Stupidity Image. Model + training configuration.

The whole project is one question: what is the smallest network that still
turns a sentence into a picture that matches it?

So every number here is a size lever, and the defaults are deliberately mean.
"""

from dataclasses import dataclass, asdict


@dataclass
class Config:
    name: str = "AS-I"

    # --- image ---
    image_size: int = 64          # 64 -> 8x8 latent. 128 -> 16x16.
    latent_size: int = 8          # derived: image_size // 8
    latent_ch: int = 4            # continuous. no codebook, no 67 MB table.

    # --- vae ---
    vae_base: int = 32            # channel width at full resolution
    vae_kl_weight: float = 1e-6   # tiny, as in latent diffusion. keeps the
                                  # latent smooth without blurring the image.

    # --- text encoder ---
    text_dim: int = 128
    text_layers: int = 2
    text_heads: int = 4
    max_tokens: int = 24

    # --- unet ---
    unet_base: int = 128
    unet_mult: tuple = (1, 2)     # 8x8 @128, 4x4 @256
    unet_heads: int = 4

    # --- diffusion ---
    timesteps: int = 1000
    schedule: str = "cosine"
    prediction: str = "v"         # v-prediction: the reason 8 steps is enough
    cfg_dropout: float = 0.1      # fraction of captions dropped for CFG

    # --- training ---
    batch_size: int = 64
    lr: float = 3e-4
    warmup: int = 200
    grad_clip: float = 1.0
    ema_decay: float = 0.999

    # --- runtime ---
    device: str = "mps"
    seed: int = 1337
    data_dir: str = "data/shapes"
    out_dir: str = "checkpoints"

    def __post_init__(self):
        self.latent_size = self.image_size // 8

    def dict(self):
        return asdict(self)


PRESETS = {
    # the honest default: 64x64, ~12M params
    "AS-I": Config(name="AS-I"),

    # half the generator. tests how much capacity the prior actually needs.
    "AS-I-S": Config(name="AS-I-S", unet_base=64, text_dim=96),

    # quarter. expected to start losing spatial relations first.
    "AS-I-XS": Config(name="AS-I-XS", unet_base=48, text_dim=64,
                      unet_mult=(1, 2), vae_base=24),

    # 128x128, for when the small one works
    "AS-I-128": Config(name="AS-I-128", image_size=128, unet_base=128),
}


def get(name: str) -> Config:
    if name not in PRESETS:
        raise SystemExit(f"unknown preset {name!r}. pick one of: {', '.join(PRESETS)}")
    return Config(**asdict(PRESETS[name]))
