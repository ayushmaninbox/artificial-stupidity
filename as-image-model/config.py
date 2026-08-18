"""AS-I — Artificial Stupidity Image. Model + training configuration.

The whole project is one question: what is the smallest network that still
turns a sentence into a picture that matches it?

So every number here is a size lever, and the defaults are deliberately mean.
"""

import math
from dataclasses import dataclass, asdict


@dataclass
class Config:
    name: str = "AS-I"

    # --- image ---
    image_size: int = 64
    latent_size: int = 16         # derived: image_size // vae_downsample
    latent_ch: int = 4            # continuous. no codebook, no 67 MB table.

    # Spatial downsampling factor of the VAE. This is the single most
    # consequential number in the file and 8 is the trap.
    #
    # Stable Diffusion uses 8, which is where the instinct to copy it comes
    # from — but SD applies it to 512px images and lands on a 64x64 latent.
    # Applying 8 to a 64px image lands on 8x8: the same compression *ratio*
    # with 64x fewer spatial cells to put detail in. Measured, that VAE
    # reconstructed a rainbow as a brown smear at 21.7 dB.
    #
    # 4 -> 16x16 latent, 12x compression, and colour survives.
    vae_downsample: int = 4

    # --- vae ---
    vae_base: int = 48            # channel width at full resolution
    vae_kl_weight: float = 1e-6   # tiny, as in latent diffusion. keeps the
                                  # latent smooth without blurring the image.

    # --- text encoder ---
    text_dim: int = 128
    text_layers: int = 2
    text_heads: int = 4
    max_tokens: int = 24

    # --- unet ---
    # 128 was the first guess and it is the wrong shape for this problem: 23.2M
    # parameters is most of the size budget spent on a prior that only has to
    # place ~1250 flat glyphs, and at a 16x16 latent it trained too slowly to
    # iterate on. 96 is 13.2M and ~1.8x faster for the same job.
    unet_base: int = 96
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
        self.latent_size = self.image_size // self.vae_downsample

    @property
    def vae_levels(self) -> int:
        """How many stride-2 stages the VAE needs. 4 -> 2, 8 -> 3."""
        return int(math.log2(self.vae_downsample))

    def dict(self):
        return asdict(self)


PRESETS = {
    # the honest default: 64x64, ~12M params
    "AS-I": Config(name="AS-I"),

    # half the generator. tests how much capacity the prior actually needs.
    "AS-I-S": Config(name="AS-I-S", unet_base=64, text_dim=96),

    # quarter. expected to start losing spatial relations first.
    "AS-I-XS": Config(name="AS-I-XS", unet_base=48, text_dim=64,
                      unet_mult=(1, 2), vae_base=32),

    # 128x128, for when the small one works
    "AS-I-128": Config(name="AS-I-128", image_size=128, unet_base=128),

    # Same network, a quarter of the vocabulary. AS-I has to spread 13.2M
    # parameters across 1254 glyph identities; this one sees ~300, so each is
    # ~4x more frequent in training. The pair answers the question the whole
    # project is about: how much coverage does a fixed parameter budget buy
    # before individual items stop being recognisable?
    "AS-I-300": Config(name="AS-I-300", data_dir="data/emoji300"),
}


def get(name: str) -> Config:
    if name not in PRESETS:
        raise SystemExit(f"unknown preset {name!r}. pick one of: {', '.join(PRESETS)}")
    return Config(**asdict(PRESETS[name]))
