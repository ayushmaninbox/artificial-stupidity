"""The compressor: 64x64x3 pixels -> 8x8x4 latent. 48x fewer numbers.

This is the piece the RQ-VAE reference repo spends 67 MB on, because a
16384-entry x 256-dim codebook x 4 quantizers is 16.8M parameters of pure
lookup table. We use a continuous latent instead and that entire table
disappears — no codebook, no commitment loss, no dead codes to babysit.

The tradeoff is real: discrete tokens let you use a cheap autoregressive or
masked prior, continuous latents need a diffusion prior. We want few-step
sampling anyway, so diffusion was the plan regardless. The codebook was
buying us nothing and costing the entire size budget.

Only the decoder runs at generation time. The encoder exists to train it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(ch):
    # groups=8 unless the layer is too narrow to split that way
    return nn.GroupNorm(min(8, ch), ch)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1 = norm(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.n2 = norm(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class Encoder(nn.Module):
    def __init__(self, base, z_ch, levels=3):
        super().__init__()
        self.stem = nn.Conv2d(3, base, 3, padding=1)
        chans, ch = [], base
        blocks = []
        for i in range(levels):
            cout = base * (2 ** min(i + 1, 2))      # base, 2base, 4base, 4base...
            blocks.append(ResBlock(ch, cout))
            blocks.append(nn.Conv2d(cout, cout, 3, stride=2, padding=1))
            ch = cout
            chans.append(ch)
        self.blocks = nn.Sequential(*blocks)
        self.mid = ResBlock(ch, ch)
        self.out = nn.Sequential(norm(ch), nn.SiLU(),
                                 nn.Conv2d(ch, 2 * z_ch, 3, padding=1))

    def forward(self, x):
        h = self.mid(self.blocks(self.stem(x)))
        return self.out(h).chunk(2, dim=1)          # mu, logvar


class Decoder(nn.Module):
    def __init__(self, base, z_ch, levels=3):
        super().__init__()
        ch = base * 4
        self.stem = nn.Conv2d(z_ch, ch, 3, padding=1)
        self.mid = ResBlock(ch, ch)
        blocks = []
        for i in range(levels):
            cout = base * (2 ** max(2 - i - 1, 0))  # 4base -> 2base -> base -> base
            cout = max(cout, base)
            blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
            blocks.append(ResBlock(ch, cout))
            ch = cout
        self.blocks = nn.Sequential(*blocks)
        self.out = nn.Sequential(norm(ch), nn.SiLU(),
                                 nn.Conv2d(ch, 3, 3, padding=1))

    def forward(self, z):
        return self.out(self.blocks(self.mid(self.stem(z))))


class VAE(nn.Module):
    """Images in [-1, 1]. Latents get rescaled to ~unit variance for the prior."""

    def __init__(self, base=32, z_ch=4):
        super().__init__()
        self.encoder = Encoder(base, z_ch)
        self.decoder = Decoder(base, z_ch)
        # filled in after training by scan_scale(); the prior trains on
        # latents * scale so it sees roughly N(0,1) inputs
        self.register_buffer("scale", torch.tensor(1.0))

    def encode(self, x, sample=True):
        mu, logvar = self.encoder(x)
        if not sample:
            return mu
        std = (0.5 * logvar.clamp(-30, 20)).exp()
        return mu + std * torch.randn_like(std)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        logvar = logvar.clamp(-30, 20)
        std = (0.5 * logvar).exp()
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon, kl

    @torch.no_grad()
    def scan_scale(self, images_u8, device, batches=16, bs=64):
        """Latent std over real data -> the 1/sigma the prior wants.

        Takes the raw uint8 NCHW store and applies the same [-1, 1] mapping
        the training loop uses; passing it pre-normalized floats would double
        the scaling and quietly mis-size every latent the prior ever sees.
        """
        vals = []
        n = images_u8.shape[0]
        for i in range(batches):
            idx = torch.randint(0, n, (bs,))
            x = images_u8[idx].to(device).float().div_(127.5).sub_(1.0)
            vals.append(self.encode(x, sample=False).flatten())
        std = torch.cat(vals).std()
        self.scale = (1.0 / std).detach().to(self.scale.device)
        return self.scale.item()

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
