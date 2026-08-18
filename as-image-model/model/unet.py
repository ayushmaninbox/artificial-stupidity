"""The prior: the part that actually invents a picture from a sentence.

This is the component missing from the RQ-VAE reference entirely. An
autoencoder can only rebuild what you hand it; something has to be able to
produce a latent it has never seen, from text alone. That is this file.

It is a U-Net rather than a transformer because at 8x8 there are only 64
spatial positions and convolution's locality is free structure we would
otherwise have to learn from data we do not have.

Conditioning is cross-attention on the text sequence (not a pooled vector) so
"red circle top left" can bind colour to the shape and position to the layout
independently — pooling averages those into one blur and spatial accuracy dies.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(ch):
    return nn.GroupNorm(min(8, ch), ch)


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period)
                      * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class ResBlock(nn.Module):
    """Conv block with the timestep injected as a per-channel scale and shift."""

    def __init__(self, cin, cout, t_dim):
        super().__init__()
        self.n1 = norm(cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.t = nn.Linear(t_dim, 2 * cout)
        self.n2 = norm(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.t(F.silu(temb))[:, :, None, None].chunk(2, dim=1)
        h = F.silu(self.n2(h) * (1 + scale) + shift)
        return self.c2(h) + self.skip(x)


class SpatialTransformer(nn.Module):
    """Self-attention over pixels, then cross-attention onto the caption."""

    def __init__(self, ch, ctx_dim, heads):
        super().__init__()
        self.norm = norm(ch)
        self.proj_in = nn.Conv2d(ch, ch, 1)
        self.n1 = nn.LayerNorm(ch)
        self.self_attn = nn.MultiheadAttention(ch, heads, batch_first=True)
        self.n2 = nn.LayerNorm(ch)
        self.cross_attn = nn.MultiheadAttention(ch, heads, batch_first=True,
                                                kdim=ctx_dim, vdim=ctx_dim)
        self.n3 = nn.LayerNorm(ch)
        self.ff = nn.Sequential(nn.Linear(ch, 4 * ch), nn.GELU(),
                                nn.Linear(4 * ch, ch))
        self.proj_out = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x, ctx, ctx_pad):
        B, C, H, W = x.shape
        res = x
        h = self.proj_in(self.norm(x))
        h = h.flatten(2).transpose(1, 2)             # B, HW, C

        a = self.n1(h)
        a, _ = self.self_attn(a, a, a, need_weights=False)
        h = h + a

        a = self.n2(h)
        a, _ = self.cross_attn(a, ctx, ctx, key_padding_mask=ctx_pad,
                               need_weights=False)
        h = h + a

        h = h + self.ff(self.n3(h))
        h = h.transpose(1, 2).reshape(B, C, H, W)
        return res + self.proj_out(h)


class UNet(nn.Module):
    def __init__(self, latent_ch=4, base=128, mult=(1, 2), ctx_dim=128,
                 heads=4, blocks_per_level=2):
        super().__init__()
        t_dim = base * 4
        self.t_embed_dim = base
        self.time = nn.Sequential(nn.Linear(base, t_dim), nn.SiLU(),
                                  nn.Linear(t_dim, t_dim))

        self.stem = nn.Conv2d(latent_ch, base, 3, padding=1)

        chans = [base * m for m in mult]
        self.downs = nn.ModuleList()
        skip_chans = [base]
        ch = base
        for i, cout in enumerate(chans):
            for _ in range(blocks_per_level):
                self.downs.append(nn.ModuleList([
                    ResBlock(ch, cout, t_dim),
                    SpatialTransformer(cout, ctx_dim, heads),
                ]))
                ch = cout
                skip_chans.append(ch)
            if i != len(chans) - 1:
                self.downs.append(nn.ModuleList([
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                ]))
                skip_chans.append(ch)

        self.mid = nn.ModuleList([
            ResBlock(ch, ch, t_dim),
            SpatialTransformer(ch, ctx_dim, heads),
            ResBlock(ch, ch, t_dim),
        ])

        self.ups = nn.ModuleList()
        for i, cout in enumerate(reversed(chans)):
            for j in range(blocks_per_level + 1):
                self.ups.append(nn.ModuleList([
                    ResBlock(ch + skip_chans.pop(), cout, t_dim),
                    SpatialTransformer(cout, ctx_dim, heads),
                ]))
                ch = cout
            if i != len(chans) - 1:
                self.ups.append(nn.ModuleList([nn.Upsample(scale_factor=2,
                                                           mode="nearest")]))

        self.out = nn.Sequential(norm(ch), nn.SiLU(),
                                 nn.Conv2d(ch, latent_ch, 3, padding=1))
        nn.init.zeros_(self.out[2].weight)
        nn.init.zeros_(self.out[2].bias)

    def forward(self, x, t, ctx, ctx_pad):
        temb = self.time(timestep_embedding(t, self.t_embed_dim))
        h = self.stem(x)
        hs = [h]
        for layer in self.downs:
            if len(layer) == 1:                      # downsample conv
                h = layer[0](h)
            else:
                res, attn = layer
                h = attn(res(h, temb), ctx, ctx_pad)
            hs.append(h)

        h = self.mid[0](h, temb)
        h = self.mid[1](h, ctx, ctx_pad)
        h = self.mid[2](h, temb)

        for layer in self.ups:
            if len(layer) == 1:                      # upsample
                h = layer[0](h)
            else:
                res, attn = layer
                h = attn(res(torch.cat([h, hs.pop()], dim=1), temb), ctx, ctx_pad)
        return self.out(h)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
