"""The text encoder. ~0.4M parameters, versus CLIP's 63M / 250 MB.

CLIP is the single least defensible component to inherit for this project.
The low-VRAM SD reference ships it at 8-bit and it still costs 250 MB — more
than twice our entire size budget — and it is trained to understand *all*
English, of which our grammar uses about forty words.

So this is a word-level transformer trained from scratch alongside the
generator. It cannot describe a photograph. It does not need to. The vocabulary
is built from the corpus and is roughly 45 tokens.

The consequence is honest and worth stating plainly: AS-I understands its own
caption grammar and nothing else. Prompt it in free English and it will pick
out the words it knows and ignore the rest.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, BOS, UNK = 0, 1, 2
SPECIALS = ["<pad>", "<bos>", "<unk>"]


class WordTokenizer:
    def __init__(self, itos):
        self.itos = list(itos)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def build(cls, captions):
        words = sorted({w for c in captions for w in c.split()})
        return cls(SPECIALS + words)

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text, max_tokens):
        ids = [BOS] + [self.stoi.get(w, UNK) for w in text.lower().split()]
        ids = ids[:max_tokens]
        return ids + [PAD] * (max_tokens - len(ids))

    def batch(self, texts, max_tokens, device):
        return torch.tensor([self.encode(t, max_tokens) for t in texts],
                            dtype=torch.long, device=device)

    def save(self, path):
        Path(path).write_text(json.dumps(self.itos))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(),
                                 nn.Linear(4 * dim, dim))

    def forward(self, x, pad_mask):
        h = self.n1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.n2(x))


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, dim=128, layers=2, heads=4, max_tokens=24):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, dim)
        self.pos = nn.Parameter(torch.zeros(1, max_tokens, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.out = nn.LayerNorm(dim)
        self.max_tokens = max_tokens
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, ids):
        """ids (B, T) -> (B, T, dim) sequence for cross-attention, + pad mask."""
        pad_mask = ids.eq(PAD)
        # a fully-padded row would make attention produce NaN; the BOS token
        # guarantees at least one real position, so this only guards the
        # unconditional embedding path
        x = self.tok(ids) + self.pos[:, :ids.shape[1]]
        for b in self.blocks:
            x = b(x, pad_mask)
        return self.out(x), pad_mask
