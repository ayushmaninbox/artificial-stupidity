"""A very small decoder-only transformer. nanoGPT in spirit, ours in code.

Every Linear inside the blocks is a BitLinear, so the entire network's
precision is controlled by one number in the config.

Following BitNet, the embedding table and the output head stay in full
precision — they're a small fraction of the parameters and quantizing them
destroys the model for no size win.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .bitlinear import BitLinear


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.dropout = cfg.dropout

        self.attn = BitLinear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias,
                              weight_bits=cfg.weight_bits, act_bits=cfg.act_bits)
        self.proj = BitLinear(cfg.n_embd, cfg.n_embd, bias=cfg.bias,
                              weight_bits=cfg.weight_bits, act_bits=cfg.act_bits)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.attn(x).split(self.n_embd, dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = BitLinear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias,
                            weight_bits=cfg.weight_bits, act_bits=cfg.act_bits)
        self.proj = BitLinear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias,
                              weight_bits=cfg.weight_bits, act_bits=cfg.act_bits)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ArtificialStupidity(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.vocab_size > 0, "set cfg.vocab_size from the tokenizer first"
        self.cfg = cfg

        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying: the output head shares the embedding table
        self.lm_head.weight = self.wte.weight

        self.apply(self._init_weights)
        # scaled init for residual projections (GPT-2 trick)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence of {T} > block_size {self.cfg.block_size}"

        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.wte(idx) + self.wpe(pos))
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        if targets is None:
            # inference: only need logits for the last position
            return self.lm_head(x[:, [-1], :]), None

        logits = self.lm_head(x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.reshape(-1)
        )
        return logits, loss

    # -- introspection ------------------------------------------------------

    def num_params(self) -> int:
        # wpe counted; lm_head is tied to wte so it isn't double-counted
        return sum(p.numel() for p in self.parameters())

    def packed_bytes(self) -> int:
        """Honest exported size: quantized layers packed, everything else fp16."""
        total = 0
        quantized = set()
        for module in self.modules():
            if isinstance(module, BitLinear):
                total += module.packed_bytes()
                quantized.update(id(p) for p in module.parameters())
        for p in self.parameters():
            if id(p) not in quantized:
                total += p.numel() * 2  # fp16
        return total

    # -- the fun part -------------------------------------------------------

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, num_samples=1)], dim=1)
        return idx
