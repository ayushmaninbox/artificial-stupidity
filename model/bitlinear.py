"""BitLinear — nn.Linear that quantizes its own weights during training.

This is the heart of the whole project. The critical idea, which is easy to
get wrong:

  You cannot train a normal FP32 model and then round the weights to 1 bit.
  That produces noise, not a dumb model. The network has to *know* it's going
  to be quantized while it learns.

So we use quantization-aware training with a straight-through estimator:

    forward:   y = x @ quantize(W).T        <- the network only ever sees quantized weights
    backward:  dL/dW flows straight through the (non-differentiable) rounding

We keep a full-precision `self.weight` as a "shadow" parameter that the
optimizer updates. It never gets used at inference — at export time we throw
it away and keep only the packed bits + one scale per output row.

Reference: BitNet (Wang et al. 2023) and BitNet b1.58 (Ma et al. 2024).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def ste(x_quant: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator: value of x_quant, gradient of x."""
    return x + (x_quant - x).detach()


# ---------------------------------------------------------------------------
# weight quantizers. all are symmetric around zero and use a per-output-row
# scale, so each row of W is reconstructed as (integer codes) * scale[row].
# ---------------------------------------------------------------------------

def quant_binary(w: torch.Tensor) -> torch.Tensor:
    """1 bit. Every weight becomes +s or -s. There is no zero. There is no mercy."""
    scale = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
    # sign() returns 0 for exactly 0.0, which would smuggle in a third state.
    # torch.where keeps it honestly binary.
    codes = torch.where(w >= 0, 1.0, -1.0)
    return codes * scale


def quant_ternary(w: torch.Tensor) -> torch.Tensor:
    """1.58 bits: {-1, 0, +1}. BitNet b1.58 absmean scaling."""
    scale = w.abs().mean(dim=-1, keepdim=True).clamp(min=1e-5)
    codes = (w / scale).round().clamp(-1, 1)
    return codes * scale


def quant_int(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric integer quantization to `bits` bits, per output row."""
    qmax = 2 ** (bits - 1) - 1
    scale = (w.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-8)
    codes = (w / scale).round().clamp(-qmax - 1, qmax)
    return codes * scale


def quantize_weight(w: torch.Tensor, bits: float) -> torch.Tensor:
    if bits >= 32:
        return w
    if bits == 1:
        return quant_binary(w)
    if abs(bits - 1.58) < 1e-6:
        return quant_ternary(w)
    return quant_int(w, int(bits))


def quantize_activation(x: torch.Tensor, bits: float) -> torch.Tensor:
    """Per-token absmax activation quantization (BitNet quantizes these to 8 bit)."""
    if bits >= 32:
        return x
    qmax = 2 ** (bits - 1) - 1
    scale = (x.abs().amax(dim=-1, keepdim=True) / qmax).clamp(min=1e-8)
    return (x / scale).round().clamp(-qmax - 1, qmax) * scale


# ---------------------------------------------------------------------------

class BitLinear(nn.Module):
    """Drop-in nn.Linear replacement with a precision dial.

    weight_bits=32 makes this behave exactly like nn.Linear, so AS-0 is a
    genuine control group running the identical code path.
    """

    def __init__(self, in_features, out_features, bias=False,
                 weight_bits=32, act_bits=32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.act_bits = act_bits

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

        # BitNet normalizes activations before quantizing; it keeps low-bit
        # training from diverging. Only pay for it when we're actually quantizing.
        self.norm = nn.LayerNorm(in_features, bias=False) if weight_bits < 32 else None

    def forward(self, x):
        if self.norm is not None:
            x = self.norm(x)
        if self.act_bits < 32:
            x = ste(quantize_activation(x, self.act_bits), x)

        w = self.weight
        if self.weight_bits < 32:
            w = ste(quantize_weight(w, self.weight_bits), w)

        return F.linear(x, w, self.bias)

    def extra_repr(self):
        return (f"in={self.in_features}, out={self.out_features}, "
                f"w_bits={self.weight_bits}, a_bits={self.act_bits}")

    # -- storage accounting -------------------------------------------------

    def packed_bytes(self) -> int:
        """Real on-disk size of this layer once exported, in bytes.

        Not the theoretical number — this counts the actual bit-packed codes
        plus the fp16 scales we need to reconstruct the weights.
        """
        n = self.out_features * self.in_features
        if self.weight_bits >= 32:
            wb = n * 4                                   # fp32
        elif abs(self.weight_bits - 1.58) < 1e-6:
            # 5 ternary values pack into one uint8 (3^5 = 243 <= 256)
            wb = math.ceil(n / 5)
            wb += self.out_features * 2                  # fp16 scale per row
        else:
            bits = int(self.weight_bits)
            wb = math.ceil(n * bits / 8)
            wb += self.out_features * 2                  # fp16 scale per row
        if self.bias is not None:
            wb += self.out_features * 2
        if self.norm is not None:
            wb += self.in_features * 2
        return wb
