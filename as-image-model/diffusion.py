"""Noise schedule, training target and sampler.

Two choices here are what make 8-step generation possible, and both are worth
stating because the obvious defaults do not work at this size:

1. v-prediction instead of epsilon-prediction. Predicting the noise is fine at
   50+ steps but degenerates at the high-noise end, where x_t is nearly pure
   noise and "predict the noise" is asking the model to echo its own input.
   v = alpha*eps - sigma*x0 stays a well-scaled target at every timestep, so
   the first of eight steps is still informative.

2. Cosine schedule instead of linear. Linear spends most of its timesteps in
   a range where the image is already destroyed, which is wasted capacity when
   you only have a few million parameters to spend.
"""

import math

import torch


class Diffusion:
    def __init__(self, timesteps=1000, schedule="cosine", device="cpu"):
        self.T = timesteps
        if schedule == "cosine":
            s = 0.008
            t = torch.linspace(0, 1, timesteps + 1, dtype=torch.float64)
            f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            acp = (f / f[0]).clamp(1e-9, 1.0)[1:]
        else:
            beta = torch.linspace(1e-4, 0.02, timesteps, dtype=torch.float64)
            acp = torch.cumprod(1.0 - beta, dim=0)
        self.acp = acp.float().to(device)
        self.alpha = self.acp.sqrt()
        self.sigma = (1 - self.acp).sqrt()

    def _g(self, arr, t, ndim):
        return arr[t].view(-1, *([1] * (ndim - 1)))

    def q_sample(self, x0, t, noise):
        a = self._g(self.alpha, t, x0.ndim)
        s = self._g(self.sigma, t, x0.ndim)
        return a * x0 + s * noise

    def v_target(self, x0, t, noise):
        a = self._g(self.alpha, t, x0.ndim)
        s = self._g(self.sigma, t, x0.ndim)
        return a * noise - s * x0

    def to_x0_eps(self, x_t, t, v):
        a = self._g(self.alpha, t, x_t.ndim)
        s = self._g(self.sigma, t, x_t.ndim)
        x0 = a * x_t - s * v
        eps = s * x_t + a * v
        return x0, eps

    @torch.no_grad()
    def ddim(self, model, shape, ctx, ctx_pad, ctx_null, ctx_null_pad,
             steps=8, guidance=3.0, device="cpu", generator=None, clamp=3.0):
        """Deterministic DDIM. `steps` forward passes, not `self.T`."""
        x = torch.randn(shape, device=device, generator=generator)
        ts = torch.linspace(self.T - 1, 0, steps + 1).round().long().to(device)

        for i in range(steps):
            t_cur, t_next = ts[i], ts[i + 1]
            tb = t_cur.repeat(shape[0])

            if guidance != 1.0:
                # one batched pass for conditional + unconditional
                v = model(torch.cat([x, x]), torch.cat([tb, tb]),
                          torch.cat([ctx, ctx_null]),
                          torch.cat([ctx_pad, ctx_null_pad]))
                v_c, v_u = v.chunk(2)
                v = v_u + guidance * (v_c - v_u)
            else:
                v = model(x, tb, ctx, ctx_pad)

            x0, eps = self.to_x0_eps(x, tb, v)
            x0 = x0.clamp(-clamp, clamp)
            # recompute eps from the clamped x0 so the two stay consistent
            a = self._g(self.alpha, tb, x.ndim)
            s = self._g(self.sigma, tb, x.ndim)
            eps = (x - a * x0) / s.clamp(min=1e-8)

            a_n = self.alpha[t_next]
            s_n = self.sigma[t_next]
            x = a_n * x0 + s_n * eps

        return x


class EMA:
    """Shadow weights. Diffusion samples are visibly better from the average."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(),
                                                     alpha=1 - self.decay)

    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in self.shadow.items():
            sd[k].copy_(v.to(sd[k].dtype))
