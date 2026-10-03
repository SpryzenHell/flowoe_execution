from __future__ import annotations
import torch
from torch import nn
from .ode import integrate_ode


class ContextEncoder(nn.Module):
    """Dual-source temporal encoder for FI-2010 and crypto L2."""
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.fi = nn.Sequential(nn.Linear(144, hidden), nn.SELU(), nn.Linear(hidden, hidden))
        self.crypto = nn.Sequential(nn.Linear(45, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.fi_temporal = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1), nn.SiLU())
        self.crypto_temporal = nn.Sequential(nn.Conv1d(hidden, hidden, 3, padding=1), nn.SiLU())

    def _encode(self, x, proj, temporal):
        return temporal(proj(x).transpose(1, 2)).mean(dim=2)

    def forward(self, x, source):
        if source == "fi2010":
            return self._encode(x, self.fi, self.fi_temporal)
        if source == "crypto":
            return self._encode(x, self.crypto, self.crypto_temporal)
        raise ValueError(f"unknown source: {source}")

    def crypto_encode(self, x):
        return self._encode(x, self.crypto, self.crypto_temporal)


class VectorField(nn.Module):
    def __init__(self, horizon=8, context_dim=96, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(horizon + context_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, horizon)
        )

    def forward(self, t, x, ctx):
        t = t.expand(x.shape[0]).reshape(-1, 1) if t.ndim == 0 else t.reshape(-1, 1)
        return self.net(torch.cat([x, ctx, t], dim=1))


class CFMPolicy(nn.Module):
    """Conditional Flow Matching policy for order-execution trajectories."""
    def __init__(self, horizon=8, context_hidden=96, hidden=128):
        super().__init__()
        self.horizon = horizon
        self.context = ContextEncoder(context_hidden)
        self.vf = VectorField(horizon, context_hidden, hidden)

    def cfm_loss(self, context, target, source, sigma=0.05):
        ctx = self.context(context, source)
        x0 = torch.randn_like(target)
        t = torch.rand(target.shape[0], device=target.device)
        tt = t[:, None]
        xt = (1 - tt) * x0 + tt * target + sigma * torch.randn_like(target)
        return torch.mean((self.vf(t, xt, ctx) - (target - x0)) ** 2)

    @torch.no_grad()
    def sample(self, context, source, steps=32, temperature=1.0):
        self.eval()
        ctx = self.context(context, source)
        x = torch.randn(context.shape[0], self.horizon, device=context.device) * temperature
        return integrate_ode(
            lambda t, z: self.vf(torch.as_tensor(t, device=z.device, dtype=z.dtype), z, ctx),
            x, steps=steps
        )


class ProbabilityFlowODEPolicy(nn.Module):
    """Conditional VP score model with reverse-time probability-flow ODE."""
    def __init__(self, horizon=8, context_hidden=96, hidden=128):
        super().__init__()
        self.horizon = horizon
        self.context = ContextEncoder(context_hidden)
        self.score = nn.Sequential(
            nn.Linear(horizon + context_hidden + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, horizon)
        )

    @staticmethod
    def beta(t):
        return 0.1 + 19.9 * t

    def score_net(self, t, x, ctx):
        return self.score(torch.cat([x, ctx, t.reshape(-1, 1)], dim=1))

    def score_loss(self, context, target, source):
        ctx = self.context(context, source)
        t = torch.rand(target.shape[0], device=target.device)
        integ = 0.1 * t + 9.95 * t * t
        alpha = torch.exp(-0.5 * integ)
        sigma = torch.sqrt(torch.clamp(1 - alpha * alpha, min=1e-6))
        eps = torch.randn_like(target)
        xt = alpha[:, None] * target + sigma[:, None] * eps
        return torch.mean((self.score_net(t, xt, ctx) + eps / sigma[:, None]) ** 2)

    @torch.no_grad()
    def sample(self, context, source, steps=64):
        self.eval()
        ctx = self.context(context, source)
        x = torch.randn(context.shape[0], self.horizon, device=context.device)

        def drift(t_scalar, z):
            t = torch.full((z.shape[0],), float(t_scalar), device=z.device, dtype=z.dtype)
            b = self.beta(t)
            return -0.5 * b[:, None] * (z + self.score_net(t, z, ctx))

        return integrate_ode(drift, x, t0=1.0, t1=0.0, steps=steps)


class FixedStepCryptoSampler(nn.Module):
    """Fixed-step CUDA/TensorRT-friendly execution sampler."""
    def __init__(self, policy: CFMPolicy, steps=16):
        super().__init__()
        self.context = policy.context
        self.vf = policy.vf
        self.steps = steps

    def forward(self, context, x):
        ctx = self.context.crypto_encode(context)
        dt = 1.0 / float(self.steps - 1)
        for i in range(self.steps - 1):
            t = x.new_full((x.shape[0],), float(i) * dt)
            x = x + dt * self.vf(t, x, ctx)
        return torch.softmax(x, dim=1)
