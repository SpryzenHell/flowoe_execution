from __future__ import annotations

import torch
from torch import nn

from .ode import integrate_ode


class ContextEncoder(nn.Module):
    """Dual-source temporal encoder for FI-2010 and crypto L2 contexts."""

    def __init__(self, hidden: int = 96):
        super().__init__()
        self.fi = nn.Sequential(
            nn.Linear(144, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
        )
        self.crypto = nn.Sequential(
            nn.Linear(45, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
        )
        # Shared temporal block makes FI-2010 auxiliary training useful to
        # the downstream crypto execution encoder without mixing dimensions.
        self.temporal = nn.Sequential(
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.SELU(),
        )

    def _encode(self, x: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected context shaped (N,T,D), got {tuple(x.shape)}")
        z = proj(x).transpose(1, 2)
        return self.temporal(z).mean(dim=2)

    def forward(self, x: torch.Tensor, source: str) -> torch.Tensor:
        if source == "fi2010":
            return self._encode(x, self.fi)
        if source == "crypto":
            return self._encode(x, self.crypto)
        raise ValueError(f"unknown source: {source}")

    def crypto_encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encode(x, self.crypto)


class VectorField(nn.Module):
    """Time-conditioned vector field for execution trajectory transport."""

    def __init__(self, horizon: int = 8, context_dim: int = 96, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(horizon + context_dim + 1, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, horizon),
        )

    def forward(self, t: torch.Tensor, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        t = t.reshape(-1, 1)
        return self.net(torch.cat([x, ctx, t], dim=1))


class CFMPolicy(nn.Module):
    """Conditional Flow Matching policy for order-execution trajectories."""

    def __init__(self, horizon: int = 8, context_hidden: int = 96, hidden: int = 128):
        super().__init__()
        self.horizon = horizon
        self.context = ContextEncoder(context_hidden)
        self.vf = VectorField(horizon, context_hidden, hidden)
        self.fi_head = nn.Sequential(
            nn.Linear(context_hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, 15),
        )

    def fi_aux_loss(self, context: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Auxiliary FI-2010 three-way movement classification at five horizons."""
        if labels.ndim != 2 or labels.shape[1] != 5:
            raise ValueError(f"expected labels shaped (N, 5), got {tuple(labels.shape)}")
        labels = labels.long()
        if labels.min() >= 1 and labels.max() <= 3:
            labels = labels - 1
        if labels.min() < 0 or labels.max() > 2:
            raise ValueError("FI-2010 labels must be encoded as 1/2/3 or 0/1/2")
        ctx = self.context(context, "fi2010")
        logits = self.fi_head(ctx).view(labels.shape[0], 5, 3)
        return torch.nn.functional.cross_entropy(logits.transpose(1, 2), labels)

    def cfm_loss(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        source: str,
        sigma: float = 0.05,
    ) -> torch.Tensor:
        ctx = self.context(context, source)
        x0 = torch.randn_like(target)
        t = torch.rand(target.shape[0], device=target.device)
        tt = t[:, None]
        xt = (1.0 - tt) * x0 + tt * target + sigma * torch.randn_like(target)
        return torch.mean((self.vf(t, xt, ctx) - (target - x0)) ** 2)

    @torch.no_grad()
    def sample(
        self,
        context: torch.Tensor,
        source: str,
        steps: int = 32,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        self.eval()
        ctx = self.context(context, source)
        x = torch.randn(
            context.shape[0],
            self.horizon,
            device=context.device,
            dtype=context.dtype,
        ) * temperature

        def field(t_scalar, z):
            t = torch.as_tensor(t_scalar, device=z.device, dtype=z.dtype)
            return self.vf(t, z, ctx)

        return integrate_ode(field, x, t0=0.0, t1=1.0, steps=steps)


class ProbabilityFlowODEPolicy(nn.Module):
    """Conditional VP score model with a reverse-time probability-flow ODE."""

    def __init__(self, horizon: int = 8, context_hidden: int = 96, hidden: int = 128):
        super().__init__()
        self.horizon = horizon
        self.context = ContextEncoder(context_hidden)
        self.score = nn.Sequential(
            nn.Linear(horizon + context_hidden + 1, hidden),
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, horizon),
        )

    @staticmethod
    def beta(t: torch.Tensor) -> torch.Tensor:
        return 0.1 + 19.9 * t

    def score_net(self, t: torch.Tensor, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        return self.score(torch.cat([x, ctx, t.reshape(-1, 1)], dim=1))

    def score_loss(
        self,
        context: torch.Tensor,
        target: torch.Tensor,
        source: str,
    ) -> torch.Tensor:
        ctx = self.context(context, source)
        t = torch.rand(target.shape[0], device=target.device)
        integral_beta = 0.1 * t + 9.95 * t * t
        alpha = torch.exp(-0.5 * integral_beta)
        sigma = torch.sqrt(torch.clamp(1.0 - alpha * alpha, min=1e-6))
        eps = torch.randn_like(target)
        xt = alpha[:, None] * target + sigma[:, None] * eps
        predicted = self.score_net(t, xt, ctx)
        return torch.mean((predicted + eps / sigma[:, None]) ** 2)

    @torch.no_grad()
    def sample(self, context: torch.Tensor, source: str, steps: int = 64) -> torch.Tensor:
        self.eval()
        ctx = self.context(context, source)
        x = torch.randn(
            context.shape[0],
            self.horizon,
            device=context.device,
            dtype=context.dtype,
        )

        def drift(t_scalar, z):
            t = torch.full(
                (z.shape[0],),
                float(t_scalar),
                device=z.device,
                dtype=z.dtype,
            )
            beta = self.beta(t)
            return -0.5 * beta[:, None] * (z + self.score_net(t, z, ctx))

        return integrate_ode(drift, x, t0=1.0, t1=0.0, steps=steps)


class FixedStepCryptoSampler(nn.Module):
    """Fixed-step sampler suitable for TensorRT/CUDA benchmarking."""

    def __init__(self, policy: CFMPolicy, steps: int = 16, fused_step=None):
        super().__init__()
        if steps < 2:
            raise ValueError("steps must be at least 2")
        self.context = policy.context
        self.vf = policy.vf
        self.steps = steps
        self.fused_step = fused_step

    def forward(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctx = self.context.crypto_encode(context)
        dt = 1.0 / float(self.steps - 1)
        for i in range(self.steps - 1):
            t = x.new_full((x.shape[0],), float(i) * dt)
            v = self.vf(t, x, ctx)
            if self.fused_step is None:
                x = x + dt * v
            else:
                if not x.is_contiguous():
                    x = x.contiguous()
                if not v.is_contiguous():
                    v = v.contiguous()
                self.fused_step(x, v, dt)
        return torch.softmax(x, dim=1)
