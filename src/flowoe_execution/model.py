from __future__ import annotations

import torch
from torch import nn

from .ode import integrate_ode
from .labels import normalize_fi2010_labels


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
    """Time-conditioned vector field with an export-friendly scalar-time path.

    The time coefficient is stored separately from the input projection. This is
    algebraically identical to a linear layer over [x, context, t], but avoids
    slicing the first-layer weight matrix during TensorRT tracing.
    """

    def __init__(self, horizon: int = 8, context_dim: int = 96, hidden: int = 128):
        super().__init__()
        # Initialize as the original combined projection, then split its time
        # column once at construction. This preserves the original initialization
        # distribution and exact operation for a fixed seed.
        original = nn.Linear(horizon + context_dim + 1, hidden)
        self.input = nn.Linear(horizon + context_dim, hidden)
        self.time_weight = nn.Parameter(torch.empty(hidden))
        with torch.no_grad():
            self.input.weight.copy_(original.weight[:, :-1])
            self.time_weight.copy_(original.weight[:, -1])
            self.input.bias.copy_(original.bias)
        self.net = nn.Sequential(
            nn.SELU(),
            nn.Linear(hidden, hidden),
            nn.SELU(),
            nn.Linear(hidden, horizon),
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        # Migrate checkpoints made with the prior Sequential Linear/SELU/Linear
        # layout. This keeps older smoke/real checkpoints loadable without
        # invoking a dynamic tensor slice in an export/trace.
        legacy = prefix + "net.0.weight"
        current = prefix + "input.weight"
        if legacy in state_dict and current not in state_dict:
            w = state_dict.pop(legacy)
            state_dict[current] = w[:, :-1].contiguous()
            state_dict[prefix + "time_weight"] = w[:, -1].contiguous()
            old_bias = prefix + "net.0.bias"
            state_dict[prefix + "input.bias"] = state_dict.pop(old_bias)
            for old_idx, new_idx in ((2, 1), (4, 3)):
                for param in ("weight", "bias"):
                    old_key = prefix + f"net.{old_idx}.{param}"
                    new_key = prefix + f"net.{new_idx}.{param}"
                    state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    def forward(self, t: torch.Tensor | float, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        z = self.input(torch.cat([x, ctx], dim=1))
        if isinstance(t, (float, int)):
            z = z + float(t) * self.time_weight
        else:
            if t.ndim == 0:
                t = t.expand(x.shape[0])
            t = t.reshape(-1, 1)
            z = z + t * self.time_weight
        return self.net(z)


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

    def fi_aux_loss(
        self,
        context: torch.Tensor,
        labels: torch.Tensor,
        label_encoding: str = "auto",
    ) -> torch.Tensor:
        """Auxiliary FI-2010 movement classification at five horizons.

        Labels are normalized to classes 0/1/2 explicitly. For auto-detection,
        pass the complete FI-2010 label matrix rather than a random minibatch,
        or specify its encoding. This avoids incorrect shifts when a minibatch
        happens to contain only a subset of the three classes.
        """
        if labels.ndim != 2 or labels.shape[1] != 5:
            raise ValueError(f"expected labels shaped (N, 5), got {tuple(labels.shape)}")
        labels, _ = normalize_fi2010_labels(labels, label_encoding)
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

    def __init__(self, policy: CFMPolicy, steps: int = 4, fused_step=None, scalar_time: bool = False):
        super().__init__()
        if steps < 2:
            raise ValueError("steps must be at least 2")
        self.context = policy.context
        self.vf = policy.vf
        self.steps = steps
        self.fused_step = fused_step
        self.scalar_time = bool(scalar_time)

    def forward(self, context: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctx = self.context.crypto_encode(context)
        dt = 1.0 / float(self.steps - 1)
        for i in range(self.steps - 1):
            # TensorRT's dynamic getitem converter has an int64/int32 shape
            # issue for x[:, 0]. The scalar-time mode takes the equivalent
            # algebraic path in VectorField without slicing a dynamic input.
            t = float(i) * dt if self.scalar_time else torch.full_like(x[:, 0], float(i) * dt)
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
