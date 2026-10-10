from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from flowoe_execution.model import CFMPolicy, FixedStepCryptoSampler

ROOT = Path(__file__).resolve().parents[1]


def summarize(values):
    x = np.asarray(values, dtype=np.float64)
    if not x.size or not np.isfinite(x).all():
        raise ValueError("Timing samples must be non-empty and finite")
    return {
        "mean_ms": float(x.mean()),
        "p50_ms": float(np.percentile(x, 50)),
        "p95_ms": float(np.percentile(x, 95)),
        "p99_ms": float(np.percentile(x, 99)),
        "min_ms": float(x.min()),
        "max_ms": float(x.max()),
    }


def measure(fn, runs, warmup):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        gpu_ms, wall_ms = [], []
        out = None
        for _ in range(runs):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            begin.record()
            out = fn()
            end.record()
            torch.cuda.synchronize()
            gpu_ms.append(begin.elapsed_time(end))
            wall_ms.append((time.perf_counter() - t0) * 1000.0)
    if out is None or not torch.isfinite(out).all():
        raise RuntimeError("Sampler produced no output or non-finite values")
    return {
        "cuda_event_ms": summarize(gpu_ms),
        "end_to_end_wall_ms": summarize(wall_ms),
    }, out


@torch.no_grad()
def rk4_schedule(model, context, x0, steps=64):
    """High-accuracy fixed-grid RK4 reference, initialized from the same noise."""
    encoded = model.context.crypto_encode(context)
    x = x0.clone()
    dt = 1.0 / float(steps)
    batch = x.shape[0]
    for i in range(steps):
        t0 = x.new_full((batch,), i * dt)
        th = x.new_full((batch,), (i + 0.5) * dt)
        t1 = x.new_full((batch,), (i + 1.0) * dt)
        k1 = model.vf(t0, x, encoded)
        k2 = model.vf(th, x + 0.5 * dt * k1, encoded)
        k3 = model.vf(th, x + 0.5 * dt * k2, encoded)
        k4 = model.vf(t1, x + dt * k3, encoded)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return torch.softmax(x, dim=1)


def select_inputs(batch, context_len, checkpoint_arg):
    real_l2 = ROOT / "data/real/crypto/BTCUSDT_l2.csv"
    smoke_l2 = ROOT / "data/smoke/crypto_l2.csv"
    real_ckpt = ROOT / "results/cfm_policy.pt"
    smoke_ckpt = ROOT / "results/cfm_policy_smoke.pt"
    if checkpoint_arg != "auto":
        checkpoint = Path(checkpoint_arg)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.exists():
            raise SystemExit(f"Checkpoint not found: {checkpoint}")
        l2_path = real_l2 if checkpoint.name == "cfm_policy.pt" and real_l2.exists() else smoke_l2
        source = "real BTCUSDT L2" if l2_path == real_l2 else "synthetic smoke L2"
    elif real_l2.exists() and real_ckpt.exists():
        l2_path, checkpoint, source = real_l2, real_ckpt, "real BTCUSDT L2"
    elif smoke_l2.exists() and smoke_ckpt.exists():
        l2_path, checkpoint, source = smoke_l2, smoke_ckpt, "synthetic smoke L2"
    else:
        raise SystemExit("No matching L2 data and trained checkpoint were found")

    from flowoe_execution.data import load_l2_csv
    from flowoe_execution.features import l2_features
    features = l2_features(load_l2_csv(l2_path, max_rows=50000).snapshots)
    if len(features) < context_len:
        raise SystemExit(f"Need at least {context_len} L2 rows; found {len(features)}")
    starts = np.linspace(0, len(features) - context_len, num=batch, dtype=int)
    context = torch.stack([
        torch.from_numpy(features[i:i + context_len]) for i in starts
    ]).cuda()
    model = CFMPolicy(8).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    return model, context, source, l2_path, checkpoint


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark fixed-step Euler latency and schedule error against RK4."
    )
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--steps", default="2,3,4,6,8,12,16,24")
    ap.add_argument("--context-len", type=int, default=32)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--reference-steps", type=int, default=64)
    ap.add_argument("--checkpoint", default="auto")
    ap.add_argument("--output", default="results/fixed_step_quality_latency.json")
    ap.add_argument("--max-output-error", type=float, default=1e-4)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if min(args.context_len, args.runs, args.reference_steps) <= 0 or args.warmup < 0:
        ap.error("context-len, runs and reference-steps must be positive; warmup non-negative")
    batches = sorted(set(int(x) for x in args.batches.split(",") if x.strip()))
    steps = sorted(set(int(x) for x in args.steps.split(",") if x.strip()))
    if not batches or min(batches) < 1 or not steps or min(steps) < 2:
        ap.error("batches must be positive; every sampler step count must be at least 2")

    from benchmark_cuda_fused import build_extension
    ext = build_extension()
    torch.manual_seed(701)
    torch.cuda.manual_seed_all(701)

    results = []
    for batch in batches:
        model, context, source, l2_path, checkpoint = select_inputs(
            batch, args.context_len, args.checkpoint
        )
        x0 = torch.randn(batch, 8, device="cuda")
        reference = rk4_schedule(model, context, x0, steps=args.reference_steps)
        for nsteps in steps:
            normal = FixedStepCryptoSampler(model, steps=nsteps).cuda().eval()
            fused = FixedStepCryptoSampler(
                model, steps=nsteps, fused_step=ext.fused_euler_step
            ).cuda().eval()
            normal_stats, normal_out = measure(
                lambda: normal(context, x0.clone()), args.runs, args.warmup
            )
            fused_stats, fused_out = measure(
                lambda: fused(context, x0.clone()), args.runs, args.warmup
            )
            if (normal_out - fused_out).abs().max().item() > args.max_output_error:
                raise RuntimeError(
                    f"Fused output mismatch batch={batch} steps={nsteps}: "
                    f"{(normal_out - fused_out).abs().max().item()}"
                )
            def quality(out):
                q = out.clamp_min(1e-12)
                p = reference.clamp_min(1e-12)
                return {
                    "mean_abs_schedule_error_vs_rk4": float((q - p).abs().mean().item()),
                    "max_abs_schedule_error_vs_rk4": float((q - p).abs().max().item()),
                    "mean_kl_reference_to_euler": float((p * (p.log() - q.log())).sum(1).mean().item()),
                    "mean_schedule_sum": float(q.sum(1).mean().item()),
                    "minimum_schedule_weight": float(q.min().item()),
                }
            results.append({
                "batch": batch,
                "steps": nsteps,
                "context_len": args.context_len,
                "feature_dim": 45,
                "source": source,
                "l2_path": str(l2_path),
                "checkpoint": str(checkpoint),
                "reference": "RK4, same initial noise, reference_steps=" + str(args.reference_steps),
                "runs": args.runs,
                "warmup": args.warmup,
                "reference_schedule": quality(reference),
                "pytorch_euler": {**normal_stats, **quality(normal_out)},
                "fused_euler": {**fused_stats, **quality(fused_out)},
                "fused_vs_pytorch_max_abs_error": float((normal_out - fused_out).abs().max().item()),
                "speedup_mean_wall": normal_stats["end_to_end_wall_ms"]["mean_ms"] / fused_stats["end_to_end_wall_ms"]["mean_ms"],
                "speedup_mean_cuda_event": normal_stats["cuda_event_ms"]["mean_ms"] / fused_stats["cuda_event_ms"]["mean_ms"],
            })

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "reference_steps": args.reference_steps,
        "batches": batches,
        "steps": steps,
        "runs": args.runs,
        "warmup": args.warmup,
        "results": results,
        "note": "Synthetic metrics are not real-market evidence. Quality error compares normalized schedules from fixed-step Euler against a same-noise RK4 reference.",
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
