from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load

from flowoe_execution.model import CFMPolicy, FixedStepCryptoSampler

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Timing samples must be non-empty, finite and one-dimensional")
    return {
        "mean_ms": float(np.mean(x)),
        "p50_ms": float(np.percentile(x, 50)),
        "p95_ms": float(np.percentile(x, 95)),
        "p99_ms": float(np.percentile(x, 99)),
        "min_ms": float(np.min(x)),
        "max_ms": float(np.max(x)),
    }


def timed(fn, runs, warmup=20):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        gpu_values, wall_values = [], []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            start.record()
            y = fn()
            end.record()
            torch.cuda.synchronize()
            wall_values.append((time.perf_counter() - t0) * 1000.0)
            gpu_values.append(start.elapsed_time(end))
    if not torch.isfinite(y).all():
        raise RuntimeError("Sampler returned non-finite output")
    return {
        "cuda_event_ms": stats(gpu_values),
        "end_to_end_wall_ms": stats(wall_values),
    }, y


def build_extension():
    return load(
        name="flowoe_fused_bench",
        sources=[
            str(ROOT / "src/flowoe_execution/cuda/fused_step.cpp"),
            str(ROOT / "src/flowoe_execution/cuda/fused_step.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )


def main():
    ap = argparse.ArgumentParser(description="Compare PyTorch and CUDA Euler updates in the full sampler.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--context-len", type=int, default=32)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--output", default="results/cuda_fused_latency.json")
    ap.add_argument("--checkpoint", default="auto")
    args = ap.parse_args()
    if min(args.batch, args.context_len, args.steps, args.runs, args.warmup) <= 0:
        ap.error("batch, context length, steps, runs and warmup must be positive")
    if args.steps < 2:
        ap.error("steps must be at least 2")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    ext = build_extension()
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)

    real_l2 = ROOT / "data/real/crypto/BTCUSDT_l2.csv"
    smoke_l2 = ROOT / "data/smoke/crypto_l2.csv"
    real_ckpt = ROOT / "results/cfm_policy.pt"
    smoke_ckpt = ROOT / "results/cfm_policy_smoke.pt"
    if args.checkpoint != "auto":
        checkpoint = Path(args.checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = ROOT / checkpoint
        if not checkpoint.exists():
            raise SystemExit(f"Checkpoint not found: {checkpoint}")
        if checkpoint.name == "cfm_policy.pt" and real_l2.exists():
            l2_path, source = real_l2, "real BTCUSDT L2"
        else:
            l2_path = real_l2 if real_l2.exists() else smoke_l2
            source = "real BTCUSDT L2" if l2_path == real_l2 else "synthetic smoke L2"
    elif real_l2.exists() and real_ckpt.exists():
        l2_path, checkpoint, source = real_l2, real_ckpt, "real BTCUSDT L2"
    elif smoke_l2.exists() and smoke_ckpt.exists():
        l2_path, checkpoint, source = smoke_l2, smoke_ckpt, "synthetic smoke L2"
    else:
        raise SystemExit("No matching L2 data and trained checkpoint were found")

    from flowoe_execution.data import load_l2_csv
    from flowoe_execution.features import l2_features
    features = l2_features(load_l2_csv(l2_path, max_rows=20000).snapshots)
    if len(features) < args.context_len:
        raise SystemExit(f"Need at least {args.context_len} L2 rows, found {len(features)}")
    starts = np.linspace(0, len(features) - args.context_len, num=args.batch, dtype=int)
    context = torch.stack([
        torch.from_numpy(features[i:i + args.context_len]) for i in starts
    ]).cuda()
    model = CFMPolicy(8).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    ref = FixedStepCryptoSampler(model, steps=args.steps).cuda().eval()
    fused = FixedStepCryptoSampler(model, steps=args.steps, fused_step=ext.fused_euler_step).cuda().eval()
    x0 = torch.randn(args.batch, 8, device="cuda")

    ref_stats, ref_out = timed(lambda: ref(context, x0.clone()), args.runs, args.warmup)
    fused_stats, fused_out = timed(lambda: fused(context, x0.clone()), args.runs, args.warmup)
    max_error = float((ref_out - fused_out).abs().max().item())
    if max_error > 1e-4:
        raise RuntimeError(f"Fused output differs from reference: max_abs_error={max_error}")

    result = {
        "batch": args.batch,
        "context_len": args.context_len,
        "steps": args.steps,
        "runs": args.runs,
        "warmup": args.warmup,
        "gpu": torch.cuda.get_device_name(0),
        "data_source": source,
        "l2_path": str(l2_path),
        "checkpoint": str(checkpoint),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "reference": ref_stats,
        "fused": fused_stats,
        "speedup_mean_end_to_end": ref_stats["end_to_end_wall_ms"]["mean_ms"] / fused_stats["end_to_end_wall_ms"]["mean_ms"],
        "speedup_mean_cuda_event": ref_stats["cuda_event_ms"]["mean_ms"] / fused_stats["cuda_event_ms"]["mean_ms"],
        "max_abs_output_error": max_error,
        "note": "Only the Euler update is moved to a CUDA extension; the vector-field network still runs in PyTorch."
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
