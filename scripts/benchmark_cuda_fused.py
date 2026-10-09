from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from flowoe_execution.model import CFMPolicy, FixedStepCryptoSampler

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    x = sorted(float(v) for v in values)
    if not x:
        raise ValueError("Cannot summarize an empty timing sample")
    return {
        "mean_ms": sum(x) / len(x),
        "p50_ms": x[min(len(x) - 1, int(0.50 * len(x)))],
        "p95_ms": x[min(len(x) - 1, int(0.95 * len(x)))],
        "p99_ms": x[min(len(x) - 1, int(0.99 * len(x)))],
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
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--runs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--output", default="results/cuda_fused_latency.json")
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
    model = CFMPolicy(8).cuda().eval()
    ref = FixedStepCryptoSampler(model, steps=args.steps).cuda().eval()
    fused = FixedStepCryptoSampler(model, steps=args.steps, fused_step=ext.fused_euler_step).cuda().eval()
    context = torch.randn(args.batch, args.context_len, 45, device="cuda")
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
