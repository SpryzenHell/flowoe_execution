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
    return {
        "mean_ms": sum(x) / len(x),
        "p50_ms": x[len(x) // 2],
        "p95_ms": x[max(0, int(0.95 * len(x)) - 1)],
        "p99_ms": x[max(0, int(0.99 * len(x)) - 1)],
    }


def timed(fn, runs):
    values = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - t0) * 1e3)
    return stats(values), y


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
    ap = argparse.ArgumentParser(description="Benchmark fused CUDA Euler updates against the PyTorch reference.")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--context-len", type=int, default=32)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--runs", type=int, default=300)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    ext = build_extension()
    torch.manual_seed(7)
    model = CFMPolicy(8).cuda().eval()
    ref = FixedStepCryptoSampler(model, steps=args.steps).cuda().eval()
    fused = FixedStepCryptoSampler(model, steps=args.steps, fused_step=ext.fused_euler_step).cuda().eval()
    context = torch.randn(args.batch, args.context_len, 45, device="cuda")
    x0 = torch.randn(args.batch, 8, device="cuda")

    with torch.no_grad():
        for _ in range(25):
            _ = ref(context, x0.clone())
            _ = fused(context, x0.clone())
        ref_stats, ref_out = timed(lambda: ref(context, x0.clone()), args.runs)
        fused_stats, fused_out = timed(lambda: fused(context, x0.clone()), args.runs)

    result = {
        "batch": args.batch,
        "context_len": args.context_len,
        "steps": args.steps,
        "runs": args.runs,
        "gpu": torch.cuda.get_device_name(0),
        "reference": ref_stats,
        "fused": fused_stats,
        "speedup_mean": ref_stats["mean_ms"] / fused_stats["mean_ms"],
        "max_abs_output_error": float((ref_out - fused_out).abs().max().item()),
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "cuda_fused_latency.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
