from __future__ import annotations
from pathlib import Path
import argparse, json, time
import numpy as np
import torch
from torch.utils.cpp_extension import load

ROOT = Path(__file__).resolve().parents[1]

def stats(values):
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p99_ms": float(np.percentile(values, 99)),
    }

def main():
    ap = argparse.ArgumentParser(description="Benchmark the fused CUDA Euler update against PyTorch add.")
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--elements", type=int, default=8 * 1024)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    ext = load(
        name="flowoe_fused",
        sources=[
            str(ROOT / "src/flowoe_execution/cuda/fused_step.cpp"),
            str(ROOT / "src/flowoe_execution/cuda/fused_step.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=False,
    )
    x0 = torch.randn(args.elements, device="cuda")
    v = torch.randn_like(x0)
    dt = 1.0 / 15.0

    for _ in range(args.warmup):
        x = x0.clone()
        ext.fused_euler_step(x, v, dt)
        x = x0.clone()
        x.add_(v, alpha=dt)
    torch.cuda.synchronize()

    fused = []
    native = []
    for _ in range(args.runs):
        x = x0.clone()
        t = time.perf_counter(); ext.fused_euler_step(x, v, dt); torch.cuda.synchronize()
        fused.append((time.perf_counter() - t) * 1e3)
    for _ in range(args.runs):
        x = x0.clone()
        t = time.perf_counter(); x.add_(v, alpha=dt); torch.cuda.synchronize()
        native.append((time.perf_counter() - t) * 1e3)

    x1 = x0.clone()
    ext.fused_euler_step(x1, v, dt)
    x2 = x0.clone()
    x2.add_(v, alpha=dt)
    result = {
        "fused_cuda": stats(fused),
        "torch_add": stats(native),
        "speedup_mean": float(np.mean(native) / np.mean(fused)),
        "max_abs_error": float((x1 - x2).abs().max().item()),
        "gpu": torch.cuda.get_device_name(0),
        "elements": args.elements,
    }
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    (out / "cuda_fused_latency.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
