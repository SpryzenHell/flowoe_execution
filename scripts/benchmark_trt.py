from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch

from flowoe_execution.model import CFMPolicy, FixedStepCryptoSampler

ROOT = Path(__file__).resolve().parents[1]


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(x)),
        "p50_ms": float(np.percentile(x, 50)),
        "p95_ms": float(np.percentile(x, 95)),
        "p99_ms": float(np.percentile(x, 99)),
        "min_ms": float(np.min(x)),
        "max_ms": float(np.max(x)),
    }


def measure(fn, runs, warmup):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        gpu_ms, wall_ms = [], []
        for _ in range(runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            start.record()
            out = fn()
            end.record()
            torch.cuda.synchronize()
            wall_ms.append((time.perf_counter() - t0) * 1000.0)
            gpu_ms.append(start.elapsed_time(end))
    if not torch.isfinite(out).all():
        raise RuntimeError("TensorRT or PyTorch produced non-finite values")
    return {"cuda_event_ms": stats(gpu_ms), "end_to_end_wall_ms": stats(wall_ms)}, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="results/flowoe_int8.pth")
    ap.add_argument("--checkpoint", default="results/cfm_policy_smoke.pt")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--runs", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--max-error", type=float, default=0.05)
    ap.add_argument("--output", default="results/tensorrt_latency.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    if args.steps < 2 or args.runs < 1 or args.warmup < 0 or args.max_error <= 0:
        ap.error("steps >= 2, runs >= 1, warmup >= 0 and max-error > 0 are required")
    import tensorrt as trt
    from torch2trt_dynamic import TRTModule

    checkpoint = ROOT / args.checkpoint
    engine_path = ROOT / args.engine
    if not checkpoint.exists() or not engine_path.exists():
        raise SystemExit("TensorRT engine or matching checkpoint is missing")
    model = CFMPolicy(8).cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    fp = FixedStepCryptoSampler(model, args.steps).cuda().eval()
    trt_model = TRTModule()
    trt_model.load_state_dict(torch.load(engine_path, map_location="cuda", weights_only=False))

    # Use real L2 contexts when the checkpoint was trained on them; otherwise
    # use smoke data. Keep the exact inputs the same for the two implementations.
    real_l2_files = sorted((ROOT / "data/real/crypto").glob("*_l2.csv"))
    smoke_l2 = ROOT / "data/smoke/crypto_l2.csv"
    if checkpoint.name == "cfm_policy.pt" and real_l2_files:
        l2_path = real_l2_files[0]
        source = f"real crypto L2 ({l2_path.stem})"
    elif smoke_l2.exists():
        l2_path, source = smoke_l2, "synthetic smoke L2"
    else:
        raise SystemExit("No matching data for TensorRT validation")
    from flowoe_execution.data import load_l2_csv
    from flowoe_execution.features import l2_features
    feats = l2_features(load_l2_csv(l2_path, max_rows=20000).snapshots)
    if len(feats) < 36:
        raise SystemExit("Need at least 36 rows for TensorRT validation")

    cases = []
    max_seen_error = 0.0
    for batch in (1, 4):
        if batch > len(feats) - 31:
            continue
        starts = np.linspace(0, len(feats) - 32, num=batch, dtype=int)
        context = torch.stack([torch.from_numpy(feats[i:i + 32]) for i in starts]).cuda()
        x = torch.randn(batch, 8, device="cuda")
        fp_stats, fp_out = measure(lambda: fp(context, x.clone()), args.runs, args.warmup)
        trt_stats, trt_out = measure(lambda: trt_model(context, x.clone()), args.runs, args.warmup)
        if fp_out.shape != trt_out.shape:
            raise RuntimeError(f"Output shape mismatch: PyTorch={fp_out.shape}, TensorRT={trt_out.shape}")
        error = float((fp_out - trt_out).abs().max().item())
        max_seen_error = max(max_seen_error, error)
        cases.append({
            "batch": batch,
            "pytorch_fixed_step": fp_stats,
            "tensorrt_int8": trt_stats,
            "speedup_mean_wall": fp_stats["end_to_end_wall_ms"]["mean_ms"] / trt_stats["end_to_end_wall_ms"]["mean_ms"],
            "speedup_mean_cuda_event": fp_stats["cuda_event_ms"]["mean_ms"] / trt_stats["cuda_event_ms"]["mean_ms"],
            "max_abs_output_error": error,
        })

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "tensorrt_version": trt.__version__,
        "data_source": source,
        "checkpoint": str(checkpoint),
        "engine": str(engine_path),
        "steps": args.steps,
        "runs": args.runs,
        "warmup": args.warmup,
        "max_abs_output_error": max_seen_error,
        "max_allowed_error": args.max_error,
        "cases": cases,
    }
    out = Path(args.output)
    if not out.is_absolute():
        out = Path.cwd() / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if max_seen_error > args.max_error:
        raise SystemExit(f"TensorRT output error {max_seen_error} exceeds threshold {args.max_error}")


if __name__ == "__main__":
    main()
