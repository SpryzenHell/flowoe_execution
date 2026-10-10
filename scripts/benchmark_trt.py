from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

import numpy as np
import torch

from flowoe_execution.model import CFMPolicy, FixedStepCryptoSampler

ROOT = Path(__file__).resolve().parents[1]


def accuracy_gate(max_abs_error: float, max_allowed_error: float) -> dict:
    if not np.isfinite(max_abs_error) or not np.isfinite(max_allowed_error) or max_abs_error < 0 or max_allowed_error <= 0:
        raise ValueError("accuracy errors must be finite; observed error non-negative and tolerance positive")
    passed = bool(max_abs_error <= max_allowed_error)
    return {
        "status": "passed" if passed else "failed_accuracy",
        "accuracy_gate_passed": passed,
        "max_abs_output_error": float(max_abs_error),
        "max_allowed_error": float(max_allowed_error),
    }


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
    # Match the export path exactly: TensorRT traces scalar time values,
    # not a batch-sized tensor produced by x[:, 0].
    fp = FixedStepCryptoSampler(model, args.steps, scalar_time=True).cuda().eval()
    trt_model = TRTModule()
    trt_model.load_state_dict(torch.load(engine_path, map_location="cuda", weights_only=False))

    # Use real L2 contexts when the checkpoint was trained on them; otherwise
    # use smoke data. Keep the exact inputs the same for the two implementations.
    real_l2_files = sorted((ROOT / "data/real/crypto").glob("*_l2.csv"))
    smoke_l2 = ROOT / "data/smoke/crypto_l2.csv"
    if checkpoint.name == "cfm_policy.pt":
        if not real_l2_files:
            raise SystemExit("Real checkpoint was requested, but no real crypto L2 input exists")
        l2_path = real_l2_files[0]
        source = f"real crypto L2 ({l2_path.stem})"
    elif checkpoint.name == "cfm_policy_smoke.pt":
        if not smoke_l2.exists():
            raise SystemExit("Smoke checkpoint was requested, but smoke L2 is missing")
        l2_path, source = smoke_l2, "synthetic smoke L2"
    elif real_l2_files:
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

    batch1 = next((case for case in cases if case["batch"] == 1), None)
    batch1_p99_wall = (
        float(batch1["tensorrt_int8"]["end_to_end_wall_ms"]["p99_ms"])
        if batch1 is not None else None
    )
    batch1_p99_gpu = (
        float(batch1["tensorrt_int8"]["cuda_event_ms"]["p99_ms"])
        if batch1 is not None else None
    )
    accuracy_passed = max_seen_error <= args.max_error
    target_met = bool(
        accuracy_passed and batch1_p99_wall is not None and batch1_p99_wall < 2.0
    )

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
        **accuracy_gate(max_seen_error, args.max_error),
        "cases": cases,
        "latency_target_p99_ms": 2.0,
        "batch1_p99_end_to_end_wall_ms": batch1_p99_wall,
        "batch1_p99_cuda_event_ms": batch1_p99_gpu,
        "sub_2ms_batch1_p99_target_met": target_met,
        "latency_claim_allowed": target_met,
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
