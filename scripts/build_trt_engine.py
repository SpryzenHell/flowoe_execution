"""Build an optional fixed-step INT8 TensorRT engine with representative calibration data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from flowoe_execution.model import CFMPolicy, FixedStepCryptoSampler

ROOT = Path(__file__).resolve().parents[1]


def calibration_contexts(context, x, count=32, prefer_real=False):
    """Use data that matches the checkpoint where available."""
    names = (
        ("data/real/crypto/BTCUSDT_l2.csv", "data/smoke/crypto_l2.csv")
        if prefer_real else
        ("data/smoke/crypto_l2.csv", "data/real/crypto/BTCUSDT_l2.csv")
    )
    for name in names:
        path = ROOT / name
        if not path.exists():
            continue
        try:
            from flowoe_execution.data import load_l2_csv
            from flowoe_execution.features import l2_features
            frame = load_l2_csv(path, max_rows=20000).snapshots
            features = l2_features(frame)
            if len(features) < context.shape[1]:
                continue
            starts = np.linspace(0, len(features) - context.shape[1], num=count, dtype=int)
            return [
                {
                    "context": torch.from_numpy(features[i:i + context.shape[1]]).unsqueeze(0).to(context.device),
                    "x": torch.randn_like(x),
                }
                for i in starts
            ], "real BTCUSDT L2" if "real/" in name else "synthetic smoke L2"
        except Exception:
            continue
    return [
        {"context": torch.randn_like(context), "x": torch.randn_like(x)}
        for _ in range(count)
    ], "random fallback (no valid L2 inputs found)"


def register_selu_converter():
    """Register the exact PyTorch SELU parameters with TensorRT's native SELU layer."""
    import tensorrt as trt
    from torch2trt_dynamic.torch2trt_dynamic import get_arg, tensorrt_converter, trt_

    @tensorrt_converter("torch.nn.functional.selu")
    def convert_selu(ctx):
        value = get_arg(ctx, "input", pos=0, default=None)
        layer = ctx.network.add_activation(
            trt_(ctx.network, value), trt.ActivationType.SELU
        )
        layer.alpha = 1.6732632423543772
        layer.beta = 1.0507009873554805
        ctx.method_return._trt = layer.get_output(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="results/cfm_policy_smoke.pt")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--output", default="results/flowoe_int8.pth")
    ap.add_argument("--calibration-size", type=int, default=32)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    if args.steps < 2 or args.calibration_size < 8:
        ap.error("steps must be at least 2 and calibration-size must be at least 8")
    try:
        import tensorrt as trt
        from torch2trt_dynamic import module2trt, BuildEngineConfig, SequenceDataset
        register_selu_converter()
    except Exception as exc:
        raise SystemExit(f"TensorRT or torch2trt_dynamic is unavailable: {exc}")

    checkpoint = ROOT / args.checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Model checkpoint not found: {checkpoint}")
    policy = CFMPolicy(8).cuda().eval()
    policy.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    wrapper = FixedStepCryptoSampler(policy, args.steps).cuda().eval()
    context = torch.zeros(1, 32, 45, device="cuda")
    x = torch.zeros(1, 8, device="cuda")
    calibration, calibration_source = calibration_contexts(
        context, x, args.calibration_size, prefer_real=checkpoint.name == "cfm_policy.pt"
    )

    cfg = BuildEngineConfig(
        shape_ranges={
            "context": {"min": (1, 32, 45), "opt": (1, 32, 45), "max": (4, 32, 45)},
            "x": {"min": (1, 8), "opt": (1, 8), "max": (4, 8)},
        },
        int8=True,
        int8_calib_dataset=SequenceDataset(calibration),
        int8_batch_size=1,
        int8_cache_file=str(ROOT / "results/flowoe_int8.cache"),
    )
    engine = module2trt(wrapper, args=[context, x], config=cfg)
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(engine.state_dict(), out)
    report = {
        "status": "built",
        "precision": "INT8",
        "tensorrt_version": trt.__version__,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "steps": args.steps,
        "calibration_samples": len(calibration),
        "calibration_source": calibration_source,
        "engine_path": str(out),
        "checkpoint_path": str(checkpoint),
    }
    (ROOT / "results/tensorrt_build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
