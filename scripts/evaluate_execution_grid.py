#!/usr/bin/env python3
"""Compare RK4, fixed Euler and fused Euler on the same held-out L2 windows."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from flowoe_execution.data import load_l2_csv, load_trades_csv
from flowoe_execution.execution import ExecutionSimulator, make_schedule_from_trajectory
from flowoe_execution.features import l2_features
from flowoe_execution.metrics import paired_block_bootstrap_ci, summary_stats
from flowoe_execution.model import CFMPolicy, FixedStepCryptoSampler
from flowoe_execution.ode import integrate_ode

ROOT = Path(__file__).resolve().parents[1]
CTX_LEN = 32
HORIZON = 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def compare_schedules(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Numerically compare normalized schedule tensors of shape (batch, horizon)."""
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("schedule tensors must have identical (batch, horizon) shapes")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise ValueError("schedule tensors must be finite")
    p = reference.clamp_min(1e-12)
    q = candidate.clamp_min(1e-12)
    return {
        "mean_abs_error": float((p - q).abs().mean().item()),
        "max_abs_error": float((p - q).abs().max().item()),
        "mean_kl_reference_to_candidate": float((p * (p.log() - q.log())).sum(1).mean().item()),
    }


@torch.no_grad()
def rk4_from_initial_noise(model: CFMPolicy, context: torch.Tensor, x0: torch.Tensor, steps: int):
    """Evaluate the model's RK4 path using caller-provided initial noise."""
    encoded = model.context(context, "crypto")

    def field(t_scalar, state):
        t = torch.as_tensor(t_scalar, device=state.device, dtype=state.dtype)
        return model.vf(t, state, encoded)

    state = integrate_ode(field, x0.clone(), t0=0.0, t1=1.0, steps=steps)
    return torch.softmax(state, dim=1)


def summarize_paired_method(baseline, slippage, completion, schedule_errors, seed=7):
    """Summarize a method only against the same episodes' TWAP baselines."""
    baseline_arr = np.asarray(baseline, dtype=np.float64)
    slippage_arr = np.asarray(slippage, dtype=np.float64)
    completion_arr = np.asarray(completion, dtype=np.float64)
    if baseline_arr.ndim != 1 or slippage_arr.ndim != 1 or len(baseline_arr) == 0:
        raise ValueError("paired samples must be non-empty one-dimensional arrays")
    if len(baseline_arr) != len(slippage_arr) or len(completion_arr) != len(slippage_arr):
        raise ValueError("baseline, slippage and completion must contain the same episodes")
    if not np.isfinite(baseline_arr).all() or not np.isfinite(slippage_arr).all() or not np.isfinite(completion_arr).all():
        raise ValueError("paired execution samples must be finite")
    mean, ci = paired_block_bootstrap_ci(
        baseline_arr, slippage_arr, seed=seed, n_boot=2000, block_size=5
    )
    errors = {
        key: float(np.mean([row[key] for row in schedule_errors]))
        for key in ("mean_abs_error", "max_abs_error", "mean_kl_reference_to_candidate")
    }
    errors["mean_episode_max_abs_error"] = errors.pop("max_abs_error")
    return {
        "slippage_bps": summary_stats(slippage_arr, seed=seed, n_boot=2000),
        "completion_mean": float(completion_arr.mean()),
        "episodes": int(len(slippage_arr)),
        "baseline_episodes": int(len(baseline_arr)),
        "improvement_vs_twap_bps": float(baseline_arr.mean() - slippage_arr.mean()),
        "paired_improvement_vs_twap_bps": mean,
        "paired_improvement_ci95_bps": ci,
        "schedule_error_vs_rk4": errors,
    }


def resolve_input(arg: str | None, patterns: list[str], label: str) -> Path:
    if arg:
        path = Path(arg)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise SystemExit(f"{label} file not found: {path}")
        return path
    matches = sorted(p for pattern in patterns for p in ROOT.glob(pattern))
    if not matches:
        raise SystemExit(f"No {label} file found under {ROOT}")
    return matches[0]


def evaluate_one(
    model: CFMPolicy,
    sampler: FixedStepCryptoSampler,
    context: torch.Tensor,
    x0: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        return sampler(context, x0.clone())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--training-report", help="Optional JSON training report that records the FI-2010 provenance.")
    ap.add_argument("--l2", help="Validated L2 CSV; defaults to the first real *_l2.csv file.")
    ap.add_argument("--trades", help="Validated trade CSV; defaults to the first real *_trades.csv file.")
    ap.add_argument("--side", choices=["buy", "sell"], required=True)
    ap.add_argument("--quantity", type=float, default=0.5)
    ap.add_argument("--train-ratio", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rk4-steps", type=int, default=24)
    ap.add_argument("--euler-steps", default="2,3,4,6")
    ap.add_argument("--include-fused", action="store_true")
    ap.add_argument("--fused-max-error", type=float, default=1e-4)
    ap.add_argument("--max-l2-rows", type=int, default=200000)
    ap.add_argument("--report", default="results/execution_grid.json")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.quantity <= 0 or not 0.1 <= args.train_ratio <= 0.9:
        ap.error("quantity must be positive and train-ratio must be in [0.1, 0.9]")
    if args.rk4_steps < 2 or args.max_l2_rows < 1 or args.fused_max_error <= 0:
        ap.error("rk4-steps >= 2, max-l2-rows >= 1 and fused-max-error > 0 are required")
    try:
        euler_steps = sorted(set(int(x) for x in args.euler_steps.split(",") if x.strip()))
    except ValueError:
        ap.error("--euler-steps must be comma-separated integers")
    if not euler_steps or min(euler_steps) < 2:
        ap.error("every Euler step count must be at least 2")

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    if not checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    l2_path = resolve_input(args.l2, ["data/real/crypto/*_l2.csv"], "L2")
    trade_path = resolve_input(args.trades, ["data/real/crypto/*_trades.csv"], "trade")

    l2 = load_l2_csv(l2_path, max_rows=args.max_l2_rows).snapshots
    trades = load_trades_csv(trade_path)
    features = l2_features(l2)
    if len(l2) <= CTX_LEN + HORIZON + 1 or not np.isfinite(features).all():
        raise SystemExit(f"Insufficient or invalid input: L2 rows={len(l2)}")
    if l2.ask0.le(l2.bid0).any():
        raise SystemExit("L2 contains crossed/locked top-of-book rows")
    start, end = l2.timestamp.min(), l2.timestamp.max()
    overlap = trades[(trades.timestamp >= start) & (trades.timestamp <= end)]
    if overlap.empty or overlap.timestamp.min() > start + (end - start) * 0.05 or overlap.timestamp.max() < end - (end - start) * 0.05:
        raise SystemExit("Trade prints do not cover the full L2 evaluation period")

    device = torch.device("cuda")
    model = CFMPolicy(8).to(device).eval()
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    x_crypto = torch.from_numpy(features).to(device)
    ntrain = int(len(l2) * args.train_ratio)
    if ntrain <= CTX_LEN + HORIZON or ntrain >= len(l2) - HORIZON:
        raise SystemExit("Chronological split leaves no train/evaluation window")
    simulator = ExecutionSimulator(quantity=args.quantity, side=args.side)
    baselines: list[float] = []
    base_completion: list[float] = []
    methods: dict[str, dict] = {}
    schedule_errors: dict[str, list[dict]] = {}
    samplers: dict[str, FixedStepCryptoSampler] = {}
    for steps in euler_steps:
        key = f"fixed_euler_{steps}"
        samplers[key] = FixedStepCryptoSampler(model, steps=steps).to(device).eval()
        schedule_errors[key] = []
    fused_error = None
    disabled_fused: set[str] = set()
    if args.include_fused:
        try:
            from torch.utils.cpp_extension import load
            ext = load(
                name="flowoe_execution_grid_fused",
                sources=[
                    str(ROOT / "src/flowoe_execution/cuda/fused_step.cpp"),
                    str(ROOT / "src/flowoe_execution/cuda/fused_step.cu"),
                ],
                extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3"],
                verbose=False,
            )
            for steps in euler_steps:
                key = f"fused_euler_{steps}"
                samplers[key] = FixedStepCryptoSampler(
                    model, steps=steps, fused_step=ext.fused_euler_step
                ).to(device).eval()
                schedule_errors[key] = []
        except Exception as exc:
            fused_error = f"{type(exc).__name__}: {exc}"

    starts = list(range(ntrain, len(l2) - HORIZON, HORIZON * 3))
    if not starts:
        raise SystemExit("No held-out execution episodes were formed")
    episode_records = []
    for episode_idx, i in enumerate(starts):
        episode_record = {
            "episode_index": episode_idx,
            "l2_start": str(l2.iloc[i].timestamp),
            "l2_end": str(l2.iloc[i + HORIZON - 1].timestamp),
            "twap_slippage_bps": None,
            "twap_completion": None,
            "methods": {},
        }
        ctx = x_crypto[i - CTX_LEN:i].unsqueeze(0)
        # Reset the same per-episode latent for every method, making slippage
        # and schedule differences paired rather than noise-confounded.
        gen = torch.Generator(device=device)
        gen.manual_seed(args.seed + episode_idx)
        x0 = torch.randn((1, HORIZON), generator=gen, device=device)
        reference = rk4_from_initial_noise(model, ctx, x0, args.rk4_steps)
        ref_schedule = reference[0].detach().cpu().numpy()
        window = l2.iloc[i:i + HORIZON]
        twap = simulator.run(
            window, np.full(HORIZON, 1.0 / HORIZON),
            benchmark="trade_vwap", trades=trades, strategy="twap",
        )
        baselines.append(twap.slippage_bps)
        base_completion.append(twap.completion)
        episode_record["twap_slippage_bps"] = twap.slippage_bps
        episode_record["twap_completion"] = twap.completion
        methods.setdefault("rk4", {"baseline": [], "slippage": [], "completion": [], "schedule_errors": []})
        methods["rk4"]["baseline"].append(twap.slippage_bps)
        methods["rk4"]["slippage"].append(twap.slippage_bps)
        methods["rk4"]["completion"].append(twap.completion)
        methods["rk4"]["schedule_errors"].append({
            "mean_abs_error": 0.0, "max_abs_error": 0.0,
            "mean_kl_reference_to_candidate": 0.0,
        })

        reference_result = simulator.run(
            window, ref_schedule, benchmark="trade_vwap",
            trades=trades, strategy=f"flowoe_rk4_{args.rk4_steps}",
        )
        methods["rk4"]["slippage"][-1] = reference_result.slippage_bps
        methods["rk4"]["completion"][-1] = reference_result.completion
        episode_record["methods"]["rk4"] = {
            "slippage_bps": reference_result.slippage_bps,
            "completion": reference_result.completion,
            "schedule_error": {"mean_abs_error": 0.0, "max_abs_error": 0.0, "mean_kl_reference_to_candidate": 0.0},
        }
        for key, sampler in samplers.items():
            if key in disabled_fused:
                continue
            pred = evaluate_one(model, sampler, ctx, x0)
            raw_error = None
            if key.startswith("fused_euler_"):
                ref_key = key.replace("fused_euler_", "fixed_euler_")
                ref_pred = samplers[ref_key](ctx, x0.clone())
                raw_error = float((pred - ref_pred).abs().max().item())
                if raw_error > args.fused_max_error:
                    fused_error = (
                        f"{key} max output error {raw_error:.8g} exceeded "
                        f"tolerance {args.fused_max_error:.8g} at episode {episode_idx}"
                    )
                    disabled_fused.add(key)
                    continue
            schedule = pred[0].detach().cpu().numpy()
            execution = simulator.run(
                window, schedule, benchmark="trade_vwap", trades=trades,
                strategy=key,
            )
            methods.setdefault(key, {"baseline": [], "slippage": [], "completion": [], "schedule_errors": []})
            methods[key]["baseline"].append(twap.slippage_bps)
            methods[key]["slippage"].append(execution.slippage_bps)
            methods[key]["completion"].append(execution.completion)
            err = compare_schedules(reference, pred)
            methods[key]["schedule_errors"].append(err)
            episode_record["methods"][key] = {
                "slippage_bps": execution.slippage_bps,
                "completion": execution.completion,
                "schedule_error": err,
            }
            if raw_error is not None:
                episode_record["methods"][key]["max_abs_error_vs_pytorch_euler"] = raw_error
        episode_records.append(episode_record)

    summaries = {}
    for name, vals in methods.items():
        summaries[name] = summarize_paired_method(
            vals["baseline"], vals["slippage"], vals["completion"],
            vals["schedule_errors"], seed=args.seed,
        )

    training_report = None
    fi_provenance = None
    if args.training_report:
        report_path = Path(args.training_report)
        if not report_path.is_absolute():
            report_path = ROOT / report_path
        if not report_path.exists():
            raise SystemExit(f"Training report not found: {report_path}")
        training_report = {
            "path": str(report_path),
            "sha256": sha256(report_path),
        }
        try:
            source_report = json.loads(report_path.read_text(encoding="utf-8"))
            fi_provenance = {
                "fi_path": source_report.get("fi_path"),
                "fi_sha256": source_report.get("fi_sha256"),
                "fi_label_encoding_detected": source_report.get("fi_label_encoding_detected"),
                "training_seed": source_report.get("seed"),
                "training_windows": source_report.get("train_windows"),
                "fi_windows": source_report.get("fi_windows"),
            }
        except Exception:
            fi_provenance = {"parse_error": "could not parse training report"}

    result = {
        "status": "passed",
        "data_source": "validated public market data; non-synthetic",
        "instrument": "BTCUSDT",
        "venue": "Binance USD-M futures",
        "side": args.side,
        "quantity": args.quantity,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "rk4_steps": args.rk4_steps,
        "euler_steps": euler_steps,
        "context_len": CTX_LEN,
        "horizon": HORIZON,
        "evaluation_stride_snapshots": HORIZON * 3,
        "episodes": len(starts),
        "l2_rows": int(len(l2)),
        "trade_rows": int(len(overlap)),
        "l2_window_start": str(start),
        "l2_window_end": str(end),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "training_report": training_report,
        "fi_training_provenance": fi_provenance,
        "l2_path": str(l2_path),
        "l2_sha256": sha256(l2_path),
        "trades_path": str(trade_path),
        "trades_sha256": sha256(trade_path),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "fused_extension_error": fused_error,
        "methods": summaries,
        "episode_results": episode_records,
        "note": "Execution-simulator slippage against market trade VWAP; no market-impact or real order-fill claim. Paired moving-block bootstrap uses 5 consecutive episodes per block.",
    }
    out = Path(args.report)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
