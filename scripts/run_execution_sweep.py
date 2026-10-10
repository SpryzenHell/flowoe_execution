#!/usr/bin/env python3
"""Run reproducible FlowOE real-data seeds/sides and paired sampler evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def split_ints(value: str, label: str) -> list[int]:
    try:
        values = sorted(set(int(x.strip()) for x in value.split(",") if x.strip()))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError(f"{label} must not be empty")
    return values


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_input(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise SystemExit(f"{label} file not found: {path}")
    return path


def run_command(cmd: list[str], log_path: Path) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    with log_path.open("w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    return {
        "command": cmd,
        "log": str(log_path),
        "return_code": int(proc.returncode),
        "status": "passed" if proc.returncode == 0 else "failed",
        "started_utc": started,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def aggregate_reports(records: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        data = load_json(Path(record["evaluation_report"]))
        if data.get("status") != "passed":
            continue
        side = data["side"]
        for method, summary in data["methods"].items():
            groups.setdefault((side, method), []).append({
                "seed": int(data["seed"]),
                "mean_slippage_bps": float(summary["slippage_bps"]["mean"]),
                "mean_improvement_bps": float(summary["paired_improvement_vs_twap_bps"]),
                "ci95": summary["paired_improvement_ci95_bps"],
                "completion_mean": float(summary["completion_mean"]),
                "episodes": int(summary["episodes"]),
                "schedule_error": summary["schedule_error_vs_rk4"],
            })
    result = {}
    for (side, method), rows in sorted(groups.items()):
        improvements = [r["mean_improvement_bps"] for r in rows]
        slips = [r["mean_slippage_bps"] for r in rows]
        result.setdefault(side, {})[method] = {
            "seeds": [r["seed"] for r in rows],
            "successful_seed_runs": len(rows),
            "episodes_per_seed": [r["episodes"] for r in rows],
            "mean_slippage_bps_across_seeds": statistics.fmean(slips),
            "between_seed_slippage_sd_bps": statistics.stdev(slips) if len(slips) > 1 else 0.0,
            "min_seed_slippage_bps": min(slips),
            "max_seed_slippage_bps": max(slips),
            "mean_paired_improvement_vs_twap_bps": statistics.fmean(improvements),
            "between_seed_improvement_sd_bps": statistics.stdev(improvements) if len(improvements) > 1 else 0.0,
            "min_seed_improvement_bps": min(improvements),
            "max_seed_improvement_bps": max(improvements),
            "seed_runs": rows,
        }
    return result


def write_markdown(path: Path, manifest: dict) -> None:
    lines = [
        "# FlowOE Real-Data Execution Sweep",
        "",
        f"- Completed UTC: {manifest['completed_utc']}",
        f"- Dataset note: {manifest['dataset_note']}",
        f"- FI-2010 input SHA-256: {manifest['fi_sha256']}",
        f"- L2 input SHA-256: {manifest['l2_sha256']}",
        f"- Trade input SHA-256: {manifest['trades_sha256']}",
        f"- Seeds: {', '.join(str(x) for x in manifest['seeds'])}",
        f"- Sides: {', '.join(manifest['sides'])}",
        f"- Euler step counts: {', '.join(str(x) for x in manifest['euler_steps'])}",
        f"- Fused path requested: {manifest['include_fused']}",
        "",
        "The table averages per-seed results. Confidence intervals remain per seed and use a paired moving-block bootstrap over contiguous evaluation episodes. The market execution simulator is not a live exchange order-fill model.",
        "",
        "## Results by side and sampler",
        "",
        "| Side | Method | Successful seeds | Episodes/seed | Mean slippage (bps) | Mean paired improvement vs TWAP (bps) | Between-seed SD of improvement (bps) | Seed improvement range (bps) |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for side, methods in manifest["aggregate"].items():
        for method, values in methods.items():
            lines.append(
                f"| {side} | {method} | {values['successful_seed_runs']} | "
                f"{','.join(str(x) for x in values['episodes_per_seed'])} | "
                f"{values['mean_slippage_bps_across_seeds']:.6f} | "
                f"{values['mean_paired_improvement_vs_twap_bps']:.6f} | "
                f"{values['between_seed_improvement_sd_bps']:.6f} | "
                f"[{values['min_seed_improvement_bps']:.6f}, {values['max_seed_improvement_bps']:.6f}] |"
            )
    lines += [
        "",
        "## Run status",
        "",
        "| Seed | Side | Training | Evaluation |",
        "|---:|---|---|---|",
    ]
    for run in manifest["runs"]:
        lines.append(
            f"| {run['seed']} | {run['side']} | {run['training']['status']} | "
            f"{run.get('evaluation', {}).get('status', 'not run')} |"
        )
    lines += [
        "",
        "## Interpretation rules",
        "",
        "- Do not claim a slippage improvement unless paired intervals, across-seed variability and completion rate support it.",
        "- Synthetic tests and latencies must never be mixed into this real-data table.",
        "- Fixed-step speedups are useful only when their schedule-quality error and slippage remain acceptable.",
        "- Fused Euler results are recorded only after its output matches the PyTorch Euler implementation within tolerance.",
        "- TensorRT latency is recorded separately after successful conversion and accuracy validation.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fi", required=True)
    ap.add_argument("--l2", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--sides", default="buy,sell")
    ap.add_argument("--euler-steps", default="2,3,4,6")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--quantity", type=float, default=0.5)
    ap.add_argument("--train-ratio", type=float, default=0.60)
    ap.add_argument("--include-fused", action="store_true")
    ap.add_argument("--out-dir", default="results/execution_sweep")
    args = ap.parse_args()
    if args.epochs < 1 or args.quantity <= 0 or not 0.1 <= args.train_ratio <= 0.9:
        ap.error("epochs/quantity must be positive and train-ratio must be between 0.1 and 0.9")
    seeds = split_ints(args.seeds, "seeds")
    euler_steps = split_ints(args.euler_steps, "euler-steps")
    if min(euler_steps) < 2:
        ap.error("every Euler step count must be at least 2")
    sides = [x.strip() for x in args.sides.split(",") if x.strip()]
    if not sides or any(x not in {"buy", "sell"} for x in sides):
        ap.error("sides must be a comma-separated subset of buy,sell")

    fi = resolve_input(args.fi, "FI-2010")
    l2 = resolve_input(args.l2, "L2")
    trades = resolve_input(args.trades, "trades")
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_note": "FI-2010 plus sequence-checked BTCUSDT USD-M futures depth/trade inputs; slippage is simulator output, not a live-fill result.",
        "fi_path": str(fi), "fi_sha256": file_hash(fi),
        "l2_path": str(l2), "l2_sha256": file_hash(l2),
        "trades_path": str(trades), "trades_sha256": file_hash(trades),
        "seeds": seeds, "sides": sides, "euler_steps": euler_steps,
        "epochs": args.epochs, "quantity": args.quantity, "train_ratio": args.train_ratio,
        "include_fused": args.include_fused, "runs": [],
    }

    total_failures = 0
    for seed in seeds:
        for side in sides:
            stem = f"seed{seed}_{side}"
            train_report = out_dir / f"train_{stem}.json"
            checkpoint = out_dir / f"model_{stem}.pt"
            train_log = out_dir / f"train_{stem}.log"
            train_cmd = [
                sys.executable, "scripts/run_experiment.py",
                "--fi", str(fi), "--l2", str(l2), "--trades", str(trades),
                "--epochs", str(args.epochs), "--device", "cuda",
                "--fi-max-rows", "50000", "--l2-max-rows", "200000",
                "--train-ratio", str(args.train_ratio), "--quantity", str(args.quantity),
                "--instrument", "BTCUSDT", "--side", side, "--seed", str(seed),
                "--data-source", "public_futures",
                "--sampler", "rk4", "--sampling-steps", "24",
                "--report-out", str(train_report), "--checkpoint-out", str(checkpoint),
            ]
            training = run_command(train_cmd, train_log)
            record = {
                "seed": seed, "side": side, "training": training,
                "training_report": str(train_report), "checkpoint": str(checkpoint),
            }
            if training["return_code"] == 0 and train_report.exists() and checkpoint.exists():
                eval_report = out_dir / f"eval_{stem}.json"
                eval_log = out_dir / f"eval_{stem}.log"
                eval_cmd = [
                    sys.executable, "scripts/evaluate_execution_grid.py",
                    "--checkpoint", str(checkpoint), "--training-report", str(train_report),
                    "--l2", str(l2), "--trades", str(trades),
                    "--data-source", "public_futures",
                    "--side", side, "--quantity", str(args.quantity),
                    "--train-ratio", str(args.train_ratio), "--seed", str(seed),
                    "--euler-steps", ",".join(str(x) for x in euler_steps),
                    "--max-l2-rows", "200000", "--report", str(eval_report),
                ]
                if args.include_fused:
                    eval_cmd.append("--include-fused")
                evaluation = run_command(eval_cmd, eval_log)
                record["evaluation"] = evaluation
                record["evaluation_report"] = str(eval_report)
                if evaluation["return_code"] != 0 or not eval_report.exists():
                    total_failures += 1
            else:
                total_failures += 1
                record["evaluation"] = {"status": "skipped", "reason": "training failed or report/checkpoint missing"}
            manifest["runs"].append(record)

    # Preserve one conventional real checkpoint for the separate TensorRT step.
    preferred = out_dir / f"model_{min(seeds)}_buy.pt"
    if preferred.exists():
        target = ROOT / "results/cfm_policy.pt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(preferred.read_bytes())
        manifest["canonical_checkpoint"] = str(target)
        manifest["canonical_checkpoint_seed"] = min(seeds)
        manifest["canonical_checkpoint_side"] = "buy"

    eval_records = [
        r for r in manifest["runs"]
        if r.get("evaluation", {}).get("return_code") == 0
        and Path(r.get("evaluation_report", "")).exists()
    ]
    manifest["aggregate"] = aggregate_reports(eval_records)
    manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_markdown(out_dir / "SWEEP_RESULTS.md", manifest)
    print(json.dumps({
        "status": "passed" if total_failures == 0 else "failed",
        "runs": len(manifest["runs"]),
        "failed_runs": total_failures,
        "summary": str(out_dir / "SWEEP_RESULTS.md"),
        "manifest": str(out_dir / "sweep_manifest.json"),
    }, indent=2))
    if total_failures:
        raise SystemExit(f"{total_failures} training/evaluation runs failed; see {out_dir / 'sweep_manifest.json'}")


if __name__ == "__main__":
    main()
