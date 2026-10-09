from __future__ import annotations
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from flowoe_execution.data import load_fi2010, load_l2_csv, load_trades_csv

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Validate real data before training or reporting results.")
    ap.add_argument("--fi", required=True)
    ap.add_argument("--l2", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--fi-max-rows", type=int, default=100000)
    ap.add_argument("--l2-max-rows", type=int, default=1000000)
    args = ap.parse_args()
    for name, value in (("fi-max-rows", args.fi_max_rows), ("l2-max-rows", args.l2_max_rows)):
        if value <= 0:
            ap.error(f"--{name} must be positive")

    fi = load_fi2010(args.fi, max_rows=args.fi_max_rows)
    l2 = load_l2_csv(args.l2, max_rows=args.l2_max_rows).snapshots
    trades = load_trades_csv(args.trades)
    if len(fi.features) < 1 or fi.features.shape[1] != 144 or fi.labels.shape[1] != 5:
        raise SystemExit("FI-2010 must have rows, 144 features and five labels")
    if not np.isfinite(fi.features).all():
        raise SystemExit("FI-2010 features contain non-finite values")
    labels = set(np.unique(fi.labels).tolist())
    if not labels.issubset({-1, 0, 1, 2, 3}):
        raise SystemExit(f"Unexpected FI-2010 label encoding: {sorted(labels)}")

    if len(l2) < 100:
        raise SystemExit(f"L2 window is too short: {len(l2)} rows")
    price_cols = [f"{side}{i}" for i in range(10) for side in ("bid", "ask")]
    size_cols = [f"{side}_size{i}" for i in range(10) for side in ("bid", "ask")]
    prices = l2[price_cols].to_numpy(dtype=np.float64)
    sizes = l2[size_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or not np.isfinite(sizes).all():
        raise SystemExit("L2 prices or sizes contain non-finite values")
    if (prices <= 0).any():
        raise SystemExit("L2 has non-positive price levels")
    if (sizes < 0).any():
        raise SystemExit("L2 has negative displayed sizes")
    if (l2.ask0 <= l2.bid0).any():
        raise SystemExit("L2 contains crossed or locked top-of-book rows")
    for i in range(9):
        if not (l2[f"bid{i}"] > l2[f"bid{i+1}"]).all():
            raise SystemExit(f"L2 bid levels are not strictly descending at level {i}")
        if not (l2[f"ask{i}"] < l2[f"ask{i+1}"]).all():
            raise SystemExit(f"L2 ask levels are not strictly ascending at level {i}")
    if l2.timestamp.duplicated().any():
        raise SystemExit("L2 has duplicate snapshot timestamps")

    if trades.empty:
        raise SystemExit("Trade input is empty")
    trade_values = trades[["price", "qty"]].to_numpy(dtype=np.float64)
    if not np.isfinite(trade_values).all() or (trade_values <= 0).any():
        raise SystemExit("Trade prices and quantities must be finite and positive")

    start, end = l2.timestamp.min(), l2.timestamp.max()
    overlap = trades[(trades.timestamp >= start) & (trades.timestamp <= end)]
    if len(overlap) == 0:
        raise SystemExit("No trades overlap the L2 evaluation period")
    if overlap.timestamp.min() > start + (end - start) * 0.05:
        raise SystemExit("Trade data starts too late to cover the L2 window")
    if overlap.timestamp.max() < end - (end - start) * 0.05:
        raise SystemExit("Trade data ends too early to cover the L2 window")

    mid = (l2.bid0 + l2.ask0).to_numpy(float) / 2
    spread_bps = (l2.ask0 - l2.bid0).to_numpy(float) / np.maximum(mid, 1e-12) * 1e4
    report = {
        "status": "passed",
        "fi_path": str(Path(args.fi)),
        "fi_sha256": sha256(args.fi),
        "fi_rows_loaded": int(len(fi.features)),
        "fi_feature_dim": int(fi.features.shape[1]),
        "fi_label_dim": int(fi.labels.shape[1]),
        "fi_labels_seen": sorted(int(x) for x in labels),
        "l2_path": str(Path(args.l2)),
        "l2_sha256": sha256(args.l2),
        "l2_rows_loaded": int(len(l2)),
        "l2_start": str(start),
        "l2_end": str(end),
        "l2_window_seconds": float((end - start).total_seconds()),
        "l2_median_spread_bps": float(np.median(spread_bps)),
        "l2_median_top_bid_size": float(np.median(l2.bid_size0)),
        "l2_median_top_ask_size": float(np.median(l2.ask_size0)),
        "trade_path": str(Path(args.trades)),
        "trade_sha256": sha256(args.trades),
        "trade_rows_total": int(len(trades)),
        "trade_rows_in_l2_window": int(len(overlap)),
        "trade_coverage_start": str(overlap.timestamp.min()),
        "trade_coverage_end": str(overlap.timestamp.max()),
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
