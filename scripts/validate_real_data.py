from __future__ import annotations
from pathlib import Path
import argparse
import json
import numpy as np
from flowoe_execution.data import load_fi2010, load_l2_csv, load_trades_csv

ROOT = Path(__file__).resolve().parents[1]

def main():
    ap = argparse.ArgumentParser(description="Validate FlowOE real-data inputs before an evidence run.")
    ap.add_argument("--fi", required=True)
    ap.add_argument("--l2", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--fi-max-rows", type=int, default=100000)
    ap.add_argument("--l2-max-rows", type=int, default=1000000)
    args = ap.parse_args()
    fi = load_fi2010(args.fi, max_rows=args.fi_max_rows)
    l2 = load_l2_csv(args.l2, max_rows=args.l2_max_rows)
    trades = load_trades_csv(args.trades)
    mids = (l2.snapshots.bid0 + l2.snapshots.ask0).to_numpy(float)
    spreads_bps = (l2.snapshots.ask0 - l2.snapshots.bid0) / np.maximum(mids, 1e-12) * 1e4
    start, end = l2.snapshots.timestamp.min(), l2.snapshots.timestamp.max()
    overlap = trades[(trades.timestamp >= start) & (trades.timestamp <= end)]
    report = {
        "fi_rows": int(len(fi.features)), "fi_feature_dim": int(fi.features.shape[1]),
        "fi_label_dim": int(fi.labels.shape[1]), "fi_label_min": int(fi.labels.min()),
        "fi_label_max": int(fi.labels.max()), "l2_rows": int(len(l2.snapshots)),
        "l2_start": str(start), "l2_end": str(end),
        "l2_median_spread_bps": float(np.median(spreads_bps)),
        "l2_median_top_level_bid_size": float(np.median(l2.snapshots.bid_size0)),
        "l2_median_top_level_ask_size": float(np.median(l2.snapshots.ask_size0)),
        "trade_rows_total": int(len(trades)), "trade_rows_in_l2_window": int(len(overlap)),
        "trade_start": str(trades.timestamp.min()), "trade_end": str(trades.timestamp.max()),
    }
    if len(overlap) == 0:
        raise SystemExit("No trades overlap the L2 evaluation period; refusing evidence run.")
    out = ROOT / "results"; out.mkdir(exist_ok=True)
    (out / "data_quality.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
