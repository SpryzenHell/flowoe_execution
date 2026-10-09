import json
import numpy as np
import pandas as pd
import pytest

from flowoe_execution.data import load_l2_csv, load_l2_jsonl, load_trades_csv
from flowoe_execution.execution import ExecutionSimulator, book_vwap, market_vwap
from flowoe_execution.metrics import bootstrap_mean_ci, improvement_bps, percentile_ms, summary_stats


def make_book(ts=1790000000000):
    row = {"timestamp": ts}
    for i in range(10):
        row[f"bid{i}"] = 100.0 - i * 0.01
        row[f"bid_size{i}"] = 1.0 + i
        row[f"ask{i}"] = 100.02 + i * 0.01
        row[f"ask_size{i}"] = 1.5 + i
    return row


@pytest.mark.parametrize("stamp", [
    1790000000,
    1790000000000,
    1790000000000000,
    1790000000000000000,
])
def test_l2_timestamp_units_load_to_same_time(tmp_path, stamp):
    path = tmp_path / "book.csv"
    pd.DataFrame([make_book(stamp)]).to_csv(path, index=False)
    book = load_l2_csv(path).snapshots
    assert len(book) == 1
    assert book.timestamp.iloc[0].year == 2026
    assert book.timestamp.iloc[0].month == 9


def test_jsonl_loader_builds_top_ten_book(tmp_path):
    bids = [[100.0 - i * 0.01, 1.0 + i] for i in range(10)]
    asks = [[100.02 + i * 0.01, 1.5 + i] for i in range(10)]
    path = tmp_path / "book.jsonl"
    path.write_text(json.dumps({"timestamp": "2026-09-01T00:00:00Z", "bids": bids, "asks": asks}) + "\n", encoding="utf-8")
    book = load_l2_jsonl(path).snapshots
    assert book.shape == (1, 41)
    assert book.iloc[0].ask0 > book.iloc[0].bid0
    assert book.iloc[0].bid_size9 == 10.0


def test_trade_loader_accepts_time_and_quantity_columns(tmp_path):
    path = tmp_path / "trades.csv"
    pd.DataFrame({
        "time": [1790000000000000, 1790000000001000],
        "price": [100.0, 100.1],
        "quantity": [0.5, 0.25],
        "trade_id": [1, 2],
    }).to_csv(path, index=False)
    trades = load_trades_csv(path)
    assert list(trades.columns) == ["timestamp", "price", "qty"]
    assert len(trades) == 2
    assert trades.timestamp.is_monotonic_increasing
    assert trades.qty.sum() == pytest.approx(0.75)


def test_l2_loader_rejects_missing_level_columns(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"timestamp": "2026-09-01T00:00:00Z", "bid0": 100}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Missing L2 columns"):
        load_l2_csv(path)


def test_trade_loader_rejects_missing_price_or_quantity(tmp_path):
    path = tmp_path / "bad-trades.csv"
    pd.DataFrame([{"timestamp": "2026-09-01T00:00:00Z", "price": 100}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="price and qty"):
        load_trades_csv(path)


def test_bootstrap_statistics_and_improvement_are_deterministic():
    x = np.array([1.0, 1.5, 2.0, 2.5, 3.0])
    mean, (low, high) = bootstrap_mean_ci(x, seed=19, n_boot=500)
    assert mean == pytest.approx(x.mean())
    assert low <= mean <= high
    stats = summary_stats(x, seed=19, n_boot=500)
    assert stats["mean"] == pytest.approx(mean)
    assert stats["median"] == pytest.approx(2.0)
    assert stats["p95"] == pytest.approx(2.9)
    assert stats["ci95"] == pytest.approx((low, high))
    assert improvement_bps(2.5, 1.75) == pytest.approx(0.75)
    assert percentile_ms([1, 2, 3, 4], q=50) == pytest.approx(2.5)


@pytest.mark.parametrize("bad", [[], [float("nan")], [float("inf")]])
def test_summary_statistics_reject_empty_or_nonfinite_values(bad):
    with pytest.raises(ValueError):
        summary_stats(bad)


def test_vwap_helpers_reject_empty_windows():
    row = pd.DataFrame([make_book(1790000000000)])
    row["timestamp"] = pd.to_datetime(row["timestamp"], unit="ms", utc=True)
    start = row.timestamp.iloc[0] + pd.Timedelta(days=1)
    end = start + pd.Timedelta(seconds=1)
    trades = pd.DataFrame({
        "timestamp": [row.timestamp.iloc[0]],
        "price": [100.0],
        "qty": [1.0],
    })
    with pytest.raises(ValueError, match="No L2 snapshots"):
        book_vwap(row, start, end)
    with pytest.raises(ValueError, match="No trades"):
        market_vwap(trades, start, end)


def test_execution_rejects_bad_constructor_and_benchmark():
    with pytest.raises(ValueError, match="quantity"):
        ExecutionSimulator(quantity=0)
    with pytest.raises(ValueError, match="levels"):
        ExecutionSimulator(levels=0)
    with pytest.raises(ValueError, match="side"):
        ExecutionSimulator(side="hold")
    row = pd.DataFrame([make_book(i) for i in range(8)])
    row["timestamp"] = pd.date_range("2026-09-01", periods=8, freq="s", tz="UTC")
    with pytest.raises(ValueError, match="benchmark"):
        ExecutionSimulator().run(row, np.ones(8), benchmark="unknown")
