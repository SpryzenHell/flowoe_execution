from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

def softmax(x: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float64) - np.max(x)) / max(temp, 1e-6)
    e = np.exp(z)
    return e / np.maximum(e.sum(), 1e-12)

def make_schedule_from_trajectory(x: np.ndarray) -> np.ndarray:
    return softmax(x)

def market_vwap(trades: pd.DataFrame, start, end) -> float:
    t = trades[(trades.timestamp >= start) & (trades.timestamp <= end)]
    if t.empty or float(t.qty.sum()) <= 0:
        raise ValueError("No trades in requested execution window")
    return float((t.price * t.qty).sum() / t.qty.sum())

def book_vwap(snapshots: pd.DataFrame, start, end) -> float:
    s = snapshots[(snapshots.timestamp >= start) & (snapshots.timestamp <= end)]
    if s.empty:
        raise ValueError("No L2 snapshots in execution window")
    mid = (s.bid0 + s.ask0) / 2
    w = s.bid_size0 + s.ask_size0
    return float((mid * w).sum() / max(w.sum(), 1e-12))

@dataclass(frozen=True)
class ExecutionResult:
    strategy: str
    side: str
    requested_qty: float
    executed_qty: float
    avg_price: float
    benchmark_vwap: float
    slippage_bps: float
    completion: float
    levels_consumed: int

class ExecutionSimulator:
    """Replay child-order schedules against the full configured L2 depth."""

    def __init__(self, quantity: float = 1.0, levels: int = 10, side: str = "buy"):
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if levels < 1:
            raise ValueError("levels must be positive")
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        self.quantity = float(quantity)
        self.levels = int(levels)
        self.side = side

    def _consume_snapshot(self, row, qty: float) -> tuple[float, float, int]:
        if qty <= 0:
            return 0.0, 0.0, 0
        if self.side == "buy":
            levels = [(float(row[f"ask{i}"]), float(row[f"ask_size{i}"])) for i in range(self.levels)]
        else:
            levels = [(float(row[f"bid{i}"]), float(row[f"bid_size{i}"])) for i in range(self.levels)]

        cost = filled = 0.0
        used = 0
        for price, available in levels:
            if available <= 0:
                continue
            take = min(qty - filled, available)
            cost += take * price
            filled += take
            used += 1
            if filled >= qty - 1e-12:
                break
        return filled, cost, used

    def run(
        self,
        snapshots: pd.DataFrame,
        fractions: np.ndarray,
        benchmark: str = "trade_vwap",
        trades: pd.DataFrame | None = None,
        strategy: str = "flowoe",
    ) -> ExecutionResult:
        if len(snapshots) < len(fractions):
            raise ValueError("Not enough snapshots for requested horizon")

        f = np.asarray(fractions, dtype=np.float64)
        if f.ndim != 1 or len(f) == 0 or not np.isfinite(f).all() or np.any(f < 0):
            raise ValueError("fractions must be a finite non-negative 1-D array")
        total = float(f.sum())
        if total <= 0:
            raise ValueError("schedule fractions must have positive mass")
        f /= total

        target = self.quantity * f
        remaining = self.quantity
        cost = executed = 0.0
        levels_consumed = 0

        for i, requested in enumerate(target):
            if remaining <= 1e-12:
                break
            filled, level_cost, used = self._consume_snapshot(snapshots.iloc[i], min(float(requested), remaining))
            executed += filled
            cost += level_cost
            levels_consumed += used
            remaining -= filled

        if executed <= 0:
            raise ValueError("Schedule did not execute any quantity")

        avg = cost / executed
        start = snapshots.iloc[0].timestamp
        end = snapshots.iloc[min(len(f) - 1, len(snapshots) - 1)].timestamp
        if benchmark == "trade_vwap":
            if trades is None:
                raise ValueError("trade_vwap requested without trade data")
            vwap = market_vwap(trades, start, end)
        elif benchmark == "book_vwap":
            vwap = book_vwap(snapshots, start, end)
        else:
            raise ValueError("benchmark must be 'trade_vwap' or 'book_vwap'")

        if self.side == "buy":
            slip = (avg - vwap) / max(vwap, 1e-12) * 1e4
        else:
            slip = (vwap - avg) / max(vwap, 1e-12) * 1e4

        return ExecutionResult(
            strategy=strategy,
            side=self.side,
            requested_qty=self.quantity,
            executed_qty=executed,
            avg_price=float(avg),
            benchmark_vwap=float(vwap),
            slippage_bps=float(slip),
            completion=float(executed / self.quantity),
            levels_consumed=levels_consumed,
        )
