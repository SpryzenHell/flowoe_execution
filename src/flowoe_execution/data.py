from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class FI2010Data:
    features: np.ndarray
    labels: np.ndarray

@dataclass(frozen=True)
class L2Data:
    snapshots: pd.DataFrame
    trades: pd.DataFrame | None = None

def load_fi2010(path: str | Path, max_rows: int | None = None) -> FI2010Data:
    kwargs = {"dtype": np.float32}
    if max_rows is not None:
        kwargs["max_rows"] = int(max_rows)
    arr = np.loadtxt(path, **kwargs)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D FI-2010 matrix, got {arr.shape}")
    if arr.shape[1] < 149 and arr.shape[0] >= 149:
        arr = arr.T
    if arr.shape[1] < 149:
        raise ValueError(f"Expected at least 149 columns (144 features + 5 labels), got {arr.shape}")
    return FI2010Data(
        features=arr[:, :144].astype(np.float32, copy=False),
        labels=arr[:, 144:149].astype(np.int16, copy=False),
    )

def _book(df: pd.DataFrame, levels: int = 10) -> pd.DataFrame:
    df = df.copy()
    if "timestamp" not in df.columns:
        if {"tsec", "tnsec"}.issubset(df.columns):
            df["timestamp"] = df.tsec.astype("int64") + df.tnsec.astype("int64") / 1e9
        elif "ts" in df.columns:
            df["timestamp"] = df.ts
        else:
            raise ValueError("L2 file must contain timestamp or tsec/tnsec")

    missing = [
        c for i in range(levels)
        for c in (f"bid{i}", f"bid_size{i}", f"ask{i}", f"ask_size{i}")
        if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing L2 columns: {missing[:8]}{'...' if len(missing) > 8 else ''}")

    cols = ["timestamp"] + [
        c for i in range(levels)
        for c in (f"bid{i}", f"bid_size{i}", f"ask{i}", f"ask_size{i}")
    ]
    out = df[cols].copy()
    if np.issubdtype(out.timestamp.dtype, np.number):
        out["timestamp"] = pd.to_datetime(out.timestamp, unit="s", errors="coerce")
    else:
        out["timestamp"] = pd.to_datetime(out.timestamp, utc=True, errors="coerce", format="mixed")
    return out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

def load_l2_csv(path: str | Path, levels: int = 10, max_rows: int | None = None) -> L2Data:
    return L2Data(_book(pd.read_csv(path, nrows=max_rows), levels=levels))

def load_l2_jsonl(path: str | Path, levels: int = 10) -> L2Data:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        row = {"timestamp": obj.get("timestamp", obj.get("ts"))}
        for i in range(levels):
            row[f"bid{i}"], row[f"bid_size{i}"] = obj["bids"][i]
            row[f"ask{i}"], row[f"ask_size{i}"] = obj["asks"][i]
        rows.append(row)
    return L2Data(_book(pd.DataFrame(rows), levels=levels))

def load_trades_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        if {"tsec", "tnsec"}.issubset(df.columns):
            df["timestamp"] = df.tsec.astype("int64") + df.tnsec.astype("int64") / 1e9
        elif "ts" in df.columns:
            df["timestamp"] = df.ts
        elif "time" in df.columns:
            df["timestamp"] = df.time
        else:
            raise ValueError("trade file must contain timestamp, ts, time, or tsec/tnsec")

    qty = "qty" if "qty" in df.columns else "quantity" if "quantity" in df.columns else None
    if "price" not in df.columns or qty is None:
        raise ValueError("trade file must contain price and qty/quantity")

    if np.issubdtype(df.timestamp.dtype, np.number):
        unit = "ms" if float(df.timestamp.iloc[0]) > 1e11 else "s"
        df["timestamp"] = pd.to_datetime(df.timestamp, unit=unit, errors="coerce")
    else:
        df["timestamp"] = pd.to_datetime(df.timestamp, utc=True, errors="coerce", format="mixed")
    return (
        df.dropna(subset=["timestamp", "price", qty])
        .rename(columns={qty: "qty"})[["timestamp", "price", "qty"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
