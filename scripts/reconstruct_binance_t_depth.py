from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

LEVELS = 10


def _timestamp_seconds(raw):
    value = float(raw)
    magnitude = abs(value)
    if magnitude >= 1e17:
        return value / 1e9
    if magnitude >= 1e14:
        return value / 1e6
    if magnitude >= 1e11:
        return value / 1e3
    return value


KEY_COLUMNS = {"symbol", "timestamp", "first_update_id", "last_update_id", "side", "update_type", "price", "qty"}

def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not KEY_COLUMNS.issubset(reader.fieldnames):
            missing = sorted(KEY_COLUMNS.difference(reader.fieldnames or []))
            raise ValueError(f"Binance T_DEPTH file is missing columns: {missing}")
        yield from reader

def _event_key(row):
    return (
        row["timestamp"],
        row["first_update_id"],
        row["last_update_id"],
        row.get("pu", ""),
    )

def _check_event_sequence(row, previous_last):
    if previous_last is None:
        return False
    first = int(row["first_update_id"])
    last = int(row["last_update_id"])
    pu_raw = row.get("pu", "")
    pu = int(pu_raw) if pu_raw not in ("", "-1", None) else None
    return first > previous_last + 1 or last <= previous_last or (pu is not None and pu != previous_last)

def apply(book, row):
    side = "ask" if row["side"].lower() in {"a", "ask"} else "bid"
    price = float(row["price"])
    qty = float(row["qty"])
    typ = row.get("update_type", "set").lower()
    if price <= 0 or qty < 0:
        raise ValueError(f"Invalid book update price/qty: {price}/{qty}")
    if typ == "delta":
        book[side][price] = book[side].get(price, 0.0) + qty
    else:
        book[side][price] = qty
    if book[side].get(price, 0.0) <= 0:
        book[side].pop(price, None)

def top(side, book):
    vals = sorted(
        ((p, q) for p, q in book[side].items() if q > 0),
        reverse=(side == "bid"),
    )[:LEVELS]
    return sorted(vals) if side == "ask" else vals

def emit(writer, ts, book):
    bids, asks = top("bid", book), top("ask", book)
    if len(bids) < LEVELS or len(asks) < LEVELS:
        return False
    out = {"timestamp": ts}
    for i in range(LEVELS):
        out[f"bid{i}"], out[f"bid_size{i}"] = bids[i]
        out[f"ask{i}"], out[f"ask_size{i}"] = asks[i]
    writer.writerow(out)
    return True

def main():
    ap = argparse.ArgumentParser(description="Reconstruct canonical top-10 snapshots from Binance T_DEPTH rows.")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--updates", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval-ms", type=int, default=100)
    ap.add_argument("--quality-report")
    ap.add_argument("--fail-on-gap", action="store_true")
    args = ap.parse_args()

    if args.interval_ms <= 0:
        raise SystemExit("--interval-ms must be positive")

    book = defaultdict(dict)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["timestamp"] + [
        c for i in range(LEVELS)
        for c in (f"bid{i}", f"bid_size{i}", f"ask{i}", f"ask_size{i}")
    ]

    snapshots_applied = 0
    update_rows = 0
    emitted = 0
    skipped_incomplete = 0
    gaps = 0
    events = 0
    previous_event = None
    previous_last = None

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for row in rows(args.snapshot):
            apply(book, row)
            snapshots_applied += 1

        next_emit = None
        interval_s = args.interval_ms / 1000.0
        for row in rows(args.updates):
            update_rows += 1
            key = _event_key(row)
            if key != previous_event:
                events += 1
                if _check_event_sequence(row, previous_last):
                    gaps += 1
                    if args.fail_on_gap:
                        raise SystemExit(
                            f"Sequence gap detected before event {events}: "
                            f"previous_last={previous_last}, first={row['first_update_id']}, "
                            f"pu={row.get('pu')}"
                        )
                previous_last = int(row["last_update_id"])
                previous_event = key

            raw = row.get("timestamp") or row.get("time") or row.get("ts")
            ts_s = _timestamp_seconds(raw)
            apply(book, row)
            if next_emit is None:
                next_emit = ts_s
            if ts_s >= next_emit:
                if emit(writer, ts_s, book):
                    emitted += 1
                else:
                    skipped_incomplete += 1
                next_emit = ts_s + interval_s

    report = {
        "snapshot_rows_applied": snapshots_applied,
        "update_rows_applied": update_rows,
        "update_events": events,
        "sequence_gaps": gaps,
        "snapshots_emitted": emitted,
        "incomplete_intervals_skipped": skipped_incomplete,
        "interval_ms": args.interval_ms,
        "levels": LEVELS,
        "fail_on_gap": args.fail_on_gap,
    }
    if args.quality_report:
        report_path = Path(args.quality_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
