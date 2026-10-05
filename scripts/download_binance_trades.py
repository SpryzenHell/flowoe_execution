"""Download Binance historical trade prints and normalize them for VWAP evaluation."""
from __future__ import annotations
import argparse
import csv
from datetime import date, timedelta
from pathlib import Path
from urllib.request import urlopen
import zipfile

def dates(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)

def day_url(market: str, symbol: str, day: date) -> str:
    stamp = day.isoformat()
    if market == "um":
        base = "https://data.binance.vision/data/futures/um/daily/trades"
    else:
        base = "https://data.binance.vision/data/spot/daily/trades"
    return f"{base}/{symbol}/{symbol}-trades-{stamp}.zip"

def convert_zip(zpath: Path, out_writer: csv.writer):
    with zipfile.ZipFile(zpath) as zf:
        member = zf.namelist()[0]
        with zf.open(member) as src:
            reader = csv.reader((line.decode("utf-8") for line in src))
            for row in reader:
                if not row or not row[0] or row[0].lower() == "id":
                    continue
                if len(row) < 5:
                    continue
                trade_id, price, qty, quote_qty, timestamp = row[:5]
                out_writer.writerow([timestamp, price, qty, trade_id])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--market", choices=["um", "spot"], default="um")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--out", default="data/real/crypto/BTCUSDT_trades.csv")
    ap.add_argument("--cache-dir", default=".cache/binance_trades")
    args = ap.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end must be on or after --start")
    cache = Path(args.cache_dir); cache.mkdir(parents=True, exist_ok=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as dst:
        writer = csv.writer(dst); writer.writerow(["timestamp", "price", "qty", "trade_id"])
        for day in dates(start, end):
            archive = cache / f"{args.symbol}-{args.market}-{day.isoformat()}.zip"
            if not archive.exists():
                url = day_url(args.market, args.symbol, day)
                print(f"fetching {url}")
                with urlopen(url, timeout=60) as response:
                    archive.write_bytes(response.read())
            convert_zip(archive, writer)
    print(out)

if __name__ == "__main__":
    main()
