"""Download the FI-2010 NoAuction z-score benchmark without fabricating data."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import zipfile

FILES = [
    "Train_Dst_NoAuction_ZScore_CF_7.txt",
    "Test_Dst_NoAuction_ZScore_CF_7.txt",
    "Test_Dst_NoAuction_ZScore_CF_8.txt",
    "Test_Dst_NoAuction_ZScore_CF_9.txt",
]
KAGGLE_DATASET = "ulfricirons/fi-2010"
ETSIN_URL = "https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real/fi2010")
    ap.add_argument("--source", choices=["kaggle", "manual"], default="kaggle")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    missing = [f for f in FILES if not (out / f).exists()]
    if not missing:
        print(f"FI-2010 files already present under {out}")
        return
    if args.source == "manual":
        print(f"Download the four NoAuction z-score files from the official Fairdata/ETSIN dataset: {ETSIN_URL}")
        print(f"Extract them into {out}/")
        return
    try:
        import kaggle
    except Exception as exc:
        raise SystemExit(
            "Kaggle API is required for automatic download. Install kaggle, "
            "configure ~/.kaggle/kaggle.json, or use --source manual."
        ) from exc
    cache = Path(".cache/fi2010_kaggle"); cache.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", str(cache)], check=True)
    zips = list(cache.glob("*.zip"))
    if not zips:
        raise SystemExit("Kaggle download produced no ZIP archive.")
    with zipfile.ZipFile(zips[-1]) as zf:
        names = {Path(n).name: n for n in zf.namelist()}
        for filename in missing:
            if filename not in names:
                raise SystemExit(f"Kaggle archive did not contain {filename}")
            with zf.open(names[filename]) as src, (out / filename).open("wb") as dst:
                dst.write(src.read())
    print(f"Downloaded {len(missing)} FI-2010 files to {out}")

if __name__ == "__main__":
    main()
