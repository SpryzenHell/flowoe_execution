# Real-data run

## Data layout

Place the data under the following directories:

```text
data/
└── real/
    ├── fi2010/
    │   ├── Train_Dst_NoAuction_ZScore_CF_7.txt
    │   ├── Test_Dst_NoAuction_ZScore_CF_7.txt
    │   ├── Test_Dst_NoAuction_ZScore_CF_8.txt
    │   └── Test_Dst_NoAuction_ZScore_CF_9.txt
    └── crypto/
        ├── BTCUSDT_l2.csv
        └── BTCUSDT_trades.csv
```

The large files are deliberately not committed to the repository.

## FI-2010

The loader expects a matrix with 144 feature columns followed by five prediction-label columns. The repository's recommended file is `Train_Dst_NoAuction_ZScore_CF_7.txt`.

The download helper supports the publicly available Kaggle mirror:

```bash
python scripts/download_fi2010.py --source kaggle
```

For environments where Kaggle credentials are not configured:

```bash
python scripts/download_fi2010.py --source manual
```

The manual command prints the official Fairdata/ETSIN dataset URL.

## Binance L2

The repository reconstructs a canonical top-10 book from Binance T_DEPTH-style rows.

The current Binance/Nautilus test-data convention uses:

```text
BTCUSDT_T_DEPTH_2022-11-01_depth_snap.csv
BTCUSDT_T_DEPTH_2022-11-01_depth_update.csv
```

For a parser/reconstruction test, a small prefix of the update stream is enough. For an evidence-quality run, use a complete, chronologically continuous window and keep the sequence-quality report.

Run the reconstruction:

```bash
python scripts/reconstruct_binance_t_depth.py \
  --snapshot path/to/BTCUSDT_T_DEPTH_2022-11-01_depth_snap.csv \
  --updates path/to/BTCUSDT_T_DEPTH_2022-11-01_depth_update.csv \
  --out data/real/crypto/BTCUSDT_l2.csv \
  --interval-ms 100 \
  --quality-report results/depth_quality.json \
  --fail-on-gap
```

The reconstruction:

- applies set/delta updates;
- keeps the full working book while emitting the best ten levels;
- normalizes timestamp units;
- validates update sequence IDs;
- refuses to continue in `--fail-on-gap` mode when a discontinuity is detected.

Public Binance depth files are large. A full BTCUSDT USD-M update file for 2022-11-01 is documented as roughly 12 GB, so start with a bounded test prefix when checking the pipeline.

## Binance trades

Normalize daily Binance trade archives with:

```bash
python scripts/download_binance_trades.py \
  --market um \
  --symbol BTCUSDT \
  --start 2025-01-01 \
  --end 2025-01-02 \
  --out data/real/crypto/BTCUSDT_trades.csv
```

The output schema is:

```text
timestamp,price,qty,trade_id
```

For the evidence run, the trade stream must overlap the complete L2 evaluation interval.

## Validate before training

```bash
python scripts/validate_real_data.py \
  --fi data/real/fi2010/Train_Dst_NoAuction_ZScore_CF_7.txt \
  --l2 data/real/crypto/BTCUSDT_l2.csv \
  --trades data/real/crypto/BTCUSDT_trades.csv
```

The validator checks:

- FI dimensions and finite values;
- FI label encoding;
- L2 row count and top-of-book validity;
- non-negative displayed sizes;
- positive trade prices and quantities;
- overlap between trade prints and the L2 window.

The validator writes `results/data_quality.json`.

## Run the experiment

```bash
python scripts/run_experiment.py \
  --fi data/real/fi2010/Train_Dst_NoAuction_ZScore_CF_7.txt \
  --l2 data/real/crypto/BTCUSDT_l2.csv \
  --trades data/real/crypto/BTCUSDT_trades.csv \
  --epochs 120
```

The experiment uses a chronological split. Schedule generation uses the historical context available at the start of each evaluation window; future prices are not supplied to the model at inference time.

The future mid-price path is used only as a training target generator for the synthetic trajectory target. It is not passed to the model during evaluation.

Each execution window is replayed against the configured ten-level book. Trade VWAP is computed from the matching trade stream over the same time interval.

The result is written to `results/real_experiment.json` and includes SHA-256 hashes of the input files.

## Interpreting the result

A result is suitable for reporting only when:

- the data sources and dates are recorded;
- the depth sequence report has no ignored gaps;
- the split is chronological;
- the benchmark uses matching trade prints;
- TWAP and FlowOE use the same windows and quantity;
- completion is reported;
- mean, median, p95 and bootstrap confidence intervals are saved.

The repository does not convert a synthetic smoke result into a real-market claim.
