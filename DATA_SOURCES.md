# Data protocol

## FI-2010

The project expects the NoAuction ZScore representation with:

- 144 feature columns;
- 5 prediction-label columns.

The main training file is:

```text
Train_Dst_NoAuction_ZScore_CF_7.txt
```

The standard held-out files are the matching CF_7, CF_8 and CF_9 test files.

Official dataset metadata:

https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649

A Kaggle download helper is also included.

## Crypto L2

The project uses top-10 L2 snapshots.

For Binance T_DEPTH-style rows, the reconstruction script:

- applies set and delta updates;
- keeps the working book;
- emits the best ten levels;
- checks event sequence IDs;
- records sequence gaps;
- can stop on a gap with `--fail-on-gap`.

The L2 stream and trade stream must cover the same instrument, venue and evaluation window for a trade-VWAP comparison.

Public market-data pages:

- https://data.binance.vision/
- https://github.com/binance/binance-public-data

## Experimental procedure

1. Build training windows from past observations.
2. Train the model on the earlier time period.
3. Generate schedules on later windows.
4. Replay schedules against the full configured L2 depth.
5. Compute trade VWAP over the same execution window.
6. Compare FlowOE with TWAP.
7. Save mean, median, p95, completion and bootstrap confidence intervals.
8. Save file hashes, dates, instrument, side, quantity, horizon, solver steps and hardware/software details.

## Result checklist

A real-data result should have:

- a clear data record;
- a passing data-quality check;
- a time-based train/test split;
- matching trade VWAP;
- no ignored depth sequence gaps;
- a saved JSON result that reproduces the reported value.

Large market-data archives are not checked into Git.
