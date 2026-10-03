# Data protocol

## FI-2010

The repository expects the NoAuction ZScore representation with 144 feature columns followed by five prediction labels. The recommended cumulative training file is `Train_Dst_NoAuction_ZScore_CF_7.txt`; the standard benchmark protocol uses the corresponding CF_7/8/9 held-out days for evaluation. FI-2010 is used here as auxiliary microstructure conditioning/augmentation, not as an execution-price source.

Source:
- Fairdata/ETSIN dataset metadata: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
- Kaggle acquisition mirror: https://www.kaggle.com/datasets/ulfricirons/fi-2010

## Crypto L2

The project consumes canonical top-10 L2 snapshots. For Binance T_DEPTH-style row data, `scripts/reconstruct_binance_t_depth.py` reconstructs the book and records event-sequence gaps instead of silently treating a discontinuity as clean data.

The depth stream and trade stream must cover the same instrument, venue, and evaluation window for a true VWAP comparison.

Useful public-data references:
- Binance public market-data repository: https://github.com/binance/binance-public-data
- Binance data site: https://data.binance.vision/
- Binance USD-M daily trades base: https://data.binance.vision/data/futures/um/daily/trades/
- Binance Spot daily trades base: https://data.binance.vision/data/spot/daily/trades/

## Experimental protocol

1. Build training windows only from observations at or before the training cut.
2. Train the trajectory policy on the training period.
3. Generate execution schedules for later windows without reading future prices or future trade prints.
4. Replay each schedule against the full configured L2 depth.
5. Use the same execution window to compute trade VWAP.
6. Compare FlowOE against TWAP and report mean, median, p95, completion, and bootstrap confidence intervals.
7. Record file paths, SHA-256 hashes, dates, instrument, side, order quantity, horizon, ODE steps, and hardware/software versions.

## Evidence gate

A result is not promoted to a resume claim until:
- raw data provenance is recorded;
- the real-data validator passes;
- the run uses a chronological split;
- the trade-VWAP window overlaps the L2 evaluation window;
- no depth sequence gap is ignored;
- the reported number is reproduced from the saved result JSON.

Large market-data archives are intentionally not checked into Git.
