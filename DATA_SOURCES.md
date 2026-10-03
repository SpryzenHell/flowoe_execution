# Data protocol

## FI-2010

The loader expects the commonly distributed normalized FI-2010 matrix with 144 feature columns followed by five label columns. FI-2010 is used as auxiliary microstructure conditioning/augmentation; its normalized values are not treated as raw execution prices.

## Crypto L2

The production execution benchmark expects reconstructed top-10 L2 snapshots and a corresponding trade stream. The L2 schema is documented in `README.md`. `scripts/reconstruct_binance_t_depth.py` converts row-based depth updates into canonical snapshots.

For an economic VWAP benchmark, provide trade prints. If no trade file is supplied, the runner uses a clearly labeled depth-weighted book-VWAP proxy.

## Experimental protocol

1. Split chronologically.
2. Fit trajectory policy only on the earlier period.
3. Generate schedules from the policy without future observations.
4. Replay schedules against later L2 observations.
5. Compare against TWAP using the same execution window.
6. Report mean slippage, bootstrap confidence interval, completion, and the exact data/hardware configuration.

Large market-data archives are not checked into the repository.
