# FlowOE Execution

**Conditional Flow Matching + Probability Flow ODE execution policy with L2-aware backtesting and an optional INT8 TensorRT/CUDA path.**

This branch turns the mechanically merged upstream code into a coherent project. The original `flowTorchcfm`, `flowTorchdyn`, and `flowTorch2trt_dynamic` trees are retained as upstream provenance; the project-specific implementation lives under `src/flowoe_execution/`.

## Resume-bullet mapping

### Continuous Normalizing Flows / Probability Flow ODEs

`CFMPolicy` implements the simulation-free conditional flow-matching objective and samples a learned vector field through an RK4 ODE. `ProbabilityFlowODEPolicy` separately implements a conditional variance-preserving score objective and reverse-time probability-flow ODE. The FI-2010 side is used as a genuine auxiliary five-horizon movement-classification task through a shared temporal encoder; the Crypto L2 side then trains the execution vector field on the same shared temporal representation.

### Slippage against VWAP

`ExecutionSimulator` replays generated child-order schedules against the configured **full top-10 L2 depth**, consuming liquidity level-by-level rather than treating the best quote as infinite. With trade prints, `benchmark=trade_vwap` computes the economic trade-VWAP benchmark. Without trades, `book_vwap` is an explicit depth-weighted proxy.

For a real-data evidence run, `scripts/validate_real_data.py` requires a trade stream overlapping the L2 evaluation window, and `run_experiment.py` refuses to silently downgrade a real run to the proxy.

### INT8 TensorRT / CUDA

`FixedStepCryptoSampler` exposes a fixed-step sampler suitable for TensorRT conversion. `build_trt_engine.py` requests INT8 calibration and `benchmark_trt.py` measures FP32 vs TensorRT p50/p95/p99, speedup, and output error. The accelerator extra installs the upstream `grimoire/torch2trt_dynamic` source directly because that project is distributed as a Git repository rather than a normal PyPI dependency. A CUDA extension provides a fused Euler update.

The repository deliberately **does not hard-code the resume's `<2 ms p99` as a result**. That number becomes a defensible claim only after a real CUDA/TensorRT run records it in `results/tensorrt_latency.json`.

## Data

### FI-2010

The recommended training file is:

`Train_Dst_NoAuction_ZScore_CF_7.txt`

The standard FI-2010 protocol uses the cumulative first-7-days CF_7 training prefix and held-out CF_7/8/9 test-day files. This project uses FI-2010 as auxiliary microstructure conditioning/augmentation; the normalized feature values are never treated as execution prices.

### Crypto L2

The production benchmark expects reconstructed top-10 L2 snapshots:

`timestamp, bid0..bid9, bid_size0..bid_size9, ask0..ask9, ask_size0..ask_size9`

Use `scripts/reconstruct_binance_t_depth.py` for row-based Binance T_DEPTH updates. The script normalizes millisecond/microsecond-style event timestamps, validates event IDs, and can fail closed on detected sequence gaps.

For an economic VWAP benchmark, provide matching trade prints. `scripts/download_binance_trades.py` can fetch daily Spot or USD-M Futures trade archives and normalize them to `timestamp,price,qty,trade_id`.

Large market-data files are deliberately not checked into Git.

## Quickstart

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/run_experiment.py --smoke --epochs 120
python scripts/benchmark_latency.py --steps 16 --runs 300
```

## Real-data evidence run

First validate the data:

```bash
python scripts/validate_real_data.py \
  --fi data/real/fi2010/Train_Dst_NoAuction_ZScore_CF_7.txt \
  --l2 data/real/crypto/BTCUSDT_l2.csv \
  --trades data/real/crypto/BTCUSDT_trades.csv
```

Then run the chronological experiment:

```bash
python scripts/run_experiment.py \
  --fi data/real/fi2010/Train_Dst_NoAuction_ZScore_CF_7.txt \
  --l2 data/real/crypto/BTCUSDT_l2.csv \
  --trades data/real/crypto/BTCUSDT_trades.csv \
  --epochs 120
```

The policy is fitted only on the earlier observations and evaluated on later observations. The experiment writes input SHA-256 hashes into the result JSON so a reported bps number can be traced to exact files.

## GPU / TensorRT

```bash
python -m pip install -e '.[accelerated]'
python scripts/build_trt_engine.py
python scripts/benchmark_trt.py --runs 500
```

Record GPU model, driver, CUDA, TensorRT, PyTorch, batch size, ODE steps, warmup count, p50/p95/p99, speedup, and output error with every performance run.

## Data acquisition

FI-2010:

```bash
python scripts/download_fi2010.py --source kaggle
# or follow the manual Fairdata/ETSIN path printed by:
python scripts/download_fi2010.py --source manual
```

Binance trades:

```bash
python scripts/download_binance_trades.py \
  --market um \
  --symbol BTCUSDT \
  --start 2025-01-01 \
  --end 2025-01-02
```

The L2 side should come from a matching Binance depth source for the same venue and time window. The reconstruction report exposes sequence gaps; do not call a run evidence-quality if the raw depth stream is incomplete.

## Local validation snapshot

The captured offline smoke run is synthetic regression evidence only. It is not FI-2010, not real crypto L2, and not a GPU benchmark. CUDA/TensorRT were unavailable in the development environment, so no `<2 ms p99` result is claimed here.

## Evidence checklist

Before promoting a numeric result to the resume, save the corresponding JSON artifact and verify:

| Gate | Required |
|---|---|
| Data provenance | Source URL, instrument, dates, SHA-256 |
| FI-2010 | NoAuction ZScore, 144 features + 5 labels |
| Crypto L2 | Full top-10 snapshots, sequence-integrity report |
| VWAP | Matching trade prints covering every evaluation window |
| Split | Chronological train/evaluation cut |
| Baseline | TWAP on the same windows and same requested quantity |
| Execution | Full configured L2 depth, explicit side and completion |
| Statistical report | Mean, median, p95, bootstrap 95% CI |
| GPU | GPU/software versions, p50/p95/p99, output error |
| Resume claim | Number copied only after the above gates pass |

## Tests

The test suite covers 45-D L2 feature construction, CFM loss/sampling, probability-flow ODE sampling, full-depth execution replay, FI-2010 matrix orientation, Binance timestamp units, sequence-gap detection, and the fused-step reference path.

## Upstream provenance

The project was assembled from:
- `atong01/conditional-flow-matching`
- `DiffEqML/torchdyn`
- `grimoire/torch2trt_dynamic`

See `README_FLOWOE.md` and `THIRD_PARTY_NOTICES.md` for project-layer and provenance details.
