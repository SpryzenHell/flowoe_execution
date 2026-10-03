# FlowOE Execution

**Conditional Flow Matching + Probability Flow ODE execution policy with L2-aware backtesting and an optional INT8 TensorRT/CUDA path.**

This branch turns the mechanically merged upstream code into a coherent project. The original `flowTorchcfm`, `flowTorchdyn`, and `flowTorch2trt_dynamic` trees are retained as upstream provenance; the project-specific implementation lives under `src/flowoe_execution/`.

## Resume-bullet mapping

### Continuous Normalizing Flows / Probability Flow ODEs

`CFMPolicy` implements the simulation-free conditional flow-matching objective and samples a learned vector field through an RK4 ODE. `ProbabilityFlowODEPolicy` separately implements a conditional variance-preserving score objective and reverse-time probability-flow ODE.

### Slippage against VWAP

`ExecutionSimulator` replays generated child-order schedules against top-of-book liquidity. With trade prints, `benchmark=trade_vwap` computes a true volume-weighted trade benchmark. Without trades, it reports a clearly labeled depth-weighted book-VWAP proxy. FI-2010 is auxiliary model input, not an economic price source.

### INT8 TensorRT / CUDA

`FixedStepCryptoSampler` exposes a fixed-step sampler suitable for TensorRT conversion. `build_trt_engine.py` requests INT8 calibration and `benchmark_trt.py` measures FP32 vs TensorRT p50/p95/p99, speedup, and output error. A CUDA extension skeleton under `src/flowoe_execution/cuda/` provides a fused Euler update.

The repository deliberately **does not hard-code the resume's `<2 ms p99` as a result**. That number becomes a defensible claim only after a real CUDA/TensorRT run records it in `results/tensorrt_latency.json`.

## Data

`load_fi2010()` validates the commonly distributed normalized FI-2010 representation as 144 feature columns followed by five label columns. `load_l2_csv()` consumes canonical top-10 L2 snapshots:

`timestamp, bid0..bid9, bid_size0..bid_size9, ask0..ask9, ask_size0..ask_size9`

Use `scripts/reconstruct_binance_t_depth.py` for row-based depth updates. Supply a corresponding trade CSV with `timestamp,price,qty` for a true trade-VWAP benchmark. Large market-data files are deliberately not checked into Git.

## Quickstart

```bash
python -m pip install -e .
python -m pytest -q
python scripts/run_experiment.py --smoke --epochs 120
python scripts/benchmark_latency.py --steps 16 --runs 300
```

## Real-data run

```bash
python scripts/run_experiment.py \
  --fi data/real/fi2010/Train_Dst_NoAuction_ZScore_CF_1.txt \
  --l2 data/real/crypto/BTCUSDT_l2.csv \
  --trades data/real/crypto/BTCUSDT_trades.csv
```

The split is chronological: the policy is fitted only on earlier observations and evaluated later.

## GPU / TensorRT

```bash
python -m pip install -e '.[accelerated]'
python scripts/build_trt_engine.py
python scripts/benchmark_trt.py --runs 500
```

Record GPU model, driver, CUDA, TensorRT, PyTorch, batch size, ODE steps, warmup count, p50/p95/p99, and output error with every performance run.

## Local validation snapshot

The captured offline smoke run used 445 crypto training windows, 450 FI auxiliary windows, and 100 held-out execution episodes. At 120 epochs it measured **1.2766 bps** mean slippage for TWAP against the synthetic trade-VWAP benchmark and **1.2751 bps** for FlowOE, an observed difference of **0.0015 bps**. These are synthetic regression results, **not** FI-2010 or real-crypto results.

The captured CPU PyTorch benchmark with 16 ODE steps had **3.637 ms p99**. CUDA/TensorRT were unavailable in that environment, so no GPU latency claim is made.

## Tests

The test suite covers 45-D L2 features, CFM loss/sampling, probability-flow ODE sampling, and execution replay.

## Upstream provenance

The project was assembled from `atong01/conditional-flow-matching`, `DiffEqML/torchdyn`, and `grimoire/torch2trt_dynamic`. See `README_FLOWOE.md` and `THIRD_PARTY_NOTICES.md` for project-layer and provenance details.
