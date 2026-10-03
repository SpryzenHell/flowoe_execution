# FlowOE Execution — Project Layer

This branch adds a clean project-specific execution layer on top of the three upstream codebases already present in the repository: Conditional Flow Matching, TorchDyn, and torch2trt_dynamic.

## What is implemented

### 1. Continuous Flow Matching
`src/flowoe_execution/model.py` contains `CFMPolicy`. It trains a conditional vector field using the simulation-free CFM objective and generates execution trajectories with a fixed RK4 ODE integrator.

### 2. Probability-flow ODE
`ProbabilityFlowODEPolicy` implements a conditional VP score objective and reverse-time probability-flow ODE sampler. It is kept as a separate model so CFM and probability-flow experiments can be compared rather than conflated.

### 3. L2 + FI-2010 data layer
- `load_fi2010` validates the expected 144 feature columns plus five label columns.
- `load_l2_csv` consumes canonical top-10 L2 snapshots.
- `reconstruct_binance_t_depth.py` converts row-based depth updates into the canonical snapshot format.
- FI-2010 is an auxiliary microstructure representation; raw execution prices come from crypto L2/trades, not normalized FI-2010 features.

### 4. Execution simulation
`ExecutionSimulator` replays generated child-order schedules against top-of-book liquidity and reports slippage in basis points. With trade prints it uses true trade VWAP; without trades it reports a clearly labeled book-VWAP proxy.

### 5. TensorRT / CUDA path
`FixedStepCryptoSampler` removes the dynamic Python ODE loop and exposes a fixed-step forward pass suitable for TensorRT conversion. `build_trt_engine.py` requests INT8 calibration and refuses to claim an engine on a non-CUDA machine. `benchmark_trt.py` reports p50/p95/p99, speedup, and maximum output error.

A small CUDA extension skeleton is also included under `src/flowoe_execution/cuda/` for a fused Euler update. It is deliberately benchmarked separately from the TensorRT path; no fabricated `<2 ms p99` number is committed.

## Local validation

The offline smoke path has four pytest tests and a full train → sample → execute → VWAP pipeline. The captured 120-epoch smoke run used 445 crypto training windows, 450 FI-style auxiliary windows, and 100 held-out execution episodes. It measured 1.2766 bps for TWAP and 1.2751 bps for FlowOE, an observed difference of 0.0015 bps. These are synthetic regression results, **not** FI-2010/real-crypto results.

The CPU PyTorch latency capture for 16 ODE steps was 3.637 ms p99. CUDA/TensorRT were unavailable in that environment, so the resume target `<2 ms p99` is not claimed by the repository until it is measured on a CUDA/TensorRT host.

## Commands

```bash
python -m pip install -e .
python -m pytest -q
python scripts/run_experiment.py --smoke --epochs 120
python scripts/benchmark_latency.py --steps 16 --runs 300
```

For real data:

```bash
python scripts/run_experiment.py \
  --fi data/real/fi2010/Train_Dst_NoAuction_ZScore_CF_1.txt \
  --l2 data/real/crypto/BTCUSDT_l2.csv \
  --trades data/real/crypto/BTCUSDT_trades.csv
```

For the GPU path:

```bash
python -m pip install -e '.[accelerated]'
python scripts/build_trt_engine.py
python scripts/benchmark_trt.py --runs 500
```

The benchmark output must be copied to `results/` only after an actual hardware run. Resume claims should be derived from those measured files, not from target values.
