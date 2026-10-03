# FlowOE Execution — Project Layer

This branch adds the project-specific execution research layer on top of the three upstream codebases already present in the repository:

- Conditional Flow Matching: `atong01/conditional-flow-matching`
- Neural ODE numerics: `DiffEqML/torchdyn`
- PyTorch → TensorRT conversion: `grimoire/torch2trt_dynamic`

The upstream trees are retained for provenance; the clean FlowOE implementation lives under `src/flowoe_execution/`.

## 1. Model

### Conditional Flow Matching

`CFMPolicy` learns a conditional vector field with the simulation-free CFM objective. Random base trajectories are transported toward target execution schedules, and an RK4 ODE integrates the learned field from (t=0) to (t=1).

The context encoder has source-specific input projections for:

- FI-2010: 144 normalized LOB-derived features.
- Crypto L2: 45 engineered features from top-10 depth, spread, microprice, imbalance, volatility and OFI.

Those projections feed a shared temporal convolutional block. FI-2010 supplies a five-horizon auxiliary movement-classification loss, while Crypto L2 supplies the execution-policy CFM loss. This makes the data augmentation path explicit rather than converting FI labels directly into artificial order schedules.

### Probability-flow ODE

`ProbabilityFlowODEPolicy` trains a conditional variance-preserving score network and integrates the corresponding reverse-time probability-flow ODE. It is kept separate from the CFM policy so the two trajectory-generation approaches can be evaluated independently.

## 2. Market-data layer

### FI-2010

`load_fi2010` validates the NoAuction ZScore representation as 144 feature columns plus five horizon labels and also handles mirrors that store the matrix transposed.

The recommended auxiliary training file is:

`Train_Dst_NoAuction_ZScore_CF_7.txt`

The standard FI-2010 anchored protocol uses the cumulative first seven training days and the corresponding later held-out days.

### Crypto L2

The execution benchmark consumes canonical top-10 snapshots:

`timestamp, bid0..bid9, bid_size0..bid_size9, ask0..ask9, ask_size0..ask_size9`

`scripts/reconstruct_binance_t_depth.py` reconstructs these snapshots from row-based Binance T_DEPTH data and records event-sequence gaps. It can fail closed with `--fail-on-gap`.

Trade prints are normalized to:

`timestamp, price, qty, trade_id`

by `scripts/download_binance_trades.py`.

## 3. Execution simulation

`ExecutionSimulator` replays a child-order schedule one time bucket at a time and consumes the full configured L2 depth level-by-level.

For a buy:

`ask0 → ask1 → ... → ask9`

are consumed until the child order is filled or the configured depth is exhausted. Sell execution mirrors this over the bid side.

The economic benchmark is trade VWAP over the same execution window. A book-VWAP proxy remains available for diagnostics, but a real evidence run requires matching trade prints by default.

The result records:

- requested and executed quantity;
- completion ratio;
- average execution price;
- VWAP benchmark;
- slippage in basis points;
- number of L2 levels consumed.

## 4. Experimental protocol

The real-data runner uses a chronological split:

`earlier observations → training → later observations → execution evaluation`

No future price or future trade stream is fed into the policy context during evaluation.

The training target is generated from future training-period information only; evaluation schedules are sampled from the learned policy using the current L2 context.

Each run records:

- dataset paths;
- SHA-256 hashes;
- number of training windows;
- FI auxiliary windows;
- evaluation episode count;
- mean / median / p95 slippage;
- bootstrap 95% confidence interval;
- benchmark type;
- training time.

`scripts/validate_real_data.py` is an explicit pre-flight gate and refuses to validate a real VWAP experiment unless trades overlap the L2 evaluation window.

## 5. TensorRT / CUDA deployment

`FixedStepCryptoSampler` uses a fixed number of Euler steps over exactly the unit interval, avoiding a dynamic Python-controlled integration loop.

The TensorRT path:

`PyTorch FP32 → fixed-step graph → TensorRT INT8 calibration → engine → latency benchmark`

uses the upstream `torch2trt_dynamic` implementation. The project installs that dependency directly from its Git repository in the `accelerated` extra because it is not maintained as a normal PyPI release.

The CUDA extension provides a fused Euler update kernel. The fixed-step sampler can call that kernel through the optional `fused_step` callback, and `scripts/benchmark_cuda_fused.py` compares it with the native PyTorch update.

The repository intentionally does not claim `<2 ms p99` until an actual target GPU benchmark records the result in `results/tensorrt_latency.json`.

## 6. Commands

### Offline regression

```bash
python -m pip install -e .
python -m pytest -q
python scripts/run_experiment.py --smoke --epochs 120
python scripts/benchmark_latency.py --steps 16 --runs 300
```

### FI-2010 acquisition

```bash
python scripts/download_fi2010.py --source kaggle
```

Or use the manual Fairdata/ETSIN route printed by:

```bash
python scripts/download_fi2010.py --source manual
```

### Binance trade acquisition

```bash
python scripts/download_binance_trades.py \
  --market um \
  --symbol BTCUSDT \
  --start 2025-01-01 \
  --end 2025-01-02
```

### Real-data evidence run

```bash
python scripts/validate_real_data.py \
  --fi data/real/fi2010/Train_Dst_NoAuction_ZScore_CF_7.txt \
  --l2 data/real/crypto/BTCUSDT_l2.csv \
  --trades data/real/crypto/BTCUSDT_trades.csv

python scripts/run_experiment.py \
  --fi data/real/fi2010/Train_Dst_NoAuction_ZScore_CF_7.txt \
  --l2 data/real/crypto/BTCUSDT_l2.csv \
  --trades data/real/crypto/BTCUSDT_trades.csv
```

### CUDA / TensorRT

```bash
python -m pip install -e '.[accelerated]'
python scripts/build_cuda_extension.py
python scripts/benchmark_cuda_fused.py --runs 1000
python scripts/build_trt_engine.py
python scripts/benchmark_trt.py --runs 500
```

## 7. Results policy

Synthetic smoke results are regression evidence only.

Real slippage claims require the saved real-data JSON, source provenance, overlapping trade VWAP, chronological split and clean depth-quality report.

GPU latency claims require the saved TensorRT/CUDA benchmark plus GPU and software metadata.

Resume numbers are derived from measured artifacts; target numbers are never written as if they were observations.
