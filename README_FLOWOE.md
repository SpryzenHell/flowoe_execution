# FlowOE Execution — Project Notes

FlowOE is a PyTorch order-execution project.

The main project guide is [README.md](README.md).

## Main flow

```text
L2 data
  ↓
45-value feature vector
  ↓
CFM / probability-flow ODE
  ↓
execution schedule
  ↓
top-10 L2 replay
  ↓
VWAP, TWAP, slippage, completion
```

## Main components

- CFM schedule generation.
- Probability-flow ODE comparison.
- FI-2010 auxiliary training.
- Full-depth L2 execution replay.
- Trade-VWAP benchmark.
- Synthetic analysis and sensitivity tests.
- Real-data validation.
- CUDA and TensorRT benchmark tools.

## Run

```bash
python -m pytest -q
python scripts/run_experiment.py --smoke --epochs 12
make analysis
```

## Results

Synthetic tests are used for regression and model-development checks.

The repository does not store the 2.4 bps or <2 ms resume numbers as measured results. Those values need real data and real GPU benchmark runs.

See the main README for installation, data preparation, TensorRT setup and the result checklist.
