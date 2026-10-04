# Getting started

This document is the shortest path from a clean checkout to a working FlowOE run.

## 1. Requirements

Use Python 3.10 or newer. Python 3.11 is used in the repository CI workflow.

For the CPU-only demonstration you need:

- Python
- pip
- a C/C++ compiler is not required
- approximately 2 GB of free disk space for the Python environment, depending on the PyTorch wheel selected by pip

CUDA and TensorRT are optional. They are not needed for the smoke test.

## 2. Clone and install

From a terminal:

```bash
git clone https://github.com/SpryzenHell/flowoe_execution.git
cd flowoe_execution

python -m venv .venv
```

Activate the environment:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

Install the package and test dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
```

The package is installed in editable mode, so changes under `src/` are used immediately.

## 3. Run the test suite

```bash
python -m pytest -q
```

The current CI run executes the same command.

## 4. Run the smoke experiment

```bash
python scripts/run_experiment.py --smoke --epochs 12
```

The script creates a small deterministic synthetic dataset under `data/smoke/`, trains a CFM policy, replays the resulting schedules against a top-10 synthetic book, compares the policy with TWAP, and writes the model and report under `results/`.

The command does not require network access after the Python dependencies are installed.

## 5. Run the latency smoke check

```bash
python scripts/benchmark_latency.py --steps 16 --runs 300
```

On a machine without CUDA this is a CPU measurement. It must not be used as a GPU latency claim.

## 6. Run with Docker

A CPU-only smoke environment is also provided:

```bash
docker build -t flowoe-execution .
docker run --rm flowoe-execution
```

The container command runs the same 12-epoch smoke experiment used by CI.

## 7. Project outputs

The smoke run produces:

- `data/smoke/crypto_l2.csv`
- `data/smoke/crypto_trades.csv`
- `data/smoke/fi2010_smoke.txt`
- `results/cfm_policy_smoke.pt`
- `results/smoke_experiment.json`

These generated files are ignored by Git.

## 8. Real-data run

A real run requires all three inputs:

1. FI-2010 NoAuction ZScore data.
2. Reconstructed top-10 L2 snapshots for the selected instrument and period.
3. Trade prints from the same venue and period.

Start with:

```bash
python scripts/validate_real_data.py \
  --fi data/real/fi2010/Train_Dst_NoAuction_ZScore_CF_7.txt \
  --l2 data/real/crypto/BTCUSDT_l2.csv \
  --trades data/real/crypto/BTCUSDT_trades.csv
```

Then:

```bash
python scripts/run_experiment.py \
  --fi data/real/fi2010/Train_Dst_NoAuction_ZScore_CF_7.txt \
  --l2 data/real/crypto/BTCUSDT_l2.csv \
  --trades data/real/crypto/BTCUSDT_trades.csv \
  --epochs 120
```

The real-data path is intentionally fail-closed when trade prints are missing. The `--allow-book-vwap` option exists for a proxy analysis and is not the trade-VWAP evidence path.

See `docs/REAL_DATA_RUN.md` for the full data layout and reconstruction steps.
