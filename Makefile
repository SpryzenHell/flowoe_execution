PYTHON ?= python
export PYTHONPATH := src

.PHONY: test smoke latency install
install:
	$(PYTHON) -m pip install -e .
test:
	$(PYTHON) -m pytest -q
smoke:
	$(PYTHON) scripts/run_experiment.py --smoke --epochs 120
latency:
	$(PYTHON) scripts/benchmark_latency.py --steps 16 --runs 300
