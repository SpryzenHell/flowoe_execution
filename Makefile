PYTHON ?= python
export PYTHONPATH := src

.PHONY: test smoke analysis latency install install-test install-analysis
install:
	$(PYTHON) -m pip install -e .
install-test:
	$(PYTHON) -m pip install -e '.[test]'
test:
	$(PYTHON) -m pytest -q
smoke:
	$(PYTHON) scripts/run_experiment.py --smoke --epochs 120
latency:
	$(PYTHON) scripts/benchmark_latency.py --steps 16 --runs 300

install-analysis:
	$(PYTHON) -m pip install -e '.[analysis,test]'

analysis:
	$(PYTHON) scripts/run_analysis_suite.py --n 1200 --epochs 3
