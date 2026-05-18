.PHONY: help venv env test demo lint clean

PYTHON ?= python3
VENV   ?= .venv

help:
	@echo "make venv   - create local virtualenv (.venv) with core + science deps"
	@echo "make test   - run unit tests + mock end-to-end pipeline"
	@echo "make demo   - run the FDH/NADP example with mock backends"
	@echo "make clean  - remove caches and the demo run directory"

venv:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip wheel
	$(VENV)/bin/pip install -e ".[dev]"
	@echo "Optional (real chemistry/MD on this machine): $(VENV)/bin/pip install -e '.[science,md]'"

# Alias used in the plan/runbook.
env: venv

test:
	$(VENV)/bin/pytest

demo:
	$(VENV)/bin/evoliez run -c configs/example_fdh_nadp.yaml --backend mock

lint:
	$(VENV)/bin/python -m compileall -q src

clean:
	rm -rf .pytest_cache .coverage htmlcov **/__pycache__ src/**/__pycache__
	rm -rf runs/demo_fdh_nadp
