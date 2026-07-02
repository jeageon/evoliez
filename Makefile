.PHONY: help venv env test test-v4-manifest test-v4 demo lint clean

PYTHON ?= python3
VENV   ?= .venv

help:
	@echo "make venv   - create local virtualenv (.venv) with core + science deps"
	@echo "make test   - run unit tests + mock end-to-end pipeline"
	@echo "make test-v4 - run V4 MVP tests"
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

test-v4-manifest:
	$(VENV)/bin/pytest tests/experimental/test_seed_manifest_schema.py tests/experimental/test_seed_manifest_expected_roles.py tests/ranking/test_claim_guard_manifest_l0.py tests/v4_reports/test_no_forbidden_claims_in_manifest_report.py

test-v4:
	$(VENV)/bin/pytest tests/experimental tests/guided tests/provenance tests/mechanism tests/benchmarks tests/ranking/test_claim_guard_manifest_l0.py tests/ranking/test_evidence_card_v4.py tests/ranking/test_candidate_accommodation_v4.py tests/ranking/test_q382r_not_catalytic_lead.py tests/v4_reports tests/stages/test_s04x_run_load_parity.py

demo:
	$(VENV)/bin/evoliez run -c configs/example_fdh_nadp.yaml --backend mock

lint:
	$(VENV)/bin/python -m compileall -q src

clean:
	rm -rf .pytest_cache .coverage htmlcov **/__pycache__ src/**/__pycache__
	rm -rf runs/demo_fdh_nadp
