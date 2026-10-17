# AML Sentinel developer commands.
# On Windows without GNU make, run the commands under each target by hand
# (or install make via `winget install GnuWin32.Make` / use Git Bash + mingw32-make).

ifeq ($(OS),Windows_NT)
	BIN := .venv/Scripts
else
	BIN := .venv/bin
endif
PY := $(BIN)/python

.PHONY: setup lock lint format typecheck test security check sample data baseline notebook run train demo reproduce clean

setup:  ## Create venv and install pinned, hash-verified dependencies
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install --require-hashes -r requirements/base.txt -r requirements/dev.txt
	$(PY) -m pip install --no-deps -e .
	$(BIN)/pre-commit install

lock:  ## Re-resolve dependencies into hashed lockfiles
	cd requirements && ../$(PY) -m piptools compile --generate-hashes --allow-unsafe --strip-extras -o base.txt base.in
	cd requirements && ../$(PY) -m piptools compile --generate-hashes --allow-unsafe --strip-extras -o dev.txt dev.in

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff check --fix src tests
	$(PY) -m ruff format src tests

typecheck:
	$(PY) -m mypy

test:
	$(PY) -m pytest

security:  ## Static analysis + dependency CVE audit (gitleaks runs via pre-commit/CI)
	$(PY) -m bandit -c pyproject.toml -r src
	$(PY) -m pip_audit --strict --require-hashes -r requirements/base.txt

check: lint typecheck test security

sample:  ## Regenerate the bundled synthetic sample (seed 42) and its parquet cache
	$(PY) -m aml_sentinel.cli generate-sample --seed 42
	$(PY) -m aml_sentinel.cli prepare-data --dataset Sample

data:  ## Validate raw files and build parquet caches (needs data/raw, see data/README.md)
	$(PY) -m aml_sentinel.cli prepare-data --dataset HI-Small
	$(PY) -m aml_sentinel.cli prepare-data --dataset LI-Small

baseline:  ## Rule-engine baseline on every prepared dataset -> reports/baseline/
	$(PY) -m aml_sentinel.cli baseline --dataset Sample
	$(PY) -m aml_sentinel.cli baseline --dataset HI-Small
	$(PY) -m aml_sentinel.cli baseline --dataset LI-Small

notebook:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

run:
	@echo "API and dashboard arrive in Phase 7." && exit 1

train:  ## System 1a: LightGBM + calibration vs baseline (needs `make baseline`)
	$(PY) -m aml_sentinel.cli train-tabular --dataset Sample
	$(PY) -m aml_sentinel.cli train-tabular --dataset HI-Small
	$(PY) -m aml_sentinel.cli train-tabular --dataset LI-Small

demo:
	@echo "Demo mode (bundled sample + small models) arrives in Phase 9." && exit 1

reproduce:  ## Grows phase by phase until it regenerates every README figure/table
	$(MAKE) sample
	$(MAKE) data
	$(MAKE) baseline
	$(MAKE) train
	$(MAKE) notebook

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis .coverage htmlcov
