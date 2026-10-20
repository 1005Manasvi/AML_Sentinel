# AML Sentinel developer commands.
# On Windows without GNU make, run the commands under each target by hand
# (or install make via `winget install GnuWin32.Make` / use Git Bash + mingw32-make).

ifeq ($(OS),Windows_NT)
	BIN := .venv/Scripts
else
	BIN := .venv/bin
endif
PY := $(BIN)/python

.PHONY: sample-models setup lock lint format typecheck test security check sample data baseline notebook run train laya-setup laya-data finetune-laya triage-text detect gnn-setup gnn rank sar evaluate llm-setup api dashboard api-key demo drift reproduce clean

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
	for lock in base dev laya gnn llm; do $(PY) -m pip_audit --strict --require-hashes --disable-pip -r requirements/$$lock.txt || exit 1; done

check: lint typecheck test security

sample-models:  ## Sample only: data, rule baseline and LightGBM, enough for `make run`
	$(MAKE) sample
	$(PY) -m aml_sentinel.cli baseline --dataset Sample
	$(PY) -m aml_sentinel.cli train-tabular --dataset Sample

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

laya-setup:  ## Optional text-model stack; then CUDA torch for NVIDIA GPUs (ADR-013)
	$(PY) -m pip install --require-hashes -r requirements/laya.txt
	@echo "GPU: $(PY) -m pip install --force-reinstall --no-deps torch==2.14.1+cu126 --index-url https://download.pytorch.org/whl/cu126"

laya-data:  ## Laya fine-tuning JSONL from the HI-Small train split (+ generator structuring)
	$(PY) -m aml_sentinel.cli build-laya-data --dataset HI-Small

finetune-laya:  ## Head-only Laya fine-tune on the local GPU (~1.5 h on a 4 GB RTX 3050)
	$(PY) -m aml_sentinel.cli finetune-laya --dataset HI-Small

triage-text:  ## System 1b: Laya variants (+Jev only if JEV_ENABLED) + fusion
	$(PY) -m aml_sentinel.cli triage-text --dataset HI-Small

detect:  ## System 2: NetworkX detectors on the alert-queue samples (strict + next-day)
	$(PY) -m aml_sentinel.cli detect-patterns --dataset Sample
	$(PY) -m aml_sentinel.cli detect-patterns --dataset HI-Small

gnn-setup:  ## Optional GNN experiment stack (after laya-setup)
	$(PY) -m pip install --require-hashes -r requirements/gnn.txt

gnn:  ## System 2 experiment: GraphSAGE pair classifier vs LightGBM and detectors
	$(PY) -m aml_sentinel.cli gnn-experiment --dataset HI-Small

rank:  ## Final calibrated ranking (System 1 p_triage + System 2 evidence)
	$(PY) -m aml_sentinel.cli rank --dataset HI-Small

evaluate:  ## Phase 8: ablation, calibration study, attempt overlap, latency -> reports/evaluation/
	$(PY) -m aml_sentinel.cli evaluate --dataset Sample
	$(PY) -m aml_sentinel.cli evaluate --dataset HI-Small

drift:  ## PSI drift per feature and for the score (train/val/test) -> reports/drift/
	$(PY) -m aml_sentinel.cli drift --dataset Sample
	$(PY) -m aml_sentinel.cli drift --dataset HI-Small

sar:  ## Draft review-stamped, grounding-checked SARs for the top test alerts (template by default)
	$(PY) -m aml_sentinel.cli draft-sar --dataset HI-Small

llm-setup:  ## Optional external SAR drafter (Anthropic SDK); only pseudonymised facts are sent
	$(PY) -m pip install --require-hashes -r requirements/llm.txt

notebook:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

api:  ## Authenticated API on 127.0.0.1:8000 (needs API_KEYS; serves API_DATASET, default Sample)
	$(PY) -m aml_sentinel.cli serve --host 127.0.0.1 --port 8000

dashboard:  ## Streamlit analyst dashboard on 127.0.0.1:8501 (talks to the API)
	$(BIN)/streamlit run src/aml_sentinel/dashboard/app.py

api-key:  ## Print a new key and its API_KEYS entry: make api-key KEY_USER=alice KEY_ROLE=analyst
	$(PY) -m aml_sentinel.cli new-api-key --user $(KEY_USER) --role $(KEY_ROLE)

KEY_USER ?= analyst1
KEY_ROLE ?= analyst

run:  ## API + dashboard together (two processes; Ctrl+C stops both)
	$(MAKE) -j2 api dashboard

train:  ## System 1a: LightGBM + calibration vs baseline (needs `make baseline`)
	$(PY) -m aml_sentinel.cli train-tabular --dataset Sample
	$(PY) -m aml_sentinel.cli train-tabular --dataset HI-Small
	$(PY) -m aml_sentinel.cli train-tabular --dataset LI-Small

demo:  ## Rebuild sample models (~1 min), then API + dashboard with keys minted for this run
	$(PY) -m aml_sentinel.cli demo

reproduce:  ## Every README/RESULTS table and figure. Needs data/raw, laya-setup + gnn-setup, NVIDIA GPU (~3 h)
	$(MAKE) sample
	$(MAKE) data
	$(MAKE) baseline
	$(MAKE) train
	$(MAKE) laya-data
	$(MAKE) finetune-laya
	$(MAKE) triage-text
	$(MAKE) detect
	$(MAKE) gnn
	$(MAKE) rank
	$(MAKE) sar
	$(MAKE) evaluate
	$(MAKE) drift
	$(MAKE) notebook
	$(PY) -m pytest tests/test_docs_consistency.py --no-cov -q

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache .hypothesis .coverage htmlcov
