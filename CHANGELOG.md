# Changelog

All notable changes are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [1.0.2] - 2026-10-20

### Fixed

- CI's `mypy --strict` failed with `import-not-found` for `torch` and
  `huggingface_hub`. Both are lazy imports from the optional Laya/GNN stacks, which CI
  does not install; they now share the existing optional-import mypy exemption.
  Verified in a fresh venv with only the CI dependency set: mypy, ruff and bandit are
  clean, and pytest gives 417 passed, 3 skipped (GNN), 93.97% coverage.
- The README coverage badge now shows the CI-measured 94% instead of the 96.8% from a
  machine with the optional stacks installed.

## [1.0.1] - 2026-10-20

### Fixed

- CI failed at the hash-verified install on Linux: `ipython` needs `pexpect` only on
  non-Windows systems, and the lockfile compiled on Windows had dropped it. `pexpect`
  and `ptyprocess` are now pinned in `requirements/dev.txt`; no other pin changed.

### Added

- `tests/test_lockfiles.py`: fails if any pinned package needs an unpinned dependency
  under Linux or Windows markers, so a Windows re-lock cannot break CI silently again.

## [1.0.0] - 2026-10-20

First complete release. Every number below is measured on **synthetic** data
(IBM AMLworld HI-Small test split, seed 42) and does not prove real-world
performance. Full results: [docs/RESULTS.md](docs/RESULTS.md).

### Highlights

- **System 1:** rule baseline; LightGBM over past-only graph features; Laya decision
  model (zero-shot and head-only fine-tuned); logistic fusion; validation-chosen
  calibration. All behind one swappable `Triage` protocol.
- **System 2:** NetworkX detectors (circular, structuring, fan-in/out,
  scatter-gather, gather-scatter) on past-only case subgraphs; a GraphSAGE experiment.
- **SAR drafting:** local template by default; optional pseudonymised external
  drafter; field-level grounding; mandatory review stamp; audited.
- **API and dashboard:** authenticated FastAPI (keys, roles, rate limits, strict
  CORS and headers) and a Streamlit analyst dashboard with ranked queue, case graph,
  SAR editor, decisions and audit-log view.
- **Demo:** `make demo` or `docker compose up --build` runs everything on the bundled
  synthetic sample, with no Kaggle data, GPU or API key.

### Measured results (HI-Small test)

- At the rule baseline's recall, false positives fall 99.1% (267,772 → 2,332). The
  baseline is weak; PR-AUC is 0.416 vs 0.0016.
- Inside the alert queue: PR-AUC 0.116 (rules) → 0.664 (+ LightGBM) → 0.675 (+ Laya).
  False positives in the top 5%: 170 → 49.
- Laya alone is near random (PR-AUC 0.096). Graph evidence and the GNN add no
  reliable, deployable ranking gain. Validation-fitted calibration worsened test ECE.
  See "What did not work" in RESULTS.md.

### Added in the final phase (Phase 9)

- `aml-sentinel demo` / `make demo`: rebuilds the sample's models, mints per-run
  analyst and admin keys (never logged), runs API + dashboard (ADR-024).
- Dockerfile (digest-pinned slim base, hash-verified deps, non-root) and
  `compose.yaml` (loopback-only ports, all capabilities dropped).
- `docs/RESULTS.md`, `docs/BLOG_POST.md`, `docs/RESUME_BULLETS.md`,
  `docs/INTERVIEW_PREP.md`, `CONTRIBUTING.md`, this changelog, MIT `LICENSE`.
- The docs consistency test now also checks RESULTS.md and the README key results
  against the committed reports.

### Fixed

- `make reproduce` skipped the Laya and GNN stages that `rank` and `evaluate` read;
  it now runs every stage in dependency order and ends with the docs consistency test.
- The `check-toml` pre-commit hook was blocked by Windows Smart App Control; it now
  runs as a Python module.

### Known limitations

- Synthetic data only. The model largely learned the simulator's documented patterns
  (26% queue recall on undocumented laundering).
- The Docker image was not built on the development machine (no Docker installed).
- No fairness measurement was possible (no customer attributes). Jev was never run.
  The full Laya fine-tune is prepared but unmeasured.
