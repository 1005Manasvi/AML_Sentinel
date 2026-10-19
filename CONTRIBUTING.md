# Contributing

Thanks for looking. This is a portfolio project, but issues and pull requests are
welcome.

## Setup

```bash
make setup   # venv, hash-verified dependencies, pre-commit hooks
make check   # ruff, ruff format, mypy --strict, pytest (>= 85% coverage), bandit, pip-audit
```

On Windows, run `make` from Git Bash. If you commit from a shell where the venv is
not activated, put `.venv/Scripts` (or `.venv/bin`) on `PATH` first, because the
ruff, bandit and mypy hooks run from the venv.

## Rules

- **No real customer data, ever.** Use the bundled sample or IBM AMLworld. Never
  commit `data/raw/`, `data/processed/`, `models/`, SAR drafts or `.env`.
- **No secrets** in code, config, tests or logs. Settings come from the environment
  (`.env.example`).
- **Numbers in docs must come from a committed report.** If you change a result,
  regenerate the report and run `pytest tests/test_docs_consistency.py`.
- **Report what did not work.** A negative result goes in `docs/RESULTS.md`, not in
  the bin.
- Dependencies: edit `requirements/*.in`, run `make lock`, and commit the hashed
  `.txt` files. Pin new GitHub Actions to a full commit SHA.
- Untrusted text (accounts, notes, SAR text) only goes to plain-text dashboard
  widgets. A test enforces this.
- Full type hints (`mypy --strict`), Google-style docstrings, and comments that
  explain why rather than what.
- Record a non-obvious design choice as a short ADR in `docs/DECISIONS.md`.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/): `feat(api): ...`,
`fix(models): ...`, `docs: ...`, `build(deps): ...`. Keep commits small and
self-contained. Never skip the hooks.

## Security issues

Please do not open a public issue; see [SECURITY.md](SECURITY.md).
