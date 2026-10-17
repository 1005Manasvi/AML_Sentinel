# AML Sentinel

AML Sentinel is a real-time anti-money-laundering transaction monitoring system. It
uses a fast System 1 triage stage and a slow System 2 investigation stage, and
drafts SARs for human review.

> **Status:** Phase 1 (scaffold and data) is complete. Results, the dashboard and
> full documentation arrive in later phases.
>
> **Synthetic data only.** Results on IBM AMLworld do not prove real-world
> performance.

## Problem

Over 90% of AML alerts at banks are false positives. AML Sentinel aims to:

1. detect laundering patterns on a transaction graph,
2. rank alerts by calibrated risk, to beat a rule-based baseline on false positives,
3. draft grounded SAR narratives for analyst review,
4. give analysts a case-management dashboard.

## Architecture

```mermaid
flowchart LR
    TX[Transaction] --> R[Rule engine<br/>baseline]
    TX --> GF[Graph features] --> LGB[LightGBM<br/>p_tabular]
    TX --> SUM[Alert summarizer] --> LAYA[Laya triage<br/>local, fine-tuned]
    SUM -. benchmark only,<br/>synthetic only .-> JEV[Jev adapter<br/>off by default]
    LGB --> FUS[Fusion + calibration<br/>p_triage]
    LAYA --> FUS
    JEV -.-> FUS
    R --> FUS
    FUS -->|below threshold| CLOSE[Auto-close + log]
    FUS -->|escalate ~1-5%| S2
    subgraph S2[System 2]
        DET[NetworkX detectors]
        GNN[GraphSAGE experiment]
        SAR[SAR drafting<br/>pseudonymised, grounded]
    end
    S2 --> RANK[Final calibrated score] --> UI[API + analyst dashboard]
```

## Quickstart

```bash
make setup     # venv + hash-verified install + pre-commit hooks
# put the IBM files in data/raw/ (see data/README.md)
make data      # validate + build parquet caches
make check     # ruff, mypy --strict, pytest, bandit, pip-audit
```

On Windows without GNU make, run the commands listed under each target in the
`Makefile`.

## Data

See [data/README.md](data/README.md) for the source, license check, schema,
checksums and known quirks.

## Results

### Rule baseline (Phase 2)

These numbers are measured on IBM AMLworld. The test split covers 2022-09-09
to 09-10, and alerts are counted per transaction. The baseline to beat is
chosen on validation F1 (ADR-009). The jurisdiction rule uses a **synthetic**
bank list (ADR-008).

| Dataset | Baseline | Test base rate | Precision | Recall | FPR | Alerts/day |
|---|---|---:|---:|---:|---:|---:|
| HI-Small | static (10k USD, ≥24 payments/day, synthetic banks) | 0.111% | 0.19% | 52.1% | 31.1% | 134,135 |
| LI-Small | tuned (≥50 payments/day only) | 0.057% | 0.14% | 22.3% | 9.2% | 54,199 |

To regenerate: `make data && make baseline`. Full reports are in
[reports/baseline/](reports/baseline/).

**How to read this:**

- The rules are only 1.7 times (HI) and 2.4 times (LI) better than random,
  measured as precision over the base rate.
- They create an unworkable alert volume: about 9–31% of all transactions.
- This is typical of threshold rules at this level of class imbalance. It also
  means the bar is low, so later phases compare models at **matched alert
  volumes** rather than claiming a big win against this baseline.

Full results arrive in Phases 3–8.

## Limitations

The full list is written in Phase 8. For now, note these:

- Synthetic data only.
- Typology labels cover only part of the positives.

## Docs

- [SECURITY.md](SECURITY.md)
- [docs/DECISIONS.md](docs/DECISIONS.md)
