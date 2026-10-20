# Resume bullets

Templates first, then the measured value for each placeholder and where it comes
from. Use only these values, or re-measured ones from `make reproduce`. Always keep
"synthetic data" in the bullet: a recruiter who checks the repo will see it anyway.

## Templates

1. Built an anti-money-laundering monitoring system in Python (LightGBM, NetworkX,
   FastAPI, Streamlit) that matched a rule baseline's recall with **{FP_REDUCTION}
   fewer false positives** ({FP_RULES} → {FP_MODEL}) on {N_TEST} synthetic IBM
   AMLworld test transactions, using time-split, past-only graph features.
2. Designed a leakage-proof evaluation: time-based splits, past-only features
   verified by property and label-scrambling tests, a {N_ABLATION}-step ablation with
   paired bootstrap CIs, and calibration reported honestly: validation-fitted
   isotonic **worsened** test ECE from {ECE_RAW} to {ECE_CAL}, while a PSI drift
   report showed the score distribution stayed stable (PSI {SCORE_PSI}), so the cause
   was a changing score-to-label relationship.
3. Fine-tuned and benchmarked an open-source decision model (Laya) against LightGBM:
   found it near random alone (PR-AUC {LAYA_AP} vs {LGBM_QUEUE_AP}) but a small
   significant gain in fusion (+{LAYA_GAIN} PR-AUC), and documented the trade-off of
   {LAYA_MS} ms of GPU time per alert versus {LGBM_US} µs per transaction.
4. Secured an analyst API and dashboard: hashed API keys with constant-time
   comparison, role-based access, rate limits, strict CORS and headers, a
   hash-chained audit log, and prompt-injection-resistant, pseudonymised SAR drafting
   with field-level grounding checks; {N_TESTS} tests at {COVERAGE} coverage, mypy
   --strict, bandit, pip-audit and gitleaks in CI.
5. Ran a GraphSAGE experiment against graph-pattern detectors and LightGBM on
   identical inputs. Message passing added nothing measurable ({GNN_DIFF} PR-AUC),
   and the same-input ablation exposed a feature bug in the first run.
6. Added per-alert explainability with exact TreeSHAP (LightGBM `pred_contrib`) to the
   analyst API and dashboard, verified by an additivity test against the model's own
   log-odds.

## Values to plug in

All HI-Small test, seed 42, measured by this repository.

| Placeholder | Value | Source |
|---|---|---|
| `{FP_REDUCTION}` | 99.1% | `reports/system1a/HI-Small.json` → `variants.all.vs_baseline.matched_recall.fp_reduction` |
| `{FP_RULES}` / `{FP_MODEL}` | 267,772 / 2,332 | same, `.baseline.fp` / `.model.fp` |
| `{N_TEST}` | 862,792 | same, `rows.test` |
| `{N_ABLATION}` | 5 (6 rows incl. Jev "not run") | `reports/evaluation/HI-Small.json` → `ablation.steps` |
| `{ECE_RAW}` / `{ECE_CAL}` | 0.00027 / 0.00036 | `reports/system1a/HI-Small.md` (System 1a table in RESULTS.md) |
| `{LAYA_AP}` / `{LGBM_QUEUE_AP}` | 0.096 / 0.649 | `reports/system1b/HI-Small.json` → `models` |
| `{LAYA_GAIN}` | 0.010 [95% CI +0.006, +0.015] | `reports/evaluation/HI-Small.json` → ablation "+ Laya" |
| `{LAYA_MS}` / `{LGBM_US}` | 304 / 8.4 | `reports/evaluation/HI-Small.json` → `latency.stages` |
| `{N_TESTS}` / `{COVERAGE}` | 438 / 94% | v1.1.0: 438 pass locally with the optional Laya/GNN stacks (96.8% coverage). CI installs neither, skips the 3 GNN tests and measured 93.97% at v1.0.2; check the v1.1.0 CI log before quoting it |
| `{SCORE_PSI}` | 0.043 | `reports/drift/HI-Small.json` → `score_psi_val_to_test` |
| `{GNN_DIFF}` | −0.002 [95% CI −0.013, +0.011] | `reports/system2/HI-Small_gnn.json` |

## Wording to avoid

- "Reduced false positives by 99%" without "versus a rule baseline" and "synthetic
  data": the baseline is weak (docs/RESULTS.md explains why).
- "Production-ready", "real-time at bank scale", "deployed": none of these were done.
- Any Jev result: Jev was never run.
