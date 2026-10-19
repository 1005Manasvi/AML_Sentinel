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
   paired bootstrap CIs, and calibration reported honestly (validation-fitted
   isotonic **worsened** test ECE from {ECE_RAW} to {ECE_CAL} due to drift).
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
| `{N_TESTS}` / `{COVERAGE}` | 424 / 97% | `make test` output for the v1.0.0 tag (Phase 9 run) |
| `{GNN_DIFF}` | −0.002 [95% CI −0.013, +0.011] | `reports/system2/HI-Small_gnn.json` |

## Wording to avoid

- "Reduced false positives by 99%" without "versus a rule baseline" and "synthetic
  data": the baseline is weak (docs/RESULTS.md explains why).
- "Production-ready", "real-time at bank scale", "deployed": none of these were done.
- Any Jev result: Jev was never run.
