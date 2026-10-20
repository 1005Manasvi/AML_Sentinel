# Model card: AML Sentinel triage models

All numbers are measured on **synthetic** data (IBM AMLworld HI-Small) with seed 42
and come from the committed reports: `reports/system1a`, `reports/system1b`,
`reports/final` and `reports/evaluation`. They say nothing about performance on real
transactions.

## Models

| Model | Role | Type | Inputs | Output |
|---|---|---|---|---|
| Rule baseline | Comparison bar | Amount ≥ 10k USD, sender velocity, synthetic "high-risk" banks | Transaction + past-only counts | Alert yes/no |
| LightGBM (`lgbm_all`) | System 1a: screens every transaction | Gradient-boosted trees, native format | 35 past-only features (row, trailing windows, daily graph snapshots) | Calibrated probability (isotonic, validation-fitted) |
| Laya (`laya[head_only]`) | System 1b: typed questions on queued alerts | Apache-2.0 decision model; head-only fine-tune, encoder frozen | Numeric alert summary (no identifiers or free text) | Typology (5 labels), risk 1–5, escalate probability |
| Fusion stacker → `p_triage` | **Ranks the analyst queue** | Logistic regression + isotonic, validation-fitted | Calibrated LightGBM + Laya outputs | Calibrated probability |
| Final stacker → `p_final` | Informational only | Logistic regression + isotonic | `p_triage` + System 2 evidence counts | Calibrated probability |
| NetworkX detectors | System 2 evidence | Deterministic graph rules | Past-only case subgraph | Named patterns with accounts and payments |
| GraphSAGE | Experiment only | 2-layer GNN pair classifier | End-of-period account graph | Pair score (not calibrated) |

## Intended use

- **Intended:** to prioritise alerts for human AML analysts, and to explain them, on
  synthetic data, as a portfolio demonstration of a monitoring pipeline.
- **Not intended:**
  - any decision about a real customer;
  - automatic account closure or filing;
  - use on real data without model risk management, regulatory validation and a
    privacy review;
  - treating a score as evidence of a crime.
- **Users:** analysts and reviewers through the authenticated API and dashboard.
  Every SAR draft is stamped *DRAFT - requires analyst review*.

## Data

- **IBM AMLworld HI-Small** (Altman et al., NeurIPS 2023), about 5.08M transactions
  with simulated laundering. The license was not verified for redistribution, so
  the data is not committed. See `data/README.md`.
- **Time splits:** train before 2022-09-07, validation 09-07 to 09-09, test 09-09
  to 09-11. The sparse tail is excluded (ADR-003).
- **Test set:** 862,792 transactions with 956 laundering (0.11%). The alert
  sample has 4,000 alerts and 271 laundering (6.8%).
- **Bundled sample:** about 6.3k transactions from the project's own seeded
  generator, used for the demo and CI. It is not evidence of model quality (ADR-007).

## Metrics (test period, HI-Small)

**Every test transaction**

| Model | PR-AUC | Precision@100 | Precision@1,000 | FP reduction vs rules at the rules' recall (52.1%) |
|---|---:|---:|---:|---:|
| Rules (static) | 0.0016 | n/a | n/a | n/a |
| LightGBM `all` | 0.416 | 99.0% | 40.9% | 99.1% (2,830 vs 268,270 alerts) |
| LightGBM `no_shortcuts` | 0.372 | 99.0% | 37.3% | 98.3% |

**Inside the alert queue (4,000 test alerts)**

| Score | PR-AUC | ROC-AUC | Precision@100 | False positives in the top 5% |
|---|---:|---:|---:|---:|
| Rules only (re-ranking the queue) | 0.116 | 0.670 | 13.0% | 170 |
| + LightGBM | 0.664 | 0.876 | 96.0% | 49 |
| + Laya | 0.675 | 0.878 | 96.0% | 50 |
| + graph evidence | 0.686 | 0.885 | 93.0% | 50 |
| + GNN (lookahead, not deployable) | 0.706 | 0.911 | 95.0% | 49 |
| Production `p_triage` (queue ranking) | 0.674 | 0.881 | 96.0% | 52 |
| Production `p_final` (informational) | 0.651 | 0.885 | 93.0% | 48 |

- **Laya alone:** PR-AUC 0.096 and ROC-AUC 0.535 (head-only). It is close to random
  within the queue.
- **Typology:** macro-F1 0.21–0.24. Zero-shot "94% accuracy" is the majority class
  (ADR-014).
- **Escalation at the validation-fitted threshold:** 6.5% of test alerts (target 5%),
  with 64.4% precision and 62.0% recall for `p_triage`.

## Calibration

- Calibrators are fitted on validation only and chosen by validation cross-fit.
- **Inside the queue:** test ECE is 0.021 for `p_triage` and 0.027 for `p_final`.
  `p_triage` per score band on test: the top decile of alerts (scores 0.17–1.00)
  were laundering 46.5% of the time, against a mean predicted 0.56. **The highest
  scores overstate certainty.**
- **On the full split:** validation calibration made test ECE worse than the raw
  LightGBM scores (0.00036 vs 0.00027), because the score-to-label relationship
  drifts between periods (ADR-011).
- Smoothed isotonic and last-day recalibration were evaluated. Validation did not
  select them (ADR-022).
- The API therefore shows each alert's measured band reliability next to the score.
- A PSI drift report (`make drift`) shows the score distribution is stable from
  validation to test (PSI 0.043), so the calibration failure comes from a changing
  score-to-label relationship rather than shifted scores (ADR-025).

## Explanations

- Each case shows the six largest exact TreeSHAP contributions to the LightGBM
  log-odds (before calibration and fusion). They describe what the model relied on,
  an association in synthetic data, not a cause or evidence of wrongdoing.
- Available up to the scorer's 100k-row history cap (the demo), not on HI-Small
  without an online feature store.

## Failure modes

- **Shortcuts:** payment format and cross-currency are simulator artifacts. Removing
  those columns does not remove the shortcut: the `no_shortcuts` top-100 are still
  all ACH (ADR-010).
- **Undocumented laundering is mostly missed.** The 39% of test positives outside
  any documented pattern reach only 26% recall in the queue (PR-AUC 0.009), against
  85–87% for documented attempts.
- **Known actors:** 39% of test positives continue attempts that started before the
  test period, and the model ranks them better (PR-AUC 0.61 vs 0.49 for new
  attempts).
- **Detectors** confuse benign hubs with fan-ins (lift 1.3×). On the bundled sample
  they are anti-informative, because its graph is much denser (ADR-015).
- **Calibration drift** between periods (above).
- **Cumulative features drift by design:** 20 of 35 features shift "major" (PSI ≥ 0.25)
  from training to test, mostly account age and since-start counts. Production would
  need fixed-window versions.
- **Evasion:** fixed thresholds can be learned. No adversarial testing was done.

## Fairness and bias

- The data has no demographic or customer-segment attributes, so group fairness
  **cannot be measured** here. Nothing in this card supports a fairness claim.
- **Real-world risk:** transaction-monitoring models and jurisdiction rules can
  concentrate alerts on particular corridors, cash-intensive businesses or
  communities. That can lead to de-risking (account closures, exclusion).
  - The synthetic jurisdiction rule in the baseline is an example of a geographic
    proxy (ADR-008).
- **A real deployment would need:**
  - alert-rate and outcome monitoring by segment;
  - review of proxy features;
  - documented human decision-making.

## Ethical and security considerations

- No real customer data is used, requested or committed.
- Laya runs locally. Jev, a hosted service, is disabled and refuses non-synthetic
  data.
- The optional external SAR drafter receives only pseudonymised facts.
- Model artifacts are loaded only after checksum verification. Nothing is unpickled.
- See [SECURITY.md](../SECURITY.md).

## Reproduce

`make reproduce` covers data, baselines, System 1a, detectors, ranking, SARs and
evaluation. Laya fine-tuning and `triage-text` need a GPU (`make laya-setup
finetune-laya triage-text`), and the GNN needs `make gnn-setup gnn`. All steps are
seeded.
