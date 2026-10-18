# Evaluation: HI-Small

Regenerate with `aml-sentinel evaluate --dataset HI-Small` (seed 42). Synthetic data: none of this shows real-world performance.

## 1. Every test transaction: rules vs LightGBM

862,792 test transactions, 956 laundering. Rule baseline `static`: precision 0.19%, recall 52.1%, FPR 31.1%, 134,135 alerts/day, PR-AUC 0.0016.

| Model | PR-AUC | P@100 | P@1,000 | P@10,000 | Alerts for the rules' recall (model vs rules) | FP reduction at matched recall | Recall at the rules' alert volume (model vs rules) |
|---|---:|---:|---:|---:|---:|---:|---:|
| lgbm_all | 0.4162 | 99.0% | 40.9% | 6.2% | 2,830 vs 268,270 | 99.1% | 100.0% vs 52.1% |
| lgbm_no_shortcuts | 0.3716 | 99.0% | 37.3% | 5.7% | 5,054 vs 268,270 | 98.3% | 95.8% vs 52.1% |

The rules are a weak bar (ADR-009), so the false-positive reduction is large by construction; precision@k and PR-AUC describe the model itself.

## 2. Ablation on the alert queue

Same 4,000 sampled test alerts from the System 1a queue for every row (271 laundering, base rate 6.8%). Each step adds inputs to the previous one; every row is a logistic stacker plus calibrator fitted on 4,000 validation alerts. Δ is the paired-bootstrap PR-AUC change versus the previous row (row resampling, so the intervals are optimistic). FP@5% counts false positives among the top 5% of test alerts (the escalation budget).

| Step | PR-AUC | ROC-AUC | P@100 | ECE | FP@5% | FP reduction vs rules | Δ PR-AUC [95% CI] | Note |
|---|---:|---:|---:|---:|---:|---:|---|---|
| rules only | 0.116 | 0.670 | 13.0% | 0.0261 | 170 | 0.0% | - | validation-selected rule baseline (ADR-009) |
| + LightGBM | 0.664 | 0.876 | 96.0% | 0.0214 | 49 | 71.2% | +0.548 [+0.501, +0.595] | calibrated lgbm_all |
| + Laya | 0.675 | 0.878 | 96.0% | 0.0239 | 50 | 70.6% | +0.010 [+0.006, +0.015] | laya[head_only], escalate probability |
| + Jev (benchmark) | - | - | - | - | - | - | - | not run: JEV_ENABLED=false or no key (benchmark-only, synthetic data only) |
| + graph evidence | 0.686 | 0.885 | 93.0% | 0.0272 | 50 | 70.6% | +0.011 [-0.008, +0.031] | past-only System 2 detector evidence (strict horizon) |
| + GNN (full stack) | 0.706 | 0.911 | 95.0% | 0.0284 | 49 | 71.2% | +0.020 [+0.007, +0.034] | LOOKAHEAD: end-of-period GraphSAGE sees the whole split, including activity after each alert; not deployable in real time (ADR-016) |

Production scores on the same alerts (not refitted):

| Score | PR-AUC | P@100 | ECE | FP@5% | Source |
|---|---:|---:|---:|---:|---|
| production: System 1 p_triage (queue ranking) | 0.674 | 96.0% | 0.0214 | 52 | fused[lgbm_all+laya[head_only]], ADR-020 |
| production: final score (System 1 + System 2) | 0.651 | 93.0% | 0.0272 | 48 | ADR-018 |

The ablation's '+ Laya' uses only Laya's escalate probability; production fusion also uses its expected risk and benign probability, which were not persisted, so the two rows differ slightly.

### Reliability of p_triage on test alerts

| p_triage band | Alerts | Mean p_triage | Observed laundering rate |
|---|---:|---:|---:|
| 0.011-0.016 | 400 | 0.014 | 0.8% |
| 0.016-0.019 | 400 | 0.018 | 1.8% |
| 0.019-0.022 | 400 | 0.021 | 0.5% |
| 0.022-0.026 | 400 | 0.024 | 1.2% |
| 0.026-0.029 | 400 | 0.027 | 2.2% |
| 0.029-0.036 | 400 | 0.032 | 3.0% |
| 0.036-0.044 | 400 | 0.039 | 1.8% |
| 0.044-0.067 | 400 | 0.053 | 4.0% |
| 0.067-0.174 | 400 | 0.103 | 6.0% |
| 0.174-1.000 | 400 | 0.559 | 46.5% |

## 3. Calibration (System 1a LightGBM, every test transaction)

Chosen on validation cross-fit: `isotonic` (held-out validation ECE: platt 0.00015, isotonic 0.00009, isotonic_smoothed 0.00010). Recent-window candidates are fitted on the last validation day only (482,773 rows).

| Calibrator | Test ECE (quantile) | Brier | Mean predicted | Observed rate | Max predicted | Share at 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| raw (uncalibrated) | 0.00027 | 0.000779 | 0.084% | 0.111% | 1.000 | 0.000% |
| platt | 0.00040 | 0.000784 | 0.140% | 0.111% | 1.000 | 0.000% |
| isotonic | 0.00036 | 0.000807 | 0.144% | 0.111% | 1.000 | 0.018% |
| isotonic_smoothed | 0.00034 | 0.000801 | 0.142% | 0.111% | 0.988 | 0.000% |
| isotonic_recent_window | 0.00040 | 0.000808 | 0.150% | 0.111% | 1.000 | 0.021% |
| platt_recent_window | 0.00043 | 0.000785 | 0.146% | 0.111% | 1.000 | 0.000% |

Final ranker (alert queue): validation chose `isotonic`; held-out validation ECE platt 0.0168, isotonic 0.0158, isotonic_smoothed 0.0173.

## 4. Attempt overlap (ADR-003)

956 test positives; queue = top 8,628 by LightGBM (top 1%). Each group is ranked against all test negatives.

| Group | Positives | Share | Recall in queue | PR-AUC |
|---|---:|---:|---:|---:|
| attempt_seen_before_test | 375 | 39.2% | 86.9% | 0.609 |
| attempt_new_in_test | 210 | 22.0% | 85.2% | 0.493 |
| undocumented_positive | 371 | 38.8% | 26.4% | 0.009 |
| account_seen_laundering_before | 484 | 50.6% | 65.7% | 0.444 |
| account_new | 472 | 49.4% | 60.4% | 0.284 |

## 5. Per-stage latency

Machine: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD, 16 logical CPUs, GPU NVIDIA GeForce RTX 3050 Laptop GPU. Single process.

| Stage | Latency | Measured |
|---|---:|---|
| features (all past-only features, batch) | 7.73 µs/tx | now |
| rules (velocity over history + 3 rules on test) | 2.92 µs/tx | now |
| LightGBM inference (batch) | 8.43 µs/tx | now |
| Laya laya[zero_shot] (per alert, GPU) | 449 ms/alert | Phase 4 run (reports/system1b), RTX 3050 Laptop GPU |
| Laya laya[head_only] (per alert, GPU) | 304 ms/alert | Phase 4 run (reports/system1b), RTX 3050 Laptop GPU |
| System 2 case build + detectors (per alert) | 17 ms median, 36 ms p95 | Phase 5 run (system2_evidence.parquet) |
| SAR draft (local template + grounding, per case) | 0.4 ms median | now |
| GNN (end-of-period batch: build graphs + score all pairs) | 9 s total | now |

Figures: `HI-Small_pr_full_split.png`, `HI-Small_ablation_pr.png`, `HI-Small_calibration.png`
