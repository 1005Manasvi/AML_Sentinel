# Evaluation: Sample

Regenerate with `aml-sentinel evaluate --dataset Sample` (seed 42). Synthetic data: none of this shows real-world performance.

## 1. Every test transaction: rules vs LightGBM

1,166 test transactions, 8 laundering. Rule baseline `tuned`: precision 4.52%, recall 100.0%, FPR 14.6%, 88 alerts/day, PR-AUC 0.0452.

| Model | PR-AUC | P@100 | P@1,000 | P@10,000 | Alerts for the rules' recall (model vs rules) | FP reduction at matched recall | Recall at the rules' alert volume (model vs rules) |
|---|---:|---:|---:|---:|---:|---:|---:|
| lgbm_all | 0.1823 | 7.0% | 0.8% | 0.7% | 285 vs 177 | -63.9% | 87.5% vs 100.0% |
| lgbm_no_shortcuts | 0.0871 | 6.0% | 0.8% | 0.7% | 259 vs 177 | -48.5% | 87.5% vs 100.0% |

The rules are a weak bar (ADR-009), so the false-positive reduction is large by construction; precision@k and PR-AUC describe the model itself.

## 2. Ablation

Not available: needs System 1b scores (`triage-text`).

## 3. Calibration (System 1a LightGBM, every test transaction)

Chosen on validation cross-fit: `isotonic_smoothed` (held-out validation ECE: platt 0.00965, isotonic 0.00882, isotonic_smoothed 0.00800). Recent-window candidates are fitted on the last validation day only (657 rows).

| Calibrator | Test ECE (quantile) | Brier | Mean predicted | Observed rate | Max predicted | Share at 1.0 |
|---|---:|---:|---:|---:|---:|---:|
| raw (uncalibrated) | 0.01137 | 0.007970 | 1.731% | 0.686% | 0.772 | 0.000% |
| platt | 0.01617 | 0.007949 | 2.303% | 0.686% | 0.680 | 0.000% |
| isotonic | 0.01853 | 0.008283 | 2.412% | 0.686% | 0.500 | 0.000% |
| isotonic_smoothed | 0.01708 | 0.007793 | 2.270% | 0.686% | 0.408 | 0.000% |
| isotonic_recent_window | 0.01658 | 0.007881 | 2.172% | 0.686% | 0.500 | 0.000% |
| platt_recent_window | 0.01394 | 0.007518 | 2.080% | 0.686% | 0.643 | 0.000% |

## 4. Attempt overlap (ADR-003)

8 test positives; queue = top 200 by LightGBM (top 1%). Each group is ranked against all test negatives.

| Group | Positives | Share | Recall in queue | PR-AUC |
|---|---:|---:|---:|---:|
| attempt_seen_before_test | 8 | 100.0% | 87.5% | 0.182 |
| attempt_new_in_test | 0 | 0.0% | n/a | n/a |
| undocumented_positive | 0 | 0.0% | n/a | n/a |
| account_seen_laundering_before | 8 | 100.0% | 87.5% | 0.182 |
| account_new | 0 | 0.0% | n/a | n/a |

## 5. Per-stage latency

Machine: AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD, 16 logical CPUs, GPU NVIDIA GeForce RTX 3050 Laptop GPU. Single process.

| Stage | Latency | Measured |
|---|---:|---|
| features (all past-only features, batch) | 8.88 µs/tx | now |
| rules (velocity over history + 3 rules on test) | 5.71 µs/tx | now |
| LightGBM inference (batch) | 6.78 µs/tx | now |
| System 2 case build + detectors (per alert) | 39 ms median, 352 ms p95 | Phase 5 run (system2_evidence.parquet) |
| API POST /score (one new transaction, past-only features) | 67.4 ms median | now |

Figures: `Sample_pr_full_split.png`, `Sample_calibration.png`
