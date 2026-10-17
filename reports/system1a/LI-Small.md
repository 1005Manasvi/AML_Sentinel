# System 1a (LightGBM): LI-Small

Measured on the test split. The model is fitted on train, early-stopped on the last train day and calibrated on validation. The baseline is the validation-selected rule config (`tuned`).

## Ranking and calibration

| Model | PR-AUC | ROC-AUC | Calibrator | ECE raw | ECE calibrated | Brier calibrated |
|---|---:|---:|---|---:|---:|---:|
| rules (baseline) | 0.0007 | 0.5654 | n/a | n/a | n/a | n/a |
| lightgbm[all] | 0.1011 | 0.9573 | isotonic | 0.00012 | 0.00032 | 0.000539 |
| lightgbm[no_shortcuts] | 0.0804 | 0.9148 | isotonic | 0.00013 | 0.00041 | 0.000545 |

Test base rate: 0.057%.

## Against the rule baseline at matched operating points

| Model | Comparison | Alerts | Model precision | Model recall | Baseline precision | Baseline recall | FP reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| lightgbm[all] | matched volume | 108,398 | 0.51% | 82.93% | 0.14% | 22.31% | 0.4% |
| lightgbm[all] | matched recall | 4,012 | 3.71% | 22.31% | 0.14% | 22.31% | 96.4% |
| lightgbm[no_shortcuts] | matched volume | 108,398 | 0.40% | 64.22% | 0.14% | 22.31% | 0.3% |
| lightgbm[no_shortcuts] | matched recall | 4,289 | 3.47% | 22.31% | 0.14% | 22.31% | 96.2% |

## Analyst budgets (precision@k)

| Model | k | Precision | Recall |
|---|---:|---:|---:|
| lightgbm[all] | 100 | 53.00% | 7.93% |
| lightgbm[all] | 1,000 | 9.90% | 14.82% |
| lightgbm[all] | 10,000 | 2.10% | 31.44% |
| lightgbm[no_shortcuts] | 100 | 43.00% | 6.44% |
| lightgbm[no_shortcuts] | 1,000 | 7.90% | 11.83% |
| lightgbm[no_shortcuts] | 10,000 | 2.10% | 31.44% |

## Top features by gain

- **all**: payment_format_code, amount_usd_log, src_tx_1h, dst_usd_24h_log, dst_age_days, src_usd_24h_log, dst_in_usd_log, src_tx_24h
- **no_shortcuts**: amount_usd_log, dst_in_usd_log, src_tx_1h, dst_usd_24h_log, src_age_days, src_usd_24h_log, src_flow_ratio, dst_age_days
