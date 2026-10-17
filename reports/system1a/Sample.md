# System 1a (LightGBM): Sample

Measured on the test split. The model is fitted on train, early-stopped on the last train day and calibrated on validation. The baseline is the validation-selected rule config (`tuned`).

## Ranking and calibration

| Model | PR-AUC | ROC-AUC | Calibrator | ECE raw | ECE calibrated | Brier calibrated |
|---|---:|---:|---|---:|---:|---:|
| rules (baseline) | 0.0599 | 0.7531 | n/a | n/a | n/a | n/a |
| lightgbm[all] | 0.6856 | 0.9920 | isotonic | 0.00646 | 0.01706 | 0.015270 |
| lightgbm[no_shortcuts] | 0.4471 | 0.9570 | platt | 0.00444 | 0.02481 | 0.017803 |

Test base rate: 1.948%.

## Against the rule baseline at matched operating points

| Model | Comparison | Alerts | Model precision | Model recall | Baseline precision | Baseline recall | FP reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| lightgbm[all] | matched volume | 184 | 12.50% | 100.00% | 8.15% | 65.22% | 4.7% |
| lightgbm[all] | matched recall | 26 | 57.69% | 65.22% | 8.15% | 65.22% | 93.5% |
| lightgbm[no_shortcuts] | matched volume | 184 | 10.87% | 86.96% | 8.15% | 65.22% | 3.0% |
| lightgbm[no_shortcuts] | matched recall | 53 | 28.30% | 65.22% | 8.15% | 65.22% | 77.5% |

## Analyst budgets (precision@k)

| Model | k | Precision | Recall |
|---|---:|---:|---:|
| lightgbm[all] | 100 | 23.00% | 100.00% |
| lightgbm[all] | 1,000 | 2.30% | 100.00% |
| lightgbm[all] | 10,000 | 1.95% | 100.00% |
| lightgbm[no_shortcuts] | 100 | 18.00% | 78.26% |
| lightgbm[no_shortcuts] | 1,000 | 2.30% | 100.00% |
| lightgbm[no_shortcuts] | 10,000 | 1.95% | 100.00% |

## Top features by gain

- **all**: payment_format_code, amount_usd_log, dst_usd_24h_log, src_usd_24h_log, src_tx_24h, src_in_usd_log, src_out_usd_log, weekday
- **no_shortcuts**: amount_usd_log, dst_usd_24h_log, src_usd_24h_log, src_tx_24h, src_in_usd_log, src_out_usd_log, hour, src_age_days
