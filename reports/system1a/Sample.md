# System 1a (LightGBM): Sample

Measured on the test split. The model is fitted on train, early-stopped on the last train day and calibrated on validation. The baseline is the validation-selected rule config (`tuned`).

## Ranking and calibration

| Model | PR-AUC | ROC-AUC | Calibrator | ECE raw | ECE calibrated | Brier calibrated |
|---|---:|---:|---|---:|---:|---:|
| rules (baseline) | 0.0452 | 0.9270 | n/a | n/a | n/a | n/a |
| lightgbm[all] | 0.1823 | 0.9537 | isotonic | 0.01137 | 0.01853 | 0.008283 |
| lightgbm[no_shortcuts] | 0.0871 | 0.9313 | platt | 0.01635 | 0.01774 | 0.008977 |

Test base rate: 0.686%.

## Against the rule baseline at matched operating points

| Model | Comparison | Alerts | Model precision | Model recall | Baseline precision | Baseline recall | FP reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| lightgbm[all] | matched volume | 177 | 3.95% | 87.50% | 4.52% | 100.00% | -0.6% |
| lightgbm[all] | matched recall | 285 | 2.81% | 100.00% | 4.52% | 100.00% | -63.9% |
| lightgbm[no_shortcuts] | matched volume | 177 | 3.95% | 87.50% | 4.52% | 100.00% | -0.6% |
| lightgbm[no_shortcuts] | matched recall | 259 | 3.09% | 100.00% | 4.52% | 100.00% | -48.5% |

## Analyst budgets (precision@k)

| Model | k | Precision | Recall |
|---|---:|---:|---:|
| lightgbm[all] | 100 | 7.00% | 87.50% |
| lightgbm[all] | 1,000 | 0.80% | 100.00% |
| lightgbm[all] | 10,000 | 0.69% | 100.00% |
| lightgbm[no_shortcuts] | 100 | 6.00% | 75.00% |
| lightgbm[no_shortcuts] | 1,000 | 0.80% | 100.00% |
| lightgbm[no_shortcuts] | 10,000 | 0.69% | 100.00% |

## Top features by gain

- **all**: amount_usd_log, dst_flow_ratio, payment_format_code, dst_usd_24h_log, src_usd_24h_log, dst_age_days, src_age_days, dst_out_tx
- **no_shortcuts**: amount_usd_log, dst_flow_ratio, dst_usd_24h_log, dst_age_days, dst_out_tx, src_usd_24h_log, self_transfer, src_age_days
