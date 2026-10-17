# System 1a (LightGBM): HI-Small

Measured on the test split. The model is fitted on train, early-stopped on the last train day and calibrated on validation. The baseline is the validation-selected rule config (`static`).

## Ranking and calibration

| Model | PR-AUC | ROC-AUC | Calibrator | ECE raw | ECE calibrated | Brier calibrated |
|---|---:|---:|---|---:|---:|---:|
| rules (baseline) | 0.0016 | 0.6093 | n/a | n/a | n/a | n/a |
| lightgbm[all] | 0.4162 | 0.9751 | isotonic | 0.00027 | 0.00036 | 0.000807 |
| lightgbm[no_shortcuts] | 0.3716 | 0.9445 | isotonic | 0.00026 | 0.00051 | 0.000828 |

Test base rate: 0.111%.

## Against the rule baseline at matched operating points

| Model | Comparison | Alerts | Model precision | Model recall | Baseline precision | Baseline recall | FP reduction |
|---|---|---:|---:|---:|---:|---:|---:|
| lightgbm[all] | matched volume | 268,270 | 0.36% | 100.00% | 0.19% | 52.09% | 0.2% |
| lightgbm[all] | matched recall | 2,830 | 17.60% | 52.09% | 0.19% | 52.09% | 99.1% |
| lightgbm[no_shortcuts] | matched volume | 268,270 | 0.34% | 95.82% | 0.19% | 52.09% | 0.2% |
| lightgbm[no_shortcuts] | matched recall | 5,054 | 9.85% | 52.09% | 0.19% | 52.09% | 98.3% |

## Analyst budgets (precision@k)

| Model | k | Precision | Recall |
|---|---:|---:|---:|
| lightgbm[all] | 100 | 99.00% | 10.36% |
| lightgbm[all] | 1,000 | 40.90% | 42.78% |
| lightgbm[all] | 10,000 | 6.21% | 64.96% |
| lightgbm[no_shortcuts] | 100 | 99.00% | 10.36% |
| lightgbm[no_shortcuts] | 1,000 | 37.30% | 39.02% |
| lightgbm[no_shortcuts] | 10,000 | 5.65% | 59.10% |

## Top features by gain

- **all**: amount_usd_log, payment_format_code, dst_usd_24h_log, src_tx_1h, src_usd_24h_log, dst_in_usd_log, src_tx_24h, dst_age_days
- **no_shortcuts**: amount_usd_log, dst_usd_24h_log, src_usd_24h_log, dst_in_usd_log, src_age_days, src_tx_1h, dst_age_days, src_flow_ratio
