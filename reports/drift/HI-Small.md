# Drift report: HI-Small

PSI with 10 quantile bins of the earlier period. Rule of thumb: < 0.1 stable, 0.1-0.25 moderate, >= 0.25 major shift.

| Period | Rows | Laundering rate |
|---|---:|---:|
| train | 3,248,921 | 0.078% |
| val | 965,524 | 0.107% |
| test | 862,792 | 0.111% |

**LightGBM score, validation → test:** PSI 0.0428 (stable); mean raw score 0.00064 → 0.00084.

## train->test: 20 major, 2 moderate of 35 features

| Feature | PSI | Reading |
|---|---:|---|
| weekday | 8.6055 | major |
| dst_age_days | 8.3549 | major |
| src_age_days | 8.1439 | major |
| dst_in_tx | 2.9268 | major |
| src_has_history | 2.2904 | major |
| dst_has_history | 2.2391 | major |
| src_out_tx | 1.9956 | major |
| dst_in_degree | 1.4555 | major |
| src_out_degree | 1.4290 | major |
| src_in_tx | 1.3489 | major |

## val->test: 5 major, 1 moderate of 35 features

| Feature | PSI | Reading |
|---|---:|---|
| weekday | 6.9076 | major |
| dst_age_days | 5.3707 | major |
| src_age_days | 3.8197 | major |
| dst_in_tx | 0.4112 | major |
| src_in_tx | 0.2622 | major |
| src_out_tx | 0.1640 | moderate |
| src_tx_24h | 0.0705 | stable |
| src_in_degree | 0.0584 | stable |
| src_in_usd_log | 0.0559 | stable |
| dst_out_tx | 0.0544 | stable |

## train->val: 19 major, 4 moderate of 35 features

| Feature | PSI | Reading |
|---|---:|---|
| dst_age_days | 7.9550 | major |
| src_age_days | 7.8091 | major |
| weekday | 6.6255 | major |
| dst_in_tx | 2.8749 | major |
| src_has_history | 2.3174 | major |
| dst_has_history | 2.2270 | major |
| src_out_tx | 2.0981 | major |
| src_out_usd_log | 1.4812 | major |
| dst_in_degree | 1.4704 | major |
| src_out_degree | 1.4521 | major |
