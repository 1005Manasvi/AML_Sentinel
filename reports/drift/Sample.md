# Drift report: Sample

PSI with 10 quantile bins of the earlier period. Rule of thumb: < 0.1 stable, 0.1-0.25 moderate, >= 0.25 major shift.

| Period | Rows | Laundering rate |
|---|---:|---:|
| train | 3,879 | 4.228% |
| val | 1,279 | 3.049% |
| test | 1,166 | 0.686% |

**LightGBM score, validation → test:** PSI 0.0465 (stable); mean raw score 0.02551 → 0.01731.

## train->test: 23 major, 0 moderate of 35 features

| Feature | PSI | Reading |
|---|---:|---|
| dst_age_days | 10.3700 | major |
| src_age_days | 10.2979 | major |
| weekday | 8.6914 | major |
| dst_flow_ratio | 2.8989 | major |
| dst_has_history | 2.4619 | major |
| src_has_history | 2.3669 | major |
| dst_in_degree | 1.8833 | major |
| dst_in_tx | 1.8791 | major |
| src_out_tx | 1.5963 | major |
| src_out_degree | 1.5943 | major |

## val->test: 4 major, 5 moderate of 35 features

| Feature | PSI | Reading |
|---|---:|---|
| weekday | 6.6921 | major |
| dst_age_days | 3.6504 | major |
| src_age_days | 3.4947 | major |
| dst_flow_ratio | 0.2709 | major |
| src_flow_ratio | 0.1941 | moderate |
| src_out_degree | 0.1592 | moderate |
| src_cycle3 | 0.1556 | moderate |
| dst_cycle3 | 0.1209 | moderate |
| src_out_tx | 0.1141 | moderate |
| dst_in_degree | 0.0968 | stable |

## train->val: 22 major, 1 moderate of 35 features

| Feature | PSI | Reading |
|---|---:|---|
| weekday | 8.8374 | major |
| src_age_days | 6.3037 | major |
| dst_age_days | 5.2795 | major |
| dst_flow_ratio | 1.3761 | major |
| dst_has_history | 1.1366 | major |
| src_in_tx | 1.1210 | major |
| src_in_degree | 1.1172 | major |
| dst_in_tx | 1.0277 | major |
| dst_in_degree | 1.0245 | major |
| src_out_degree | 1.0103 | major |
