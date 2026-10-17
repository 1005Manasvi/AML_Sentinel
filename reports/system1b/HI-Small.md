# System 1b (Laya, fusion): HI-Small

Alert queue = top 1% of each split by LightGBM[all]. Text models score a seeded sample: val 4,000 of 9,655 alerts (290 laundering), test 4,000 of 8,628 (271 laundering). **All metrics are within the alert queue**, so they are not comparable with full-split metrics.

Jev: not run (JEV_ENABLED=false or no key).

| Model | PR-AUC | P@100 | P@500 | ECE | Escalate 5%: precision | recall | Typology acc. | Typology macro-F1 | ms/alert |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| lgbm_all | 0.649 | 95.0% | 39.0% | 0.0231 | 63.5% | 62.4% | n/a | n/a | n/a |
| lgbm_no_shortcuts | 0.591 | 94.0% | 34.8% | 0.0143 | 65.9% | 53.5% | n/a | n/a | n/a |
| laya[zero_shot] | 0.091 | 7.0% | 9.8% | 0.4577 | 7.1% | 4.4% | 94.2% | 0.243 | 448.5 |
| laya[head_only] | 0.096 | 15.0% | 11.6% | 0.4093 | 12.5% | 6.6% | 66.1% | 0.213 | 303.5 |
| fused[lgbm_all] | 0.645 | 95.0% | 39.0% | 0.0228 | 63.5% | 62.4% | n/a | n/a | n/a |
| fused[lgbm_no_shortcuts] | 0.591 | 94.0% | 34.8% | 0.0112 | 65.9% | 53.5% | n/a | n/a | n/a |
| fused[lgbm_all+laya[zero_shot]] | 0.670 | 96.0% | 39.2% | 0.0190 | 64.9% | 62.0% | n/a | n/a | n/a |
| fused[lgbm_all+laya[head_only]] | 0.674 | 96.0% | 39.2% | 0.0214 | 64.4% | 62.0% | n/a | n/a | n/a |
| fused[lgbm_no_shortcuts+laya[zero_shot]] | 0.593 | 93.0% | 35.4% | 0.0124 | 62.9% | 56.8% | n/a | n/a | n/a |
| fused[lgbm_no_shortcuts+laya[head_only]] | 0.613 | 91.0% | 36.2% | 0.0116 | 62.4% | 55.7% | n/a | n/a | n/a |

## Does adding the text model help? (paired bootstrap, test alerts)

| Comparison | PR-AUC gain | 95% CI | P(no gain) |
|---|---:|---|---:|
| fused[lgbm_all+laya[zero_shot]] vs fused[lgbm_all] | +0.0253 | [+0.0155, +0.0359] | 0.000 |
| fused[lgbm_all+laya[head_only]] vs fused[lgbm_all] | +0.0293 | [+0.0207, +0.0374] | 0.000 |
| fused[lgbm_no_shortcuts+laya[zero_shot]] vs fused[lgbm_no_shortcuts] | +0.0024 | [-0.0110, +0.0152] | 0.355 |
| fused[lgbm_no_shortcuts+laya[head_only]] vs fused[lgbm_no_shortcuts] | +0.0217 | [+0.0096, +0.0349] | 0.001 |

Row resampling only: fusion-fit variance is not included, so the intervals are optimistic.

## Fusion coefficients (standardised inputs)

- **fused[lgbm_all]** (isotonic): logit_p_tabular=+2.28
- **fused[lgbm_no_shortcuts]** (platt): logit_p_tabular=+1.92
- **fused[lgbm_all+laya[zero_shot]]** (platt): logit_p_tabular=+2.16, logit_p_escalate=+0.34, risk_expected=+0.26, logit_p_benign=+0.04
- **fused[lgbm_all+laya[head_only]]** (platt): logit_p_tabular=+2.30, logit_p_escalate=-0.04, risk_expected=-0.09, logit_p_benign=+0.21
- **fused[lgbm_no_shortcuts+laya[zero_shot]]** (isotonic): logit_p_tabular=+1.81, logit_p_escalate=+0.41, risk_expected=+0.29, logit_p_benign=-0.06
- **fused[lgbm_no_shortcuts+laya[head_only]]** (platt): logit_p_tabular=+2.02, logit_p_escalate=-0.18, risk_expected=-0.15, logit_p_benign=+0.26
