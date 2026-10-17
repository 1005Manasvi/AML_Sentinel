# Final ranking: HI-Small

Validation-fitted stacker over `p__fused[lgbm_all+laya[head_only]]` plus System 2 evidence; 4,000 test alerts (271 laundering).

| Score | PR-AUC | P@100 | ECE | Escalate 5%: precision | recall |
|---|---:|---:|---:|---:|---:|
| system1_only | 0.674 | 96.0% | 0.0214 | 64.4% | 62.0% |
| final | 0.651 | 93.0% | 0.0272 | 57.9% | 66.1% |

Graph-evidence gain (paired bootstrap, PR-AUC): -0.0229 [-0.0498, +0.0019].

Coefficients (standardised): logit_p_triage=+2.05, ev_n_findings=+0.72, ev_max_scatter_gather=+0.10, ev_max_fan_out=-0.41, ev_max_gather_scatter=+0.12, ev_max_circular=-0.07, ev_max_fan_in=-0.81, ev_max_structuring=+0.30
