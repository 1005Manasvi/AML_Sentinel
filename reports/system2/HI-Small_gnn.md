# GNN experiment (GraphSAGE): HI-Small

**Experiment, not production.** Unit = account pair in the test period (375,225 pairs, 926 laundering). Setting: end-of-period batch (GNN) vs real-time per-transaction (LightGBM, max per pair). That favours the GNN.

| Model | PR-AUC | ROC-AUC | P@100 | P@1,000 | P@10,000 |
|---|---:|---:|---:|---:|---:|
| gnn | 0.0421 | 0.8440 | 24.0% | 9.3% | 2.6% |
| lgbm_same_pair_inputs | 0.0436 | 0.9135 | 10.0% | 8.5% | 3.5% |
| lgbm_all_pairmax | 0.4142 | 0.9582 | 99.0% | 39.2% | 6.0% |
| lgbm_no_shortcuts_pairmax | 0.3684 | 0.9164 | 99.0% | 35.3% | 5.5% |

| Comparison (paired bootstrap) | PR-AUC diff | 95% CI |
|---|---:|---|
| gnn vs lgbm_same_pair_inputs | -0.0015 | [-0.0129, +0.0110] |
| gnn vs lgbm_all_pairmax | -0.3721 | [-0.4008, -0.3393] |
| gnn vs lgbm_no_shortcuts_pairmax | -0.3263 | [-0.3535, -0.2897] |

## Against the NetworkX detectors (same 4,000 sampled test alerts, 271 laundering)

| Method | Alerts flagged | Precision | Recall |
|---|---:|---:|---:|
| detectors (any finding, strict) | 1,156 | 12.1% | 51.7% |
| GNN top-n (same n) | 1,156 | 17.6% | 75.3% |
