# GNN experiment (GraphSAGE): Sample

**Experiment, not production.** Unit = account pair in the test period (1,027 pairs, 8 laundering). Setting: end-of-period batch (GNN) vs real-time per-transaction (LightGBM, max per pair). That favours the GNN.

| Model | PR-AUC | ROC-AUC | P@100 | P@1,000 | P@10,000 |
|---|---:|---:|---:|---:|---:|
| gnn | 0.0340 | 0.7426 | 2.0% | 0.8% | 0.8% |
| lgbm_all_pairmax | 0.1884 | 0.9540 | 7.0% | 0.8% | 0.8% |
| lgbm_no_shortcuts_pairmax | 0.0961 | 0.9345 | 7.0% | 0.8% | 0.8% |

| Comparison (paired bootstrap) | PR-AUC diff | 95% CI |
|---|---:|---|
| gnn vs lgbm_all_pairmax | -0.1544 | [-0.4571, +0.0173] |
| gnn vs lgbm_no_shortcuts_pairmax | -0.0620 | [-0.2735, +0.0901] |

## Against the NetworkX detectors (same 633 sampled test alerts, 211 laundering)

| Method | Alerts flagged | Precision | Recall |
|---|---:|---:|---:|
| detectors (any finding, strict) | 495 | 30.7% | 72.0% |
| GNN top-n (same n) | 495 | 34.7% | 81.5% |
