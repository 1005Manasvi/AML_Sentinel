"""``gnn-experiment``: train the GraphSAGE pair classifier and compare it fairly.

Comparisons, all on held-out **test-period** data:

1. GNN vs. LightGBM on the same account pairs. LightGBM's per-transaction scores
   are aggregated by max over each pair's test-period payments. The comparison
   includes a paired bootstrap of the PR-AUC difference.
2. GNN vs. the NetworkX detectors on the same sampled alerts. The detectors' firing
   set is compared with the GNN's top-n alerts at the same n.

The GNN sees each test pair's whole-period structure (batch, end of period), while
LightGBM scores each payment in real time. That favours the GNN, and the report
states it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aml_sentinel.config import Settings
from aml_sentinel.data.processed import load_processed
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.evaluation.metrics import binary_metrics
from aml_sentinel.evaluation.ranking import (
    bootstrap_ap_difference,
    pr_curve_points,
    precision_at_k,
    ranking_summary,
)
from aml_sentinel.exceptions import DataNotFoundError
from aml_sentinel.features.graph_features import account_codes
from aml_sentinel.integrity import sha256_file
from aml_sentinel.system2.gnn import GnnConfig, TrainedGnn, train_gnn
from aml_sentinel.system2.gnn_data import (
    PeriodGraph,
    build_split_graphs,
    pair_max_scores,
)

logger = logging.getLogger(__name__)
TOP_K: Final = (100, 1_000, 10_000)
BOOTSTRAP: Final = 300


def pair_features(graph: PeriodGraph) -> np.ndarray:
    """Edge features plus both endpoints' node features (the GNN's inputs, flattened).

    Args:
        graph: Period graph.

    Returns:
        Matrix with one row per pair.
    """
    return np.hstack([graph.edge_attr, graph.x[graph.pair_src], graph.x[graph.pair_dst]])


def same_input_lgbm(graphs: dict[str, PeriodGraph], seed: int) -> np.ndarray:
    """Ablation: LightGBM on exactly the GNN's inputs, without message passing.

    Args:
        graphs: Train/val/test period graphs.
        seed: Random seed.

    Returns:
        Test-pair scores.
    """
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "verbosity": -1,
        "seed": seed,
        "num_threads": 4,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "deterministic": True,
        "force_row_wise": True,
    }
    model = lgb.train(
        params,
        lgb.Dataset(pair_features(graphs["train"]), graphs["train"].y),
        num_boost_round=1000,
        valid_sets=[lgb.Dataset(pair_features(graphs["val"]), graphs["val"].y)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return np.asarray(model.predict(pair_features(graphs["test"])), dtype=np.float64)


def alert_pair_scores(
    frame: pd.DataFrame, graph: PeriodGraph, gnn_scores: np.ndarray, tx_ids: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """GNN score of each alert's (payer, payee) pair in a period graph.

    Args:
        frame: Processed transactions.
        graph: Period graph the scores belong to.
        gnn_scores: Score per pair of ``graph``.
        tx_ids: Alert transaction ids.

    Returns:
        ``(scores, found)``. Alerts whose pair is not in the graph score 0.
    """
    pos = pd.Index(frame[Col.TX_ID]).get_indexer(list(tx_ids))
    src, dst, n_nodes = account_codes(frame)
    key = src[pos] * n_nodes + dst[pos]
    pair_key = graph.pair_src * n_nodes + graph.pair_dst
    order = np.argsort(pair_key)
    j = np.searchsorted(pair_key[order], key)
    found = (j < pair_key.size) & (pair_key[order][np.minimum(j, pair_key.size - 1)] == key)
    scores = np.zeros(len(pos))
    scores[found] = gnn_scores[order[j[found]]]
    return scores, found


def compare_with_detectors(
    frame: pd.DataFrame, graph: PeriodGraph, gnn_scores: np.ndarray, evidence: pd.DataFrame
) -> dict[str, Any]:
    """Detector firing set vs. GNN top-n on the same sampled test alerts.

    Args:
        frame: Processed transactions.
        graph: Test-period graph.
        gnn_scores: GNN score per pair of ``graph``.
        evidence: Strict-horizon detector evidence (``tx_id``, ``n_findings``, ``split``).

    Returns:
        Detector and GNN metrics at the same alert count.
    """
    ev = evidence.loc[evidence["split"] == "test"].reset_index(drop=True)
    alert_score, found = alert_pair_scores(frame, graph, gnn_scores, ev["tx_id"].tolist())
    pos = pd.Index(frame[Col.TX_ID]).get_indexer(ev["tx_id"])
    y = frame[Col.IS_LAUNDERING].to_numpy()[pos]
    fired = ev["n_findings"].to_numpy() > 0
    n = int(fired.sum())
    top = np.zeros(len(ev), dtype=bool)
    if n:
        top[np.argsort(-alert_score, kind="stable")[:n]] = True
    return {
        "alerts": len(ev),
        "positives": int(y.sum()),
        "detectors_any": binary_metrics(y, fired).to_dict(),
        "gnn_top_n": binary_metrics(y, top).to_dict(),
        "alerts_without_test_pair": int((~found).sum()),
    }


def gnn_command(settings: Settings, dataset: DatasetName, *, cfg: GnnConfig | None = None) -> Path:
    """Run the GNN experiment and write report, figure and safetensors weights.

    Args:
        settings: Settings.
        dataset: Dataset variant.
        cfg: Hyperparameters.

    Returns:
        Path of the JSON report.

    Raises:
        DataNotFoundError: If System 1a scores are missing.
    """
    cfg = cfg or GnnConfig(seed=settings.random_seed)
    scores_path = settings.processed_dir / f"{dataset.value}.system1a_scores.parquet"
    if not scores_path.is_file():
        raise DataNotFoundError(f"{scores_path.name} missing; run train-tabular first")
    frame = load_processed(settings.processed_dir, dataset)
    graphs = build_split_graphs(frame)
    trained = train_gnn(graphs["train"], graphs["val"], cfg)
    test = graphs["test"]
    gnn = trained.predict(test)

    s1a = pd.read_parquet(scores_path).set_index("tx_id")
    in_test = (frame["split"].astype("string") == "test").to_numpy()
    lgbm = {
        name: pair_max_scores(frame, in_test, s1a[f"p_{name}_raw"], test)
        for name in ("all", "no_shortcuts")
    }

    report: dict[str, Any] = {
        "dataset": dataset.value,
        "setting": "end-of-period batch (GNN) vs real-time per-transaction "
        "(LightGBM, max per pair)",
        "graphs": {
            k: {"pairs": int(g.y.size), "positive_pairs": int(g.y.sum())} for k, g in graphs.items()
        },
        "training": {"best_epoch": trained.best_epoch, "best_val_ap": trained.best_val_ap},
        "pairs": {},
        "significance": {},
    }
    same_input = same_input_lgbm(graphs, settings.random_seed)
    candidates = {
        "gnn": gnn,
        "lgbm_same_pair_inputs": same_input,
        **{f"lgbm_{k}_pairmax": v for k, v in lgbm.items()},
    }
    for name, s in candidates.items():
        report["pairs"][name] = {
            "ranking": ranking_summary(test.y, s),
            "precision_at_k": precision_at_k(test.y, s, TOP_K),
            "pr_curve": pr_curve_points(test.y, s),
        }
    for name, scores in candidates.items():
        if name != "gnn":
            report["significance"][f"gnn vs {name}"] = bootstrap_ap_difference(
                test.y, gnn, scores, n_boot=BOOTSTRAP, seed=settings.random_seed
            )

    evidence_path = settings.processed_dir / f"{dataset.value}.system2_evidence.parquet"
    if evidence_path.is_file():
        report["vs_detectors"] = compare_with_detectors(
            frame, test, gnn, pd.read_parquet(evidence_path)
        )

    _save_weights(trained, settings.models_dir / dataset.value, report)
    out_dir = settings.reports_dir / "system2"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{dataset.value}_gnn.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out_dir / f"{dataset.value}_gnn.md").write_text(
        render_markdown(report), encoding="utf-8", newline="\n"
    )
    _plot(report, out_dir / f"{dataset.value}_gnn_pr.png")
    return json_path


def _save_weights(trained: TrainedGnn, model_dir: Path, report: dict[str, Any]) -> None:
    from safetensors.torch import save_file

    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / "gnn_sage.safetensors"
    save_file(
        {k: v.detach().cpu().contiguous() for k, v in trained.model.state_dict().items()}, str(path)
    )
    report["artifacts"] = {"weights": path.name, "sha256": sha256_file(path)}


def _plot(report: dict[str, Any], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, m in report["pairs"].items():
        ax.plot(m["pr_curve"]["recall"], m["pr_curve"]["precision"], label=name)
    ax.set(
        xlabel="Recall",
        ylabel="Precision",
        yscale="log",
        title=f"Test account pairs: {report['dataset']} (GNN experiment)",
    )
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def render_markdown(report: dict[str, Any]) -> str:
    """Render the GNN experiment report.

    Args:
        report: Output of :func:`gnn_command`.

    Returns:
        Markdown text.
    """
    g = report["graphs"]["test"]
    lines = [
        f"# GNN experiment (GraphSAGE): {report['dataset']}",
        "",
        "**Experiment, not production.** Unit = account pair in the test period "
        f"({g['pairs']:,} pairs, {g['positive_pairs']:,} laundering). Setting: "
        f"{report['setting']}. That favours the GNN.",
        "",
        "| Model | PR-AUC | ROC-AUC | P@100 | P@1,000 | P@10,000 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, m in report["pairs"].items():
        pk = m["precision_at_k"]
        lines.append(
            f"| {name} | {m['ranking']['average_precision']:.4f} | {m['ranking']['roc_auc']:.4f} "
            f"| {_pct(pk['100']['precision'])} | {_pct(pk['1000']['precision'])} "
            f"| {_pct(pk['10000']['precision'])} |"
        )
    lines += ["", "| Comparison (paired bootstrap) | PR-AUC diff | 95% CI |", "|---|---:|---|"]
    for name, b in report["significance"].items():
        lines.append(f"| {name} | {b['diff']:+.4f} | [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] |")
    if "vs_detectors" in report:
        v = report["vs_detectors"]
        d, n = v["detectors_any"], v["gnn_top_n"]
        lines += [
            "",
            f"## Against the NetworkX detectors (same {v['alerts']:,} sampled test alerts, "
            f"{v['positives']} laundering)",
            "",
            "| Method | Alerts flagged | Precision | Recall |",
            "|---|---:|---:|---:|",
            f"| detectors (any finding, strict) | {d['alerts']:,} | {_pct(d['precision'])} "
            f"| {_pct(d['recall'])} |",
            f"| GNN top-n (same n) | {n['alerts']:,} | {_pct(n['precision'])} "
            f"| {_pct(n['recall'])} |",
        ]
    return "\n".join(lines) + "\n"
