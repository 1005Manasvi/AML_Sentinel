"""Run System 1b end to end and write its report, figures and fusion models."""

from __future__ import annotations

import functools
import gc
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from aml_sentinel.config import Settings
from aml_sentinel.data.processed import load_processed
from aml_sentinel.data.schema import SYNTHETIC_DATASETS, DatasetName
from aml_sentinel.exceptions import DataNotFoundError
from aml_sentinel.features.builder import build_features
from aml_sentinel.integrity import sha256_file
from aml_sentinel.models.jev_triage import JevTriage
from aml_sentinel.models.laya_finetune import laya_paths
from aml_sentinel.models.laya_triage import LayaTriage, load_agent
from aml_sentinel.models.system1b import (
    build_alert_sets,
    evaluate_system1b,
    run_text_model,
)
from aml_sentinel.models.triage import Triage, TriageOutput

logger = logging.getLogger(__name__)
ModelFactory = Callable[[], Triage]


def _pct(v: float | None, d: int = 1) -> str:
    return "n/a" if v is None else f"{100 * v:.{d}f}%"


def _num(v: float | None, d: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{d}f}"


def render_markdown(report: dict[str, Any], dataset: str) -> str:
    """Render System 1b headline tables.

    Args:
        report: System 1b report.
        dataset: Dataset name.

    Returns:
        Markdown text.
    """
    a = report["alerts"]
    lines = [
        f"# System 1b (Laya, fusion): {dataset}",
        "",
        f"Alert queue = top 1% of each split by LightGBM[all]. Text models score a seeded "
        f"sample: val {a['val']['sampled']:,} of {a['val']['queue']:,} alerts "
        f"({a['val']['positives']} laundering), test {a['test']['sampled']:,} of "
        f"{a['test']['queue']:,} ({a['test']['positives']} laundering). **All metrics are "
        "within the alert queue**, so they are not comparable with full-split metrics.",
        "",
        f"Jev: {report['jev_status']}.",
        "",
        "| Model | PR-AUC | P@100 | P@500 | ECE | Escalate 5%: precision | recall | "
        "Typology acc. | Typology macro-F1 | ms/alert |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = {**report["models"], **report["fusion"]}
    for name, m in rows.items():
        typ = m.get("typology", {})
        esc = m["escalation"]["test"]
        lines.append(
            f"| {name} | {_num(m['ranking']['average_precision'])} "
            f"| {_pct(m['precision_at_k']['100']['precision'])} "
            f"| {_pct(m['precision_at_k']['500']['precision'])} "
            f"| {_num(m['ece_quantile'], 4)} | {_pct(esc['precision'])} | {_pct(esc['recall'])} "
            f"| {_pct(typ.get('accuracy'))} | {_num(typ.get('macro_f1'))} "
            f"| {_num(m.get('latency_ms_per_alert'), 1)} |"
        )
    lines += [
        "",
        "## Does adding the text model help? (paired bootstrap, test alerts)",
        "",
        "| Comparison | PR-AUC gain | 95% CI | P(no gain) |",
        "|---|---:|---|---:|",
    ]
    for name, b in report.get("significance", {}).items():
        lines.append(
            f"| {name} | {b['diff']:+.4f} | [{b['ci_low']:+.4f}, {b['ci_high']:+.4f}] "
            f"| {b['p_not_better']:.3f} |"
        )
    lines += [
        "",
        "Row resampling only: fusion-fit variance is not included, so the intervals "
        "are optimistic.",
        "",
        "## Fusion coefficients (standardised inputs)",
        "",
    ]
    for name, m in report["fusion"].items():
        coefs = ", ".join(f"{k}={v:+.2f}" for k, v in m["coefficients"].items())
        lines.append(f"- **{name}** ({m['calibrator']}): {coefs}")
    return "\n".join(lines) + "\n"


def plot_reliability(report: dict[str, Any], path: Path, dataset: str) -> None:
    """Reliability diagram for text-model escalation probabilities and fused scores.

    Args:
        report: System 1b report.
        path: Output PNG.
        dataset: Dataset name.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, m in {**report["models"], **report["fusion"]}.items():
        table = m.get("reliability_test")
        if not table:
            continue
        ax.plot(
            [r["mean_pred"] for r in table], [r["frac_pos"] for r in table], "o-", label=name, ms=3
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set(
        xlabel="Mean predicted probability (test alerts, equal-mass bins)",
        ylabel="Observed laundering rate",
        title=f"System 1b reliability: {dataset}",
    )
    ax.legend(fontsize=6)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def default_text_models(settings: Settings, dataset: DatasetName) -> dict[str, ModelFactory]:
    """Laya variants with an existing verified checkpoint, plus Jev only if enabled.

    Args:
        settings: Settings.
        dataset: Dataset (selects fine-tuned checkpoints).

    Returns:
        ``{name: factory}``; factories load lazily so only one model holds GPU
        memory at a time.
    """
    paths = laya_paths(settings, dataset)
    factories: dict[str, ModelFactory] = {}
    for variant, key in (("zero_shot", "base"), ("head_only", "head_only"), ("full", "full")):
        ckpt = paths[key]
        if (ckpt / "manifest.json").is_file():
            factories[f"laya[{variant}]"] = functools.partial(_laya, ckpt, variant)
    if settings.jev_enabled:
        factories["jev"] = functools.partial(JevTriage.from_settings, settings)
    return factories


def _laya(checkpoint: Path, variant: str) -> Triage:
    return LayaTriage(load_agent(checkpoint), variant=variant)


def system1b_command(
    settings: Settings,
    dataset: DatasetName,
    *,
    text_models: dict[str, ModelFactory] | None = None,
) -> Path:
    """Run System 1b and write report, figure, scores and fusion models.

    Args:
        settings: Settings.
        dataset: Dataset variant.
        text_models: Model factories (defaults to :func:`default_text_models`).

    Returns:
        Path of the JSON report.

    Raises:
        DataNotFoundError: If System 1a scores are missing.
    """
    scores_path = settings.processed_dir / f"{dataset.value}.system1a_scores.parquet"
    if not scores_path.is_file():
        raise DataNotFoundError(f"{scores_path.name} missing; run train-tabular first")
    frame = load_processed(settings.processed_dir, dataset)
    features = build_features(frame)
    alert_sets = build_alert_sets(frame, pd.read_parquet(scores_path), seed=settings.random_seed)
    factories = default_text_models(settings, dataset) if text_models is None else text_models

    outputs: dict[str, dict[str, TriageOutput]] = {}
    latency: dict[str, float] = {}
    for name, factory in factories.items():
        model = factory()
        outputs[name] = {}
        for split, alerts in alert_sets.items():
            out, ms = run_text_model(
                model, features, alerts, is_synthetic=dataset in SYNTHETIC_DATASETS
            )
            outputs[name][split] = out
            if split == "test":
                latency[name] = ms
        del model
        gc.collect()
        _free_gpu()
        logger.info("text model scored", extra={"model": name})

    report, test_scores, fusions = evaluate_system1b(alert_sets, outputs, latency)
    report = {
        "dataset": dataset.value,
        "jev_status": "benchmarked (synthetic only)"
        if "jev" in outputs
        else "not run (JEV_ENABLED=false or no key)",
        **report,
    }

    model_dir = settings.models_dir / dataset.value
    model_dir.mkdir(parents=True, exist_ok=True)
    for combo, fusion in fusions.items():
        path = model_dir / f"fusion_{combo.replace('+', '__')}.json"
        path.write_text(json.dumps(fusion.to_dict()) + "\n", encoding="utf-8", newline="\n")
        report["fusion"][f"fused[{combo}]"]["sha256"] = sha256_file(path)

    parts = []
    for split, key in (("val", "p_val"), ("test", "p")):
        alerts = alert_sets[split]
        part = pd.DataFrame(
            {"tx_id": frame["tx_id"].to_numpy()[alerts.rows], "y": alerts.y, "split": split}
        )
        for name, cols in test_scores.items():
            part[f"p__{name}"] = cols[key]
        parts.append(part)
    table = pd.concat(parts, ignore_index=True)
    table.to_parquet(
        settings.processed_dir / f"{dataset.value}.system1b_scores.parquet", index=False
    )

    out_dir = settings.reports_dir / "system1b"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{dataset.value}.json"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    (out_dir / f"{dataset.value}.md").write_text(
        render_markdown(report, dataset.value), encoding="utf-8", newline="\n"
    )
    plot_reliability(report, out_dir / f"{dataset.value}_reliability.png", dataset.value)
    return json_path


def _free_gpu() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
