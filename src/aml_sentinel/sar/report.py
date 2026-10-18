"""``draft-sar``: draft SARs for the highest-risk test alerts and audit every generation.

Uses the final calibrated score when available (``rank``), otherwise System 1b's
fused score. The external drafter runs only when ``LLM_PROVIDER=external``;
otherwise everything stays local (template). Every generation appends an audit
record: who, what, when and which case, plus non-sensitive metadata only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Final

import pandas as pd

from aml_sentinel.audit import AuditLog
from aml_sentinel.config import Settings
from aml_sentinel.data.processed import load_processed
from aml_sentinel.data.schema import DatasetName
from aml_sentinel.exceptions import DataNotFoundError
from aml_sentinel.sar.drafting import draft_sar
from aml_sentinel.sar.facts import build_case_facts
from aml_sentinel.sar.llm import AnthropicClient, LlmClient
from aml_sentinel.system2.cases import CaseIndex
from aml_sentinel.system2.detectors import run_detectors

logger = logging.getLogger(__name__)
SYSTEM_ACTOR: Final = "system.sar_drafter"


def audit_log(settings: Settings) -> AuditLog:
    """The project audit log (``<DATA_DIR>/audit/audit.jsonl``, git-ignored).

    Args:
        settings: Settings.

    Returns:
        The audit log.
    """
    return AuditLog(settings.audit_path)


def _top_alerts(settings: Settings, dataset: DatasetName, top: int) -> pd.DataFrame:
    final = settings.processed_dir / f"{dataset.value}.final_scores.parquet"
    if final.is_file():
        scores = pd.read_parquet(final)[["tx_id", "p_final"]].rename(columns={"p_final": "risk"})
    else:
        s1b = settings.processed_dir / f"{dataset.value}.system1b_scores.parquet"
        if not s1b.is_file():
            raise DataNotFoundError("no alert scores; run triage-text (and rank) first")
        table = pd.read_parquet(s1b)
        if "split" in table.columns:
            table = table.loc[table["split"] == "test"]
        col = next(c for c in table.columns if c.startswith("p__fused[lgbm_all"))
        scores = table[["tx_id", col]].rename(columns={col: "risk"})
    return scores.nlargest(top, "risk")


def draft_command(
    settings: Settings,
    dataset: DatasetName,
    *,
    top: int = 5,
    client: LlmClient | None = None,
    actor: str = SYSTEM_ACTOR,
) -> Path:
    """Draft SARs for the ``top`` highest-risk test alerts.

    Args:
        settings: Settings.
        dataset: Dataset variant.
        top: Number of cases to draft.
        client: External drafter override (tests); by default chosen from settings.
        actor: Audit-log actor id.

    Returns:
        The output directory.
    """
    if client is None and settings.llm_provider == "external":
        client = AnthropicClient.from_settings(settings)
    alerts = _top_alerts(settings, dataset, top)
    frame = load_processed(settings.processed_dir, dataset)
    index = CaseIndex(frame)
    out_dir = settings.reports_dir / "sar" / dataset.value
    out_dir.mkdir(parents=True, exist_ok=True)
    log = audit_log(settings)
    rows = []
    for tx_id, risk in zip(alerts["tx_id"], alerts["risk"], strict=True):
        case = index.build(int(tx_id))
        draft = draft_sar(
            build_case_facts(case, run_detectors(case), risk_score=float(risk)), client
        )
        (out_dir / f"{draft.case_id}.md").write_text(draft.text, encoding="utf-8", newline="\n")
        (out_dir / f"{draft.case_id}.json").write_text(
            json.dumps(draft.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        log.append(
            actor=actor,
            action="sar.generated",
            case_id=draft.case_id,
            details={
                "source": draft.source[:64],
                "grounded": draft.is_grounded,
                "n_issues": len(draft.grounding_issues),
            },
        )
        rows.append(
            f"| {draft.case_id} | {risk:.3f} | {draft.source} | "
            f"{'yes' if draft.is_grounded else 'NO'} |"
        )
    (out_dir / "README.md").write_text(
        "# SAR drafts (synthetic data)\n\nEvery draft is stamped "
        "*DRAFT - requires analyst review*.\n\n| Case | Risk | Drafted by | Grounded |\n"
        "|---|---:|---|---|\n" + "\n".join(rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    logger.info("sar drafts written", extra={"dataset": dataset.value, "cases": len(rows)})
    return out_dir
