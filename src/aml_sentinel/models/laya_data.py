"""Build the Laya fine-tuning set: synthetic alert summaries with typed targets.

Sources (all synthetic, all from **training-period** rows only):

* IBM AMLworld train split: every laundering row, plus random benign rows and
  *hard-negative* benign rows that look suspicious (large, fast, cyclic or
  near-threshold), so the model cannot learn "anything unusual is laundering".
* This project's generator: STRUCTURING attempts (absent from IBM) and the benign
  near-threshold cash look-alikes (ADR-012).

Targets per row, in Laya's ``expected`` format (verified in ``laya/train.py``): the
choice label string, the score level index, and a boolean for ``noul``. Undocumented
positives omit the typology target, because their pattern is unknown.

Risk levels are a documented convention, not ground truth: easy benign = 1, hard
benign = 2, undocumented laundering = 4, documented laundering = 5.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from aml_sentinel.data.fx import add_usd_amounts, estimate_usd_rates
from aml_sentinel.data.loader import load_transactions
from aml_sentinel.data.patterns import TYPOLOGY, Typology, attach_typology, load_patterns
from aml_sentinel.data.schema import Col
from aml_sentinel.data.synthetic import SyntheticConfig, generate, write_dataset
from aml_sentinel.features.builder import build_features
from aml_sentinel.models.decisions import QUESTIONS
from aml_sentinel.models.summarizer import Noise, summarize
from aml_sentinel.models.typology import LayaLabel, laya_label

ROLE_EASY: Final = "benign_easy"
ROLE_HARD: Final = "benign_hard"
ROLE_POS_DOC: Final = "laundering_documented"
ROLE_POS_UNDOC: Final = "laundering_undocumented"
RISK_LEVEL: Final = {ROLE_EASY: 0, ROLE_HARD: 1, ROLE_POS_UNDOC: 3, ROLE_POS_DOC: 4}


@dataclass(frozen=True, slots=True)
class SetSizes:
    """How many rows of each kind to sample.

    Attributes:
        max_positives: Cap on IBM laundering rows.
        benign_easy: Random benign rows.
        benign_hard: Suspicious-looking benign rows.
        structuring: Generator STRUCTURING rows.
        structuring_lookalikes: Generator benign near-threshold cash rows.
        noisy_copies: Extra augmented renderings per row (0 = canonical only).
    """

    max_positives: int = 3000
    benign_easy: int = 2500
    benign_hard: int = 2500
    structuring: int = 400
    structuring_lookalikes: int = 400
    noisy_copies: int = 1


def hard_negative_mask(features: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """Benign-candidate rows that *look* suspicious (labels are not used here).

    Args:
        features: Feature frame.

    Returns:
        Boolean mask.
    """
    usd = np.expm1(features["amount_usd_log"].to_numpy())
    return np.asarray(
        (usd >= 5_000)
        | (features["src_tx_24h"].to_numpy() >= 10)
        | (features["dst_rx_24h"].to_numpy() >= 5)
        | (features["src_cycle3"].to_numpy() > 0)
        | (features["dst_cycle3"].to_numpy() > 0),
        dtype=bool,
    )


def role_of(is_laundering: bool, typology: Any, hard: bool) -> str:
    """Assign a sampling role.

    Args:
        is_laundering: Ground truth.
        typology: Documented typology or missing.
        hard: Whether the row is a hard negative candidate.

    Returns:
        One of the ``ROLE_*`` constants.
    """
    if is_laundering:
        return ROLE_POS_DOC if isinstance(typology, str) and typology else ROLE_POS_UNDOC
    return ROLE_HARD if hard else ROLE_EASY


def expected_answers(role: str, label: LayaLabel | None) -> dict[str, Any]:
    """Laya ``expected`` targets for one row.

    Args:
        role: Sampling role.
        label: Typology label, or ``None`` if unknown.

    Returns:
        ``{"escalate": bool, "risk": int, "typology": str?}``.
    """
    out: dict[str, Any] = {
        "escalate": role in (ROLE_POS_DOC, ROLE_POS_UNDOC),
        "risk": RISK_LEVEL[role],
    }
    if label is not None:
        out["typology"] = label.value
    return out


def sample_ibm_rows(
    frame: pd.DataFrame, features: pd.DataFrame, sizes: SetSizes, rng: np.random.Generator
) -> pd.DataFrame:
    """Pick training-split rows and their roles.

    Args:
        frame: Processed transactions (with ``split`` and ``typology``).
        features: Feature frame aligned with ``frame``.
        sizes: Sample sizes.
        rng: Random generator.

    Returns:
        ``DataFrame`` with ``row`` (positional index), ``role`` and ``typology``.
    """
    train = (frame["split"].astype("string") == "train").to_numpy()
    y = frame[Col.IS_LAUNDERING].to_numpy() == 1
    hard = hard_negative_mask(features)

    def pick(mask: npt.NDArray[np.bool_], n: int) -> npt.NDArray[np.int64]:
        idx = np.flatnonzero(mask)
        return np.sort(rng.choice(idx, size=min(n, idx.size), replace=False))

    chosen = np.concatenate(
        [
            pick(train & y, sizes.max_positives),
            pick(train & ~y & hard, sizes.benign_hard),
            pick(train & ~y & ~hard, sizes.benign_easy),
        ]
    )
    typ = frame[TYPOLOGY].astype("string").to_numpy()[chosen]
    roles = [
        role_of(bool(y[i]), t if isinstance(t, str) else None, bool(hard[i]))
        for i, t in zip(chosen, typ, strict=True)
    ]
    return pd.DataFrame({"row": chosen, "role": roles, "typology": typ})


def generator_structuring(sizes: SetSizes, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate STRUCTURING positives and near-threshold benign look-alikes.

    Args:
        sizes: Desired counts.
        seed: Generator seed (kept distinct from the bundled sample's).

    Returns:
        ``(features, meta)`` where meta has ``role`` and ``typology``.
    """
    cfg = SyntheticConfig(
        seed=seed,
        attempts_per_typology=max(4, sizes.structuring // 8),
        lookalikes_per_kind=max(4, sizes.structuring_lookalikes // 4),
    )
    with tempfile.TemporaryDirectory() as tmp:
        write_dataset(generate(cfg), Path(tmp), name="Gen")
        frame = load_transactions(Path(tmp) / "Gen_Trans.csv")
        frame = add_usd_amounts(frame, estimate_usd_rates(frame))
        frame = attach_typology(frame, load_patterns(Path(tmp) / "Gen_Patterns.txt"))
    feats = build_features(frame)
    typ = frame[TYPOLOGY].astype("string")
    usd = frame[Col.AMOUNT_PAID_USD].to_numpy()
    is_struct = (typ == Typology.STRUCTURING.value).fillna(False).to_numpy()
    lookalike = (
        (frame[Col.IS_LAUNDERING] == 0).to_numpy()
        & (frame[Col.PAYMENT_FORMAT].astype("string") == "Cash").to_numpy()
        & (usd >= 5_000)
        & (usd < 10_000)
    )
    rows = np.concatenate(
        [
            np.flatnonzero(is_struct)[: sizes.structuring],
            np.flatnonzero(lookalike)[: sizes.structuring_lookalikes],
        ]
    )
    meta = pd.DataFrame(
        {
            "role": [ROLE_POS_DOC if is_struct[i] else ROLE_HARD for i in rows],
            "typology": [Typology.STRUCTURING.value if is_struct[i] else None for i in rows],
        }
    )
    return feats.iloc[rows].reset_index(drop=True), meta


def iter_jsonl_rows(
    features: pd.DataFrame,
    meta: pd.DataFrame,
    *,
    noisy_copies: int,
    rng: np.random.Generator,
    source: str,
) -> Iterator[dict[str, Any]]:
    """Render summaries and targets as Laya training rows.

    Args:
        features: Feature rows aligned with ``meta``.
        meta: ``role`` and ``typology`` per row.
        noisy_copies: Augmented renderings per row in addition to the canonical one.
        rng: Random generator.
        source: Provenance tag stored with each row.

    Yields:
        ``{"state", "questions", "expected", "meta"}`` rows.
    """
    noise = Noise()
    for i in range(len(meta)):
        role, typ = meta["role"].iloc[i], meta["typology"].iloc[i]
        label = laya_label(
            role in (ROLE_POS_DOC, ROLE_POS_UNDOC), typ if isinstance(typ, str) else None
        )
        expected = expected_answers(role, label)
        row = {str(k): float(v) for k, v in features.iloc[i].items()}
        renderings = [summarize(row)] + [
            summarize(row, noise=noise, rng=rng) for _ in range(noisy_copies)
        ]
        for text in renderings:
            yield {
                "state": text,
                "questions": QUESTIONS,
                "expected": expected,
                "meta": {"role": role, "source": source},
            }


def write_jsonl(rows: Iterator[dict[str, Any]], path: Path) -> dict[str, int]:
    """Write rows to JSONL and count them by role.

    Args:
        rows: Rows from :func:`iter_jsonl_rows`.
        path: Output file.

    Returns:
        Row counts per role, plus ``total``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
            role = row["meta"]["role"]
            counts[role] = counts.get(role, 0) + 1
    counts["total"] = sum(counts.values())
    return counts
