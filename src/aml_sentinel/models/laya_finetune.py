"""Build the Laya training set and fine-tune Laya through its own ``laya.train`` API.

Two variants (ADR-013):

* ``head_only``: ``TrainConfig(freeze_encoder=True)`` runs locally on a 4 GB GPU.
  The 421M-parameter encoder stays frozen and only the decision head trains.
* ``full``: every weight trains. That needs about 16 GB of GPU memory, so it runs
  on Kaggle via ``scripts/kaggle_laya_full_finetune.py`` and the checkpoint is then
  copied into ``models/laya/full``.

Both start from the base checkpoint pinned and verified in ``models/laya/base``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Final

import numpy as np

from aml_sentinel.config import Settings
from aml_sentinel.data.processed import load_processed
from aml_sentinel.data.schema import DatasetName
from aml_sentinel.exceptions import ConfigurationError
from aml_sentinel.features.builder import build_features
from aml_sentinel.models.laya_data import (
    SetSizes,
    generator_structuring,
    iter_jsonl_rows,
    sample_ibm_rows,
    write_jsonl,
)
from aml_sentinel.models.laya_triage import download_base, verified_digests, write_manifest

logger = logging.getLogger(__name__)

LAYA_DIR: Final = "laya"


def laya_paths(settings: Settings, dataset: DatasetName) -> dict[str, Path]:
    """Canonical locations of Laya data and checkpoints.

    Args:
        settings: Settings.
        dataset: Dataset whose training split feeds Laya.

    Returns:
        Paths ``train_jsonl``, ``base``, ``head_only`` and ``full``.
    """
    root = settings.models_dir / LAYA_DIR
    return {
        "train_jsonl": settings.processed_dir / LAYA_DIR / f"{dataset.value}.train.jsonl",
        "base": root / "base",
        "head_only": root / f"{dataset.value}-head_only",
        "full": root / f"{dataset.value}-full",
    }


def build_laya_dataset(
    settings: Settings, dataset: DatasetName, *, sizes: SetSizes | None = None
) -> dict[str, Any]:
    """Write the fine-tuning JSONL from the training split plus generator structuring.

    Args:
        settings: Settings.
        dataset: Source dataset.
        sizes: Sample sizes.

    Returns:
        Row counts per role and the output path.
    """
    sizes = sizes or SetSizes()
    rng = np.random.default_rng(settings.random_seed)
    frame = load_processed(settings.processed_dir, dataset)
    features = build_features(frame)
    picked = sample_ibm_rows(frame, features, sizes, rng)
    ibm_feats = features.iloc[picked["row"].to_numpy()].reset_index(drop=True)
    # A seed distinct from the bundled sample, so Laya never trains on demo rows.
    gen_feats, gen_meta = generator_structuring(sizes, seed=settings.random_seed + 1000)

    def rows() -> Any:
        yield from iter_jsonl_rows(
            ibm_feats,
            picked[["role", "typology"]],
            noisy_copies=sizes.noisy_copies,
            rng=rng,
            source="ibm-train",
        )
        yield from iter_jsonl_rows(
            gen_feats, gen_meta, noisy_copies=sizes.noisy_copies, rng=rng, source="generator"
        )

    path = laya_paths(settings, dataset)["train_jsonl"]
    counts = write_jsonl(rows(), path)
    stats = {"path": path.as_posix(), "counts": counts}
    path.with_suffix(".stats.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    logger.info("laya dataset written", extra={"rows": counts["total"]})
    return stats


def finetune_head_only(
    settings: Settings,
    dataset: DatasetName,
    *,
    epochs: int = 3,
    micro_batch: int = 8,
    grad_accum: int = 4,
) -> Path:
    """Fine-tune Laya's decision head with the encoder frozen.

    Args:
        settings: Settings.
        dataset: Dataset whose JSONL was built by :func:`build_laya_dataset`.
        epochs: Training epochs.
        micro_batch: Items per forward pass (bounded by GPU memory).
        grad_accum: Gradient accumulation steps.

    Returns:
        The output checkpoint directory (with a fresh manifest).

    Raises:
        ConfigurationError: If ``laya`` is not installed or the dataset is missing.
    """
    paths = laya_paths(settings, dataset)
    if not paths["train_jsonl"].is_file():
        raise ConfigurationError("Laya training data missing; run build-laya-data first")
    if not (paths["base"] / "manifest.json").is_file():
        download_base(paths["base"])
    verified_digests(paths["base"])
    try:
        import torch
        from laya.train import TrainConfig, finetune
    except ImportError:
        raise ConfigurationError(
            "laya is not installed; pip install -r requirements/laya.txt"
        ) from None

    config = TrainConfig(
        freeze_encoder=True,
        epochs=epochs,
        micro_batch=micro_batch,
        grad_accum=grad_accum,
        seed=settings.random_seed,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = paths["head_only"]
    summary = finetune(
        str(paths["train_jsonl"]), str(paths["base"]), str(out), config, device=device
    )
    base_manifest = json.loads((paths["base"] / "manifest.json").read_text(encoding="utf-8"))
    write_manifest(out, source="finetune:head_only", revision=base_manifest.get("revision"))
    (out / "aml_train_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
    )
    logger.info("laya head-only fine-tune done", extra={"device": device})
    return out
