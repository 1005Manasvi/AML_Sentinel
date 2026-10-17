"""Laya triage adapter: local, open-source (Apache-2.0) typed-decision model.

The API used here was verified against the installed ``laya`` package (0.4.9) and its
README. It uses ``laya.load(path, device=..., expected_sha256=...)`` and
``agent.predict_batch(states, questions, batch_size=...)``. The latter returns one
dict per state, with ``answers`` keyed by question id and ``usage["truncated"]``.

Security:
    Checkpoints load only from a local directory with a ``manifest.json`` written when
    the checkpoint was downloaded or trained. Every listed file is SHA-256-verified by
    this module, and the digests are also passed to Laya's own ``expected_sha256`` check,
    which runs before any weight is parsed. Weights are safetensors, never pickle.

``laya`` and ``torch`` are optional dependencies (``requirements/laya.txt``) and are
imported lazily, so the core pipeline and CI run without them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from aml_sentinel.exceptions import (
    ConfigurationError,
    DataNotFoundError,
    DataValidationError,
    ModelOutputError,
)
from aml_sentinel.integrity import sha256_file, verify_checksum
from aml_sentinel.models.decisions import QUESTIONS, TYPOLOGY_LABELS, parse_answers
from aml_sentinel.models.summarizer import validate_summary
from aml_sentinel.models.triage import AlertBatch, TriageOutput

logger = logging.getLogger(__name__)

BASE_REPO: Final = "convaiinnovations/laya"
MANIFEST: Final = "manifest.json"
CHECKPOINT_FILES: Final = (
    "model.safetensors",
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "encoder/config.json",
)


class LayaAgentLike(Protocol):
    """The subset of ``laya.Agent`` this adapter relies on (injectable for tests)."""

    def predict_batch(
        self,
        states: list[str],
        questions: dict[str, dict[str, Any]],
        batch_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """Answer ``questions`` for every state."""
        ...


def write_manifest(checkpoint_dir: Path, *, source: str, revision: str | None) -> dict[str, Any]:
    """Record provenance and SHA-256 digests of a checkpoint's files.

    Args:
        checkpoint_dir: Directory holding a Laya checkpoint.
        source: Where it came from (Hub repo id or ``"finetune:<variant>"``).
        revision: Pinned Hub commit, if downloaded.

    Returns:
        The manifest that was written.
    """
    present = [f for f in CHECKPOINT_FILES if (checkpoint_dir / f).is_file()]
    manifest = {
        "source": source,
        "revision": revision,
        "sha256": {f: sha256_file(checkpoint_dir / f) for f in present},
    }
    (checkpoint_dir / MANIFEST).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return manifest


def verified_digests(checkpoint_dir: Path) -> dict[str, str]:
    """Verify every file listed in the checkpoint manifest.

    Args:
        checkpoint_dir: Checkpoint directory.

    Returns:
        ``{relative_path: sha256}`` for Laya's ``expected_sha256``.

    Raises:
        DataNotFoundError: If the manifest or a listed file is missing.
        IntegrityError: If any digest does not match.
        ConfigurationError: If the manifest does not cover the weights.
    """
    manifest_path = checkpoint_dir / MANIFEST
    if not manifest_path.is_file():
        raise DataNotFoundError(f"{MANIFEST} not found in checkpoint directory")
    digests = json.loads(manifest_path.read_text(encoding="utf-8")).get("sha256", {})
    if "model.safetensors" not in digests:
        raise ConfigurationError("manifest does not cover model.safetensors")
    for rel, digest in digests.items():
        verify_checksum(checkpoint_dir / rel, str(digest))
    return {str(k): str(v) for k, v in digests.items()}


def download_base(dest: Path, *, repo: str = BASE_REPO, revision: str | None = None) -> Path:
    """Download the base checkpoint at a pinned commit and write its manifest.

    Without an explicit ``revision`` the current Hub commit is resolved once and pinned
    in the manifest (trust on first use). Later loads verify against those digests.

    Args:
        dest: Target directory.
        repo: Hub repository id.
        revision: Commit SHA to pin, or ``None`` to pin the current one.

    Returns:
        The checkpoint directory.
    """
    from huggingface_hub import HfApi, snapshot_download

    pinned = revision or HfApi().model_info(repo).sha
    snapshot_download(
        repo,
        revision=pinned,
        local_dir=dest,
        allow_patterns=["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"],
    )
    write_manifest(dest, source=repo, revision=pinned)
    logger.info("downloaded laya base", extra={"revision": pinned})
    return dest


def load_agent(checkpoint_dir: Path, *, device: str | None = None) -> LayaAgentLike:
    """Load a verified local Laya checkpoint.

    Args:
        checkpoint_dir: Directory with a manifest (see :func:`write_manifest`).
        device: ``"cuda"``, ``"cpu"`` or ``None`` for Laya's auto-detection.

    Returns:
        The loaded agent.

    Raises:
        ConfigurationError: If the optional ``laya`` dependency is missing.
    """
    digests = verified_digests(checkpoint_dir)
    try:
        import laya
    except ImportError:
        raise ConfigurationError(
            "laya is not installed; pip install -r requirements/laya.txt"
        ) from None
    agent: LayaAgentLike = laya.load(str(checkpoint_dir), device=device, expected_sha256=digests)
    return agent


class LayaTriage:
    """Laya as a ``Triage`` model over alert summaries.

    Args:
        agent: A loaded Laya agent (or a test double).
        variant: Tag for reports, e.g. ``"zero_shot"`` or ``"head_only"``.
        batch_size: Summaries per forward pass.
    """

    def __init__(self, agent: LayaAgentLike, *, variant: str, batch_size: int = 32) -> None:
        self._agent = agent
        self.variant = variant
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        """``laya[<variant>]``."""
        return f"laya[{self.variant}]"

    def triage(self, batch: AlertBatch) -> TriageOutput:
        """Answer the typology, risk and escalation questions for every alert.

        Args:
            batch: Needs ``summaries`` aligned with ``features``.

        Returns:
            ``scores`` = P(escalate); ``typology``, ``risk`` (1-5), ``escalate``
            (P >= 0.5); ``extras`` hold per-label typology probabilities and the
            expected risk.

        Raises:
            DataValidationError: If summaries are missing, misaligned or invalid.
            ModelOutputError: If Laya truncated a summary or returned malformed answers.
        """
        summaries = checked_summaries(batch)
        results = self._agent.predict_batch(list(summaries), QUESTIONS, batch_size=self.batch_size)
        return decisions_to_output(results, len(summaries))


def checked_summaries(batch: AlertBatch) -> Sequence[str]:
    """Return the batch summaries after validating count and content.

    Args:
        batch: Alert batch.

    Returns:
        The summaries.

    Raises:
        DataValidationError: If summaries are missing, misaligned or invalid.
    """
    if batch.summaries is None or len(batch.summaries) != len(batch):
        raise DataValidationError("text triage needs one summary per alert")
    for s in batch.summaries:
        validate_summary(s)
    return batch.summaries


def decisions_to_output(results: Sequence[Any], expected: int) -> TriageOutput:
    """Convert raw typed-decision results into a validated :class:`TriageOutput`.

    Args:
        results: One raw result per alert (``{"answers": ..., "usage": ...}``).
        expected: Number of alerts sent.

    Returns:
        The triage output.

    Raises:
        ModelOutputError: On a count mismatch, truncation or malformed answers.
    """
    if len(results) != expected:
        raise ModelOutputError("model returned the wrong number of results")
    parsed = []
    for r in results:
        if not isinstance(r, dict):
            raise ModelOutputError("result is not an object")
        if r.get("usage", {}).get("truncated"):
            # A truncated summary means the answer did not see the whole alert.
            raise ModelOutputError("summary was truncated by the model context")
        parsed.append(parse_answers(r.get("answers")))
    extras = {
        f"p_{label}": np.array([p.typology_probs[label] for p in parsed], dtype=np.float64)
        for label in TYPOLOGY_LABELS
    }
    extras["risk_expected"] = np.array([p.risk_expected for p in parsed], dtype=np.float64)
    scores = np.array([p.escalate_p for p in parsed], dtype=np.float64)
    return TriageOutput(
        scores=scores,
        typology=tuple(p.typology for p in parsed),
        risk=np.array([p.risk_level for p in parsed], dtype=np.int64),
        escalate=scores >= 0.5,
        extras=extras,
    )
