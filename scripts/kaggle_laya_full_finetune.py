"""Full Laya fine-tune for Kaggle (GPU T4 x1 or x2, ~16 GB VRAM). Standalone script.

Why Kaggle: a full fine-tune of the 421M-parameter encoder needs about 7+ GB for
weights and AdamW state alone, more than the 4 GB laptop GPU used for the local
head-only run (ADR-013).

Steps (all data is synthetic; never upload real customer data):

1. On your machine: ``make laya-data``. This writes
   ``data/processed/laya/HI-Small.train.jsonl`` (synthetic summaries only, with no
   account IDs).
2. Create a *private* Kaggle dataset containing that JSONL file.
3. New Kaggle notebook -> Accelerator: GPU T4 -> add the dataset -> run:

       !pip install -q laya==0.4.9
       !python kaggle_laya_full_finetune.py --data <dataset dir>/HI-Small.train.jsonl
           --out /kaggle/working/HI-Small-full     (one line in the notebook)

4. Download ``/kaggle/working/HI-Small-full`` into ``models/laya/HI-Small-full``.
5. Locally: ``make triage-text``. The loader verifies every file against the
   ``manifest.json`` written below before any weight is parsed.

Laya's README reports roughly 4-5 h on 2x T4 for 4 epochs of ~30k questions. This
set has about 47k question items, so expect several hours; ``--epochs 2`` halves it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

# Same pinned base commit as the local runs (models/laya/base/manifest.json).
BASE_REPO = "convaiinnovations/laya"
BASE_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
FILES = (
    "model.safetensors",
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "encoder/config.json",
)

logger = logging.getLogger("kaggle_laya")


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Download the pinned base, fine-tune every weight, write a checksum manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    from huggingface_hub import snapshot_download
    from laya.train import TrainConfig, finetune

    base = Path(
        snapshot_download(
            BASE_REPO,
            revision=BASE_REVISION,
            local_dir="/kaggle/working/laya-base",
            allow_patterns=[
                "rl_agent_config.json",
                "model.safetensors",
                "tokenizer/*",
                "encoder/*",
            ],
        )
    )
    config = TrainConfig(epochs=args.epochs, seed=args.seed)  # freeze_encoder=False: full
    summary = finetune(str(args.data), str(base), str(args.out), config, device="cuda")

    manifest = {
        "source": "finetune:full",
        "revision": BASE_REVISION,
        "sha256": {f: sha256(args.out / f) for f in FILES if (args.out / f).is_file()},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out / "aml_train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("done: %s", args.out)


if __name__ == "__main__":
    main()
