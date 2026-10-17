"""Command-line entry point.

Usage::

    aml-sentinel generate-sample --seed 42
    aml-sentinel prepare-data --dataset HI-Small

``generate-sample`` writes the small synthetic dataset to ``data/sample/``.

``prepare-data`` loads and validates the raw CSV, estimates FX rates on the training
window only, adds USD amounts and typology labels, and writes a parquet cache plus a
JSON manifest (row counts, FX rates, SHA-256 checksums of inputs and output) to
``data/processed/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from aml_sentinel.config import Settings, load_settings
from aml_sentinel.data.fx import add_usd_amounts, estimate_usd_rates
from aml_sentinel.data.loader import load_transactions, patterns_path, transactions_path
from aml_sentinel.data.patterns import TYPOLOGY, attach_typology, load_patterns
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.data.splits import SplitName, TimeSplit, assert_no_future_rows
from aml_sentinel.data.synthetic import SyntheticConfig, generate, write_dataset
from aml_sentinel.exceptions import AmlSentinelError
from aml_sentinel.integrity import sha256_file
from aml_sentinel.logging_utils import configure_logging
from aml_sentinel.models.laya_finetune import build_laya_dataset, finetune_head_only
from aml_sentinel.models.system1a_report import system1a_command
from aml_sentinel.models.system1b_report import system1b_command
from aml_sentinel.rules.baseline import baseline_command

logger = logging.getLogger(__name__)


def prepare_data(settings: Settings, dataset: DatasetName) -> Path:
    """Build the processed parquet cache for one dataset.

    Args:
        settings: Validated settings.
        dataset: Dataset variant.

    Returns:
        Path of the written parquet file.

    Raises:
        AmlSentinelError: On missing files, schema violations or FX failures.
    """
    split = TimeSplit.from_settings(settings)
    source = settings.sample_dir if dataset is DatasetName.SAMPLE else settings.raw_dir
    trans_file = transactions_path(source, dataset)
    pat_file = patterns_path(source, dataset)

    frame = load_transactions(trans_file)
    train = split.partition(frame)[SplitName.TRAIN]
    assert_no_future_rows(train, split.train_end, what="fx fit data")
    rates = estimate_usd_rates(train)
    frame = add_usd_amounts(frame, rates)
    frame = attach_typology(frame, load_patterns(pat_file))
    split_names = split.name_of(frame[Col.TIMESTAMP])
    frame["split"] = split_names

    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    out = settings.processed_dir / f"{dataset.value}.parquet"
    frame.to_parquet(out, index=False)

    manifest = {
        "dataset": dataset.value,
        "rows": len(frame),
        "positives": int(frame[Col.IS_LAUNDERING].sum()),
        "rows_with_typology": int(frame[TYPOLOGY].notna().sum()),
        "split_rows": {str(k): int(v) for k, v in split_names.value_counts().items()},
        "split_bounds": {
            "train_end": split.train_end.isoformat(),
            "val_end": split.val_end.isoformat(),
            "test_end": split.test_end.isoformat(),
        },
        "fx_usd_rates_fit_on": "train",
        "fx_usd_rates": rates,
        "sha256": {
            trans_file.name: sha256_file(trans_file),
            pat_file.name: sha256_file(pat_file),
            out.name: sha256_file(out),
        },
    }
    manifest_file = settings.processed_dir / f"{dataset.value}.manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("prepared dataset", extra={"dataset": dataset.value, "rows": len(frame)})
    return out


def generate_sample(settings: Settings, seed: int) -> Path:
    """Generate the bundled synthetic sample into ``data/sample/``.

    Args:
        settings: Validated settings.
        seed: RNG seed; the same seed reproduces identical files.

    Returns:
        Path of the written transactions file.
    """
    data = generate(SyntheticConfig(seed=seed))
    out = write_dataset(data, settings.sample_dir, name=DatasetName.SAMPLE.value)
    logger.info(
        "generated synthetic sample",
        extra={"rows": len(data.transactions), "attempts": len(data.attempts), "seed": seed},
    )
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aml-sentinel")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate-sample", help="write the synthetic sample to data/sample")
    gen.add_argument("--seed", type=int, default=None, help="defaults to RANDOM_SEED")
    prep = sub.add_parser("prepare-data", help="validate raw data and build parquet cache")
    prep.add_argument(
        "--dataset",
        type=DatasetName,
        choices=list(DatasetName),
        default=DatasetName.HI_SMALL,
    )
    base = sub.add_parser("baseline", help="fit and evaluate the rule-engine baseline")
    base.add_argument(
        "--dataset",
        type=DatasetName,
        choices=list(DatasetName),
        default=DatasetName.HI_SMALL,
    )
    tab = sub.add_parser("train-tabular", help="System 1a: LightGBM + calibration vs baseline")
    tab.add_argument(
        "--dataset",
        type=DatasetName,
        choices=list(DatasetName),
        default=DatasetName.HI_SMALL,
    )
    for name, help_text in (
        ("build-laya-data", "System 1b: write the Laya fine-tuning JSONL"),
        ("finetune-laya", "System 1b: head-only Laya fine-tune on the local GPU"),
        ("triage-text", "System 1b: Laya (+Jev if enabled) on the alert queue, then fusion"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument(
            "--dataset",
            type=DatasetName,
            choices=list(DatasetName),
            default=DatasetName.HI_SMALL,
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Arguments (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code: 0 on success, 2 on a handled error.
    """
    args = _build_parser().parse_args(argv)
    try:
        settings = load_settings()
        configure_logging(settings.log_level, json_output=settings.log_json)
        if args.command == "generate-sample":
            generate_sample(settings, settings.random_seed if args.seed is None else args.seed)
        elif args.command == "prepare-data":
            prepare_data(settings, args.dataset)
        elif args.command == "baseline":
            baseline_command(settings, args.dataset)
        elif args.command == "train-tabular":
            system1a_command(settings, args.dataset)
        elif args.command == "build-laya-data":
            build_laya_dataset(settings, args.dataset)
        elif args.command == "finetune-laya":
            finetune_head_only(settings, args.dataset)
        elif args.command == "triage-text":
            system1b_command(settings, args.dataset)
    except AmlSentinelError as exc:
        # Our exceptions carry safe messages; anything else is a bug and should
        # surface with a traceback.
        logging.getLogger(__name__).error("command failed: %s", exc)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
