"""Shared fixtures: a tiny synthetic dataset in the exact IBM AMLworld file format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aml_sentinel.api.app import create_app
from aml_sentinel.api.auth import hash_api_key
from aml_sentinel.api.queue import AlertQueue, build_queue
from aml_sentinel.api.scoring import TransactionScorer, build_scorer
from aml_sentinel.api.service import CaseService
from aml_sentinel.api.store import CaseStore
from aml_sentinel.audit import AuditLog
from aml_sentinel.cli import generate_sample, prepare_data
from aml_sentinel.config import Settings, load_settings
from aml_sentinel.data.processed import load_processed
from aml_sentinel.data.schema import RAW_HEADER, DatasetName
from aml_sentinel.models.lgbm import LgbmParams
from aml_sentinel.models.system1a_report import system1a_command
from aml_sentinel.rules.baseline import baseline_command
from aml_sentinel.system2.cases import CaseIndex

HEADER = ",".join(RAW_HEADER)
EUR_USD = 1.10  # USD per EUR planted in the training period


def raw_row(
    ts: str,
    from_bank: str,
    from_acct: str,
    to_bank: str,
    to_acct: str,
    amount_received: float,
    receiving_currency: str,
    amount_paid: float,
    payment_currency: str,
    fmt: str = "Wire",
    label: int = 0,
) -> str:
    return (
        f"{ts},{from_bank},{from_acct},{to_bank},{to_acct},{amount_received:.2f},"
        f"{receiving_currency},{amount_paid:.2f},{payment_currency},{fmt},{label}"
    )


# A laundering cycle A -> B -> C -> A on 2022-09-07 (validation period).
CYCLE_ROWS = [
    raw_row(
        "2022/09/07 09:00",
        "010",
        "AAA1",
        "020",
        "BBB2",
        900,
        "US Dollar",
        900,
        "US Dollar",
        "ACH",
        1,
    ),
    raw_row(
        "2022/09/07 10:00",
        "020",
        "BBB2",
        "030",
        "CCC3",
        880,
        "US Dollar",
        880,
        "US Dollar",
        "ACH",
        1,
    ),
    raw_row(
        "2022/09/07 11:00",
        "030",
        "CCC3",
        "010",
        "AAA1",
        860,
        "US Dollar",
        860,
        "US Dollar",
        "ACH",
        1,
    ),
]


def trans_rows() -> list[str]:
    rows: list[str] = []
    # Deliberately out of order: the loader must sort.
    rows.append(
        raw_row(
            "2022/09/12 08:00",
            "010",
            "AAA1",
            "040",
            "DDD4",
            50,
            "US Dollar",
            50,
            "US Dollar",
            "Cash",
            1,
        )
    )  # tail
    for day in range(1, 7):  # training period: EUR<->USD at 1.10
        ts = f"2022/09/0{day} 12:00"
        rows.append(raw_row(ts, "010", "AAA1", "020", "BBB2", 110, "US Dollar", 100, "Euro"))
        rows.append(raw_row(ts, "020", "BBB2", "010", "AAA1", 100, "Euro", 110, "US Dollar"))
        rows.append(
            raw_row(
                ts,
                "00110",
                "EEE5",
                "00110",
                "EEE5",
                500,
                "US Dollar",
                500,
                "US Dollar",
                "Reinvestment",
            )
        )
    rows.extend(CYCLE_ROWS)
    # Test period, with an absurd EUR rate: must NOT influence FX fitted on train.
    for hour in range(10, 20):
        rows.append(
            raw_row(
                f"2022/09/09 {hour}:00", "040", "DDD4", "050", "FFF6", 9000, "US Dollar", 1, "Euro"
            )
        )
    return rows


PATTERNS_TEXT = "\n".join(
    [
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 3 hops",
        *CYCLE_ROWS,
        "END LAUNDERING ATTEMPT - CYCLE",
        "",
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 1-degree Fan-Out",
        raw_row(
            "2022/09/12 08:00",
            "010",
            "AAA1",
            "040",
            "DDD4",
            50,
            "US Dollar",
            50,
            "US Dollar",
            "Cash",
            1,
        ),
        "END LAUNDERING ATTEMPT - FAN-OUT",
        "",
    ]
)


@dataclass(frozen=True)
class RawDataset:
    settings: Settings
    trans_file: Path
    patterns_file: Path


@pytest.fixture
def raw_dataset(tmp_path: Path) -> RawDataset:
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    trans_file = raw / "HI-Small_Trans.csv"
    trans_file.write_text("\n".join([HEADER, *trans_rows()]) + "\n", encoding="utf-8")
    patterns_file = raw / "HI-Small_Patterns.txt"
    patterns_file.write_text(PATTERNS_TEXT, encoding="utf-8")
    settings = load_settings(_env_file=None, data_dir=tmp_path / "data", log_json=True)
    return RawDataset(settings, trans_file, patterns_file)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's real environment out of tests."""
    for var in (
        "LLM_PROVIDER",
        "LLM_API_KEY",
        "JEV_ENABLED",
        "JEV_API_KEY",
        "DATA_DIR",
        "TRAIN_END",
        "REPORTS_DIR",
        "MODELS_DIR",
        "VAL_END",
        "TEST_END",
        "ENVIRONMENT",
        "API_KEYS",
        "API_DATASET",
        "API_CORS_ORIGINS",
        "API_ALLOWED_HOSTS",
        "API_RATE_LIMIT_PER_MINUTE",
        "API_MAX_REQUEST_BYTES",
        "API_SCORE_MAX_HISTORY_ROWS",
        "DASHBOARD_API_URL",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------- API / dashboard
FAST_LGBM = LgbmParams(max_rounds=30, early_stopping_rounds=5, min_child_samples=20, num_threads=1)
ANALYST_KEY = "amls_test-analyst-key-0000000000000000000000000"
ADMIN_KEY = "amls_test-admin-key-00000000000000000000000000000"


@dataclass(frozen=True)
class ServedSample:
    """The bundled sample after the System 1a pipeline, ready to be served."""

    settings: Settings
    frame: pd.DataFrame
    index: CaseIndex
    queue: AlertQueue
    scorer: TransactionScorer | None


@pytest.fixture(scope="session")
def served_sample(tmp_path_factory: pytest.TempPathFactory) -> ServedSample:
    """Generate, prepare and train the sample once per test session (about 10 s)."""
    root = tmp_path_factory.mktemp("served")
    # Every field the API reads is passed explicitly: this fixture is session-scoped,
    # so the per-test environment isolation above does not apply to it.
    settings = load_settings(
        _env_file=None,
        data_dir=root / "data",
        reports_dir=root / "reports",
        models_dir=root / "models",
        api_dataset=DatasetName.SAMPLE,
        api_allowed_hosts=("testserver",),
        api_cors_origins=("http://localhost:8501",),
        api_keys=f"ana:analyst:{hash_api_key(ANALYST_KEY)},adm:admin:{hash_api_key(ADMIN_KEY)}",
        api_rate_limit_per_minute=10_000,
        llm_provider="template",
    )
    generate_sample(settings, seed=42)
    prepare_data(settings, DatasetName.SAMPLE)
    baseline_command(settings, DatasetName.SAMPLE)
    system1a_command(
        settings,
        DatasetName.SAMPLE,
        params=LgbmParams(
            max_rounds=30, early_stopping_rounds=5, min_child_samples=20, num_threads=1
        ),
    )
    frame = load_processed(settings.processed_dir, DatasetName.SAMPLE)
    index = CaseIndex(frame)
    queue = build_queue(settings, DatasetName.SAMPLE, frame, index)
    scorer, _ = build_scorer(settings, DatasetName.SAMPLE, frame)
    return ServedSample(settings, frame, index, queue, scorer)


def make_service(served: ServedSample, tmp_path: Path) -> CaseService:
    """A service over the shared sample with its own, empty store and audit log."""
    return CaseService(
        queue=served.queue,
        index=served.index,
        store=CaseStore(tmp_path / "cases.sqlite3"),
        audit=AuditLog(tmp_path / "audit.jsonl"),
        scorer=served.scorer,
    )


@pytest.fixture
def service(served_sample: ServedSample, tmp_path: Path) -> CaseService:
    return make_service(served_sample, tmp_path)


@pytest.fixture
def client(served_sample: ServedSample, service: CaseService) -> TestClient:
    return TestClient(create_app(served_sample.settings, service=service))
