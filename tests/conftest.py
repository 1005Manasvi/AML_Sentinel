"""Shared fixtures: a tiny synthetic dataset in the exact IBM AMLworld file format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from aml_sentinel.config import Settings, load_settings
from aml_sentinel.data.schema import RAW_HEADER

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
    ):
        monkeypatch.delenv(var, raising=False)
