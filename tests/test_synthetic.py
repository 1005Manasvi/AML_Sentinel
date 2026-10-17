from __future__ import annotations

from pathlib import Path

import pytest

from aml_sentinel.cli import generate_sample, main, prepare_data
from aml_sentinel.config import load_settings
from aml_sentinel.data.fx import estimate_usd_rates
from aml_sentinel.data.loader import load_transactions
from aml_sentinel.data.patterns import TYPOLOGY, Typology, attach_typology, load_patterns
from aml_sentinel.data.schema import Col, DatasetName
from aml_sentinel.data.synthetic import SAMPLE_FX, SyntheticConfig, generate, write_dataset

SMALL = SyntheticConfig(n_accounts=150, n_benign=1200, attempts_per_typology=2)


@pytest.fixture(scope="module")
def sample_files(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("sample")
    write_dataset(generate(SMALL), out)
    return out


def test_deterministic_for_same_seed() -> None:
    assert generate(SMALL) == generate(SMALL)
    assert generate(SMALL) != generate(SyntheticConfig(**{**_asdict(SMALL), "seed": 7}))


def _asdict(cfg: SyntheticConfig) -> dict[str, object]:
    return {f: getattr(cfg, f) for f in cfg.__slots__}  # type: ignore[attr-defined]


def test_output_passes_loader_validation(sample_files: Path) -> None:
    df = load_transactions(sample_files / "Sample_Trans.csv")
    data = generate(SMALL)
    assert len(df) == len(data.transactions)
    assert int(df[Col.IS_LAUNDERING].sum()) == sum(len(a.rows) for a in data.attempts)
    # Some bank codes must carry leading zeros, mirroring IBM's identifiers.
    assert df[Col.FROM_BANK].astype(str).str.startswith("0").any()


def test_every_typology_planted_and_documented_rows_attach(sample_files: Path) -> None:
    data = generate(SMALL)
    assert {a.typology for a in data.attempts} == set(Typology)
    df = attach_typology(
        load_transactions(sample_files / "Sample_Trans.csv"),
        load_patterns(sample_files / "Sample_Patterns.txt"),
    )
    documented = sum(len(a.rows) for a in data.attempts if a.documented)
    assert int(df[TYPOLOGY].notna().sum()) == documented
    assert (df.loc[df[TYPOLOGY].notna(), Col.IS_LAUNDERING] == 1).all()


def test_has_undocumented_attempts_and_benign_lookalikes() -> None:
    data = generate(SyntheticConfig())
    assert any(not a.documented for a in data.attempts)
    assert data.n_lookalike_rows > 0
    lookalike_free = generate(SyntheticConfig(lookalikes_per_kind=0))
    assert lookalike_free.n_lookalike_rows == 0


def test_fx_estimator_recovers_planted_rates(sample_files: Path) -> None:
    rates = estimate_usd_rates(load_transactions(sample_files / "Sample_Trans.csv"))
    for currency, rate in rates.items():
        # 2-decimal rounding on small amounts is the only source of error.
        assert rate == pytest.approx(SAMPLE_FX[currency], rel=1e-3)


def test_cli_generate_and_prepare_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = load_settings(_env_file=None, data_dir=tmp_path)
    generate_sample(settings, seed=3)
    out = prepare_data(settings, DatasetName.SAMPLE)
    assert out.name == "Sample.parquet"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "other"))
    assert main(["generate-sample", "--seed", "1"]) == 0
    assert (tmp_path / "other" / "sample" / "Sample_Trans.csv").exists()


def test_bundled_sample_matches_generator() -> None:
    """The committed data/sample files must be exactly what seed 42 regenerates."""
    bundled = Path(__file__).resolve().parents[1] / "data" / "sample" / "Sample_Trans.csv"
    if not bundled.exists():
        pytest.skip("bundled sample not generated yet")
    data = generate(SyntheticConfig(seed=42))
    lines = bundled.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(data.transactions) + 1
    assert lines[1].split(",") == list(data.transactions[0])
