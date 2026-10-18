"""The README quotes measured numbers; they must match the committed reports.

This guards against a real drift found in Phase 8: System 1b was re-run, its Laya
latency changed, and the README still showed the old numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def _report(*parts: str) -> dict[str, Any]:
    path = ROOT.joinpath("reports", *parts)
    if not path.is_file():  # pragma: no cover - reports are committed
        pytest.skip(f"{path.name} not present")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _row(first_cell: str) -> list[str]:
    for line in README.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == first_cell:
            return cells
    raise AssertionError(f"README has no table row starting with {first_cell!r}")


@pytest.mark.parametrize(
    ("name", "section"),
    [
        ("lgbm_all", "models"),
        ("lgbm_no_shortcuts", "models"),
        ("laya[zero_shot]", "models"),
        ("laya[head_only]", "models"),
        ("fused[lgbm_all+laya[head_only]]", "fusion"),
        ("fused[lgbm_no_shortcuts+laya[head_only]]", "fusion"),
    ],
)
def test_system1b_table_matches_report(name: str, section: str) -> None:
    m = _report("system1b", "HI-Small.json")[section][name]
    cells = _row(name)
    assert cells[1] == f"{m['ranking']['average_precision']:.3f}"
    assert cells[2] == f"{m['ranking']['roc_auc']:.3f}"
    assert cells[3] == f"{100 * m['precision_at_k']['100']['precision']:.1f}%"
    assert cells[4] == f"{m['ece_quantile']:.3f}"
    latency = m.get("latency_ms_per_alert")
    assert cells[6] == ("n/a" if latency is None else f"{latency:.0f}")


@pytest.mark.parametrize(
    ("readme_name", "step"),
    [
        ("rules only (re-ranking the queue)", "rules only"),
        ("+ LightGBM", "+ LightGBM"),
        ("+ Laya (head-only)", "+ Laya"),
        ("+ graph evidence", "+ graph evidence"),
        ("+ GNN (**lookahead**, not deployable)", "+ GNN (full stack)"),
    ],
)
def test_ablation_table_matches_report(readme_name: str, step: str) -> None:
    steps = _report("evaluation", "HI-Small.json")["ablation"]["steps"]
    m = next(s for s in steps if s["name"] == step)
    cells = _row(readme_name)
    assert cells[1] == f"{m['ranking']['average_precision']:.3f}"
    assert cells[2] == f"{100 * m['precision_at_k']['100']['precision']:.1f}%"
    assert cells[3] == str(m["top_5pct"]["fp"])
    d = m["vs_previous"]
    if d is not None:
        expected = f"{d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
        assert cells[4].replace(chr(0x2212), "-") == expected  # README uses a true minus sign


def test_final_ranking_numbers_match_report() -> None:
    r = _report("final", "HI-Small.json")
    s1 = r["system1_only"]["ranking"]["average_precision"]
    final = r["final"]["ranking"]["average_precision"]
    assert _row("System 1 only (`fused[lgbm_all+laya[head_only]]`)")[1] == f"{s1:.3f}"
    assert _row("Final (System 1 + System 2 evidence)")[1] == f"{final:.3f}"
