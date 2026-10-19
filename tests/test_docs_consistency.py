"""README and docs/RESULTS.md quote measured numbers; they must match the committed reports.

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
RESULTS = (ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
MINUS = chr(0x2212)  # docs use a true minus sign


def _report(*parts: str) -> dict[str, Any]:
    path = ROOT.joinpath("reports", *parts)
    if not path.is_file():  # pragma: no cover - reports are committed
        pytest.skip(f"{path.name} not present")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def _row(first_cell: str, doc: str = RESULTS) -> list[str]:
    for line in doc.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == first_cell:
            return cells
    raise AssertionError(f"doc has no table row starting with {first_cell!r}")


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
        assert cells[4].replace(MINUS, "-") == expected


def test_final_ranking_numbers_match_report() -> None:
    r = _report("final", "HI-Small.json")
    s1 = r["system1_only"]["ranking"]["average_precision"]
    final = r["final"]["ranking"]["average_precision"]
    assert _row("System 1 only (`fused[lgbm_all+laya[head_only]]`)")[1] == f"{s1:.3f}"
    assert _row("Final (System 1 + System 2 evidence)")[1] == f"{final:.3f}"


def _ablation(step: str) -> dict[str, Any]:
    steps = _report("evaluation", "HI-Small.json")["ablation"]["steps"]
    found: dict[str, Any] = next(s for s in steps if s["name"] == step)
    return found


def test_readme_key_results_match_reports() -> None:
    s1a = _report("system1a", "HI-Small.json")
    lgbm = s1a["variants"]["all"]
    cells = _row("PR-AUC, every test transaction", README)
    assert cells[1] == f"{s1a['baseline_test']['ranking']['average_precision']:.4f}"
    assert cells[2] == f"{lgbm['ranking']['average_precision']:.4f} (LightGBM)"

    matched = lgbm["vs_baseline"]["matched_recall"]
    recall = 100 * matched["baseline"]["recall"]
    cells = _row(f"False positives at the rules' recall ({recall:.1f}%)", README)
    assert cells[1] == f"{matched['baseline']['fp']:,}"
    assert cells[2] == f"{matched['model']['fp']:,} ({MINUS}{100 * matched['fp_reduction']:.1f}%)"

    cells = _row("False positives in the top 5% of the 4,000-alert queue", README)
    assert cells[1] == str(_ablation("rules only")["top_5pct"]["fp"])
    assert cells[2] == f"{_ablation('+ LightGBM')['top_5pct']['fp']} (+ LightGBM)"

    cells = _row("PR-AUC inside the alert queue (6.8% base rate)", README)
    assert cells[1] == f"{_ablation('rules only')['ranking']['average_precision']:.3f}"
    assert cells[2] == f"{_ablation('+ Laya')['ranking']['average_precision']:.3f} (+ Laya)"

    laya = _report("system1b", "HI-Small.json")["models"]["laya[head_only]"]
    cells = _row("Laya alone inside the queue, PR-AUC", README)
    assert cells[2] == f"{laya['ranking']['average_precision']:.3f} (near random)"

    stages = _report("evaluation", "HI-Small.json")["latency"]["stages"]
    rules = next(v for k, v in stages.items() if k.startswith("rules"))
    model = stages["LightGBM inference (batch)"]
    cells = _row("Latency per transaction, rules vs LightGBM", README)
    assert cells[1:3] == [f"{rules['us_per_tx']:.1f} µs", f"{model['us_per_tx']:.1f} µs"]


def test_readme_defers_detail_to_results() -> None:
    # The detailed tables live in one place, so they cannot drift between two copies.
    for detail_row in ("| lgbm_all |", "| + graph evidence |", "| scatter_gather |"):
        assert detail_row in RESULTS
        assert detail_row not in README
    assert "docs/RESULTS.md" in README
