from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "docs/projects/regression/index.md"
REGRESSION_DIR = REPORT_PATH.parent


def _report_text() -> str:
    return REPORT_PATH.read_text(encoding="utf-8")


def _artifact(name: str) -> dict:
    return json.loads((REGRESSION_DIR / name).read_text(encoding="utf-8"))


def _usd(value: float) -> str:
    formatted = f"{value:,.2f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def test_required_sections_are_in_order_and_complete():
    report = _report_text()
    sections = re.findall(r"^## ([0-7])\. ", report, flags=re.MULTILINE)
    assert sections == list("01234567")
    assert "pendente" not in report.casefold()


def test_summary_has_exactly_eight_rows_and_matches_artifacts():
    report = _report_text()
    summary = report.split("### Resumo dos resultados", maxsplit=1)[1]
    rows = re.findall(r"^\| ([1-8]) \|.*$", summary, flags=re.MULTILINE)
    assert rows == list("12345678")

    manifest = _artifact("split_manifest.json")
    config = _artifact("final_config.json")
    final = _artifact("final_test_results.json")
    analysis = _artifact("error_analysis.json")

    split_rows = [
        manifest["splits"][name]["rows"]
        for name in ("train", "validation", "test_locked")
    ]
    assert " · ".join(f"{value:,}".replace(",", ".") for value in split_rows) in summary

    mlp_models = [final["models"][f"mlp_seed_{seed}"] for seed in (42, 43, 44)]
    aggregate = final["models"]["mlp_aggregate"]
    ridge = final["models"]["ridge"]
    criterion = final["success_criterion"]
    model = config["model"]
    training = config["training"]
    worst_group = analysis["error_by_credential"][0]

    assert f"MAE US$ {_usd(ridge['mae_usd'])}" in summary
    assert "→".join(map(str, [model["input_dim"], *model["hidden_dims"], 1])) in summary
    assert f"{model['parameter_count']:,}".replace(",", ".") in summary
    assert f"batch {training['batch_size']}" in summary
    for model in mlp_models:
        assert str(model["stopped_epoch"]) in summary
    assert f"MAE US$ {_usd(aggregate['mae_usd']['mean'])}" in summary
    assert f"US$ {_usd(aggregate['mae_usd']['std'])}" in summary
    assert "3 seeds" in summary
    assert f"{_usd(criterion['observed_improvement_pct'])}%" in summary
    assert f"MAE US$ {_usd(worst_group['mae_usd'])}" in summary
    assert f"`n={worst_group['count']}`" in summary


def test_multiseed_result_and_access_history_are_explicit():
    report = _report_text().casefold()
    final = _artifact("final_test_results.json")
    assert final["protocol"]["seeds"] == [42, 43, 44]
    assert final["protocol"]["multiple_seeds"] is True
    assert final["models"]["mlp_aggregate"]["seed_count"] == 3
    assert final["limitations"] == {
        "test_reopened_for_multiseed_correction": True,
        "configuration_changed_after_first_test_access": False,
    }
    assert "seeds 42, 43 e 44" in report
    assert "média ± desvio" in report
    assert "teste precisou ser reaberto uma segunda vez" in report


def test_all_referenced_figures_and_snippets_exist():
    report = _report_text()
    figures = re.findall(r"!\[[^]]*\]\(([^)]+)\)", report)
    snippets = re.findall(r'--8<-- "([^"]+)"', report)

    assert figures
    assert snippets
    for relative_path in figures:
        path = REGRESSION_DIR / relative_path
        assert path.is_file(), f"Figura ausente: {path}"
        assert path.stat().st_size > 0, f"Figura vazia: {path}"
    for relative_path in snippets:
        path = ROOT / relative_path
        assert path.is_file(), f"Tabela ausente: {path}"
        assert path.stat().st_size > 0, f"Tabela vazia: {path}"


def test_regression_report_is_registered_once_in_mkdocs_navigation():
    mkdocs = (ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert mkdocs.count("projects/regression/index.md") == 1
