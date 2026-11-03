from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from docs.projects.regression.code import baselines
from docs.projects.regression.code.data import (
    load_development_splits,
    prepare_development_data,
)


@pytest.fixture(scope="session")
def prepared_data():
    return prepare_development_data(load_development_splits())


@pytest.fixture(scope="session")
def baseline_study(prepared_data):
    return baselines.run_baselines(prepared_data)


def test_regression_metrics_known_example():
    metrics = baselines.regression_metrics(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([2.0, 2.0, 1.0]),
    )
    assert metrics["mae_usd"] == pytest.approx(1.0)
    assert metrics["rmse_usd"] == pytest.approx(np.sqrt(5 / 3))
    assert metrics["r2"] == pytest.approx(-1.5)


@pytest.mark.parametrize(
    ("true", "predicted"),
    [([], []), ([1.0], [1.0, 2.0]), ([1.0], [np.nan])],
)
def test_regression_metrics_reject_invalid_inputs(true, predicted):
    with pytest.raises(ValueError):
        baselines.regression_metrics(np.asarray(true), np.asarray(predicted))


def test_study_contains_dummy_and_eight_ridges(baseline_study):
    results = baseline_study.results
    assert len(results) == 9
    assert sum(result.model == "DummyRegressor" for result in results) == 1
    assert sum(result.model == "Ridge" for result in results) == 8
    assert {result.configuration for result in results if result.model == "Ridge"} == {
        f"alpha={alpha:g}" for alpha in baselines.RIDGE_ALPHAS
    }


def test_study_is_finite_sorted_and_defines_threshold(baseline_study):
    results = baseline_study.results
    maes = [result.mae_usd for result in results]
    assert maes == sorted(maes)
    assert all(
        np.isfinite([result.mae_usd, result.rmse_usd, result.r2]).all()
        for result in results
    )
    assert baseline_study.best_baseline == results[0]
    assert baseline_study.mlp_success_threshold_mae_usd == pytest.approx(
        0.85 * results[0].mae_usd
    )


def test_all_models_receive_the_same_prepared_matrices(monkeypatch):
    calls = []

    class SpyModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, X, y):
            calls.append(("fit", id(X), id(y)))
            return self

        def predict(self, X):
            calls.append(("predict", id(X)))
            return np.asarray([1.0, 2.0])

    monkeypatch.setattr(baselines, "DummyRegressor", SpyModel)
    monkeypatch.setattr(baselines, "Ridge", SpyModel)
    prepared = SimpleNamespace(
        X_train=sparse.csr_matrix([[1.0], [2.0]]),
        X_validation=sparse.csr_matrix([[3.0], [4.0]]),
        y_train=np.asarray([1.0, 2.0]),
        y_validation=np.asarray([1.0, 2.0]),
    )
    baselines.run_baselines(prepared)
    fit_calls = [call for call in calls if call[0] == "fit"]
    predict_calls = [call for call in calls if call[0] == "predict"]
    assert len(fit_calls) == 9
    assert len(predict_calls) == 9
    assert {call[1:] for call in fit_calls} == {
        (id(prepared.X_train), id(prepared.y_train))
    }
    assert {call[1] for call in predict_calls} == {id(prepared.X_validation)}


def test_payload_does_not_contain_test_results(baseline_study):
    payload = baselines.study_payload(baseline_study)
    assert payload["selection_split"] == "validation"
    assert payload["test_accessed"] is False
    assert "test_metrics" not in payload
    assert payload["best_baseline"] == payload["results"][0]


def test_versioned_outputs_match_study(baseline_study):
    baselines.check_outputs(baseline_study)
    payload = json.loads(baselines.RESULTS_PATH.read_text(encoding="utf-8"))
    assert payload == baselines.study_payload(baseline_study)
    assert baselines.TABLE_PATH.read_text(encoding="utf-8") == (
        baselines.render_markdown_table(payload)
    )


def test_check_outputs_reports_json_divergence(tmp_path, baseline_study):
    results_path = tmp_path / "results.json"
    table_path = tmp_path / "table.md"
    baselines.write_outputs(
        baseline_study, results_path=results_path, table_path=table_path
    )
    changed = replace(
        baseline_study,
        mlp_success_threshold_mae_usd=(
            baseline_study.mlp_success_threshold_mae_usd + 1.0
        ),
    )
    with pytest.raises(baselines.BaselineContractError, match="Resultados divergentes"):
        baselines.check_outputs(
            changed, results_path=results_path, table_path=table_path
        )


def test_check_outputs_reports_table_divergence(tmp_path, baseline_study):
    results_path = tmp_path / "results.json"
    table_path = tmp_path / "table.md"
    baselines.write_outputs(
        baseline_study, results_path=results_path, table_path=table_path
    )
    table_path.write_text("tabela alterada\n", encoding="utf-8")
    with pytest.raises(baselines.BaselineContractError, match="Tabela divergente"):
        baselines.check_outputs(
            baseline_study, results_path=results_path, table_path=table_path
        )
