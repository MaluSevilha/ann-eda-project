from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from docs.projects.eda.code.pipeline import GROUP_COLUMN, MODEL_FEATURES, TARGET
from docs.projects.regression.code import data as regression_data
from docs.projects.regression.code import final_evaluation as evaluation
from docs.projects.regression.code.baselines import regression_metrics


def _test_frame(rows=3):
    values = {column: [0] * rows for column in MODEL_FEATURES}
    values.update(
        {
            "program_id": [f"p{index}" for index in range(rows)],
            GROUP_COLUMN: [f"g{index // 2}" for index in range(rows)],
            "credential_name": ["Bachelor"] * rows,
            "institution_control": ["Public"] * rows,
            TARGET: np.asarray([10_000.0, 20_000.0, 30_000.0])[:rows],
        }
    )
    return pd.DataFrame(values)


def test_private_test_loader_checks_manifest_identity(tmp_path, monkeypatch):
    test = _test_frame()
    expected = regression_data._split_identity(test)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"splits": {"test_locked": expected}}))
    monkeypatch.setattr(regression_data, "_load_modeling_data", lambda _path: object())
    monkeypatch.setattr(
        regression_data,
        "_split_all",
        lambda _data: (pd.DataFrame(), pd.DataFrame(), test),
    )
    observed = regression_data._load_locked_test_for_final_evaluation(
        tmp_path / "data.csv", manifest
    )
    assert observed is test
    payload = json.loads(manifest.read_text())
    payload["splits"]["test_locked"]["rows"] += 1
    manifest.write_text(json.dumps(payload))
    with pytest.raises(regression_data.DataContractError, match="Identidade"):
        regression_data._load_locked_test_for_final_evaluation(
            tmp_path / "data.csv", manifest
        )


def test_success_criterion_uses_seed_42_against_best_test_baseline():
    metrics = {
        "mlp_seed_42": {"mae_usd": 80.0, "rmse_usd": 100.0, "r2": 0.8},
        "ridge": {"mae_usd": 100.0, "rmse_usd": 120.0, "r2": 0.7},
        "dummy": {"mae_usd": 200.0, "rmse_usd": 220.0, "r2": 0.0},
    }
    payload = evaluation._build_core_payload(metrics, {"rows": 3}, 4, 8)
    assert payload["protocol"]["seed"] == 42
    assert payload["protocol"]["multiple_seeds"] is False
    assert payload["success_criterion"]["best_baseline"] == "ridge"
    assert payload["success_criterion"]["maximum_mlp_mae_usd"] == 85.0
    assert payload["success_criterion"]["observed_improvement_pct"] == 20.0
    assert payload["success_criterion"]["met"] is True
    assert payload["limitations"]["seed_variability_estimated"] is False


def test_baselines_fit_the_exact_same_training_matrices(monkeypatch):
    prepared = SimpleNamespace(
        X_train=object(),
        y_train=object(),
    )
    calls = []

    class Model:
        def __init__(self, label):
            self.label = label

        def fit(self, features, target):
            calls.append((self.label, features, target))
            return self

    monkeypatch.setattr(evaluation, "_selected_baseline_contract", lambda: {})
    monkeypatch.setattr(
        evaluation, "DummyRegressor", lambda **_kwargs: Model("dummy")
    )
    monkeypatch.setattr(evaluation, "Ridge", lambda **_kwargs: Model("ridge"))
    evaluation._fit_baselines(prepared)
    assert calls == [
        ("dummy", prepared.X_train, prepared.y_train),
        ("ridge", prepared.X_train, prepared.y_train),
    ]


def test_final_evaluation_trains_before_loading_test_and_transforms_once(monkeypatch):
    events = []

    class Preprocessor:
        def transform(self, _frame):
            events.append("transform_test")
            return sparse.csr_matrix(np.ones((3, 2), dtype=np.float32))

    prepared = SimpleNamespace(
        X_train=sparse.csr_matrix(np.ones((4, 2), dtype=np.float32)),
        X_validation=sparse.csr_matrix(np.ones((2, 2), dtype=np.float32)),
        y_train=np.asarray([10_000, 20_000, 30_000, 40_000], dtype=np.float32),
        y_validation=np.asarray([15_000, 25_000], dtype=np.float32),
        preprocessor=Preprocessor(),
    )
    frozen = SimpleNamespace(
        model_config=object(),
        training_config=SimpleNamespace(
            seed=42,
            device="cpu",
            validation_batch_size=10,
        ),
    )
    training = SimpleNamespace(
        model=object(),
        target_transformer=object(),
        best_epoch=4,
        stopped_epoch=8,
    )

    class Baseline:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float64)

        def predict(self, _matrix):
            return self.values

    monkeypatch.setattr(evaluation, "load_development_splits", lambda: object())
    monkeypatch.setattr(evaluation, "prepare_development_data", lambda _splits: prepared)
    monkeypatch.setattr(evaluation, "load_frozen_config", lambda: frozen)

    def fake_train(*_args):
        events.append("train_mlp")
        return training

    def fake_baselines(_prepared):
        events.append("fit_baselines")
        return Baseline([18_000, 18_000, 18_000]), Baseline([11_000, 19_000, 31_000])

    def fake_load_test():
        events.append("load_test")
        return _test_frame()

    monkeypatch.setattr(evaluation, "train_model", fake_train)
    monkeypatch.setattr(evaluation, "_fit_baselines", fake_baselines)
    monkeypatch.setattr(evaluation, "_load_locked_test_for_final_evaluation", fake_load_test)
    monkeypatch.setattr(
        evaluation,
        "predict_usd",
        lambda *_args, **_kwargs: np.asarray([9_000, 21_000, 29_000], dtype=np.float64),
    )
    result = evaluation.run_final_evaluation()
    assert events == ["train_mlp", "fit_baselines", "load_test", "transform_test"]
    assert result.mlp_training is training
    assert len(result.predictions) == 3
    assert result.metrics["mlp_seed_42"]["mae_usd"] == pytest.approx(1_000.0)


def _patch_artifact_paths(tmp_path, monkeypatch):
    paths = {
        "RESULTS_PATH": tmp_path / "results.json",
        "PREDICTIONS_PATH": tmp_path / "predictions.csv",
        "TABLE_PATH": tmp_path / "table.md",
        "PREDICTED_FIGURE_PATH": tmp_path / "predicted.png",
        "RESIDUAL_FIGURE_PATH": tmp_path / "residual.png",
        "SPLIT_MANIFEST_PATH": tmp_path / "manifest.json",
    }
    for name, path in paths.items():
        monkeypatch.setattr(evaluation, name, path)
    return paths


def _synthetic_result():
    predictions = pd.DataFrame(
        {
            "program_id": ["p0", "p1", "p2"],
            GROUP_COLUMN: ["g0", "g0", "g1"],
            "credential_name": ["Bachelor"] * 3,
            "institution_control": ["Public"] * 3,
            "actual_usd": [10_000.0, 20_000.0, 30_000.0],
            "mlp_seed_42_usd": [9_000.0, 21_000.0, 29_000.0],
            "ridge_usd": [8_000.0, 22_000.0, 28_000.0],
            "dummy_usd": [20_000.0, 20_000.0, 20_000.0],
        }
    )
    predictions["mlp_residual_usd"] = (
        predictions["actual_usd"] - predictions["mlp_seed_42_usd"]
    )
    predictions["mlp_absolute_error_usd"] = predictions["mlp_residual_usd"].abs()
    metrics = {
        "mlp_seed_42": regression_metrics(
            predictions["actual_usd"], predictions["mlp_seed_42_usd"]
        ),
        "ridge": regression_metrics(predictions["actual_usd"], predictions["ridge_usd"]),
        "dummy": regression_metrics(predictions["actual_usd"], predictions["dummy_usd"]),
    }
    return evaluation.FinalEvaluationResult(
        mlp_training=SimpleNamespace(best_epoch=4, stopped_epoch=8),
        predictions=predictions,
        metrics=metrics,
        test_identity={
            "rows": 3,
            "institutions": 2,
            "program_id_sha256": evaluation._program_id_sha256(predictions["program_id"]),
        },
    )


def test_write_and_offline_check_validate_all_artifacts(tmp_path, monkeypatch):
    paths = _patch_artifact_paths(tmp_path, monkeypatch)
    result = _synthetic_result()
    paths["SPLIT_MANIFEST_PATH"].write_text(
        json.dumps({"splits": {"test_locked": result.test_identity}})
    )
    monkeypatch.setattr(
        evaluation,
        "load_frozen_config",
        lambda: SimpleNamespace(training_config=SimpleNamespace(seed=42)),
    )
    monkeypatch.setattr(
        evaluation,
        "_load_locked_test_for_final_evaluation",
        lambda: pytest.fail("--check não pode carregar o teste"),
    )
    evaluation.write_outputs(result)
    payload = evaluation.check_outputs()
    assert payload["test_access_count"] == 1
    assert payload["models"]["mlp_seed_42"]["best_epoch"] == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in paths.values())

    paths["PREDICTIONS_PATH"].write_text(
        paths["PREDICTIONS_PATH"].read_text().replace("9000.0000000000", "1.0000000000")
    )
    with pytest.raises(evaluation.FinalEvaluationError):
        evaluation.check_outputs()


def test_second_write_is_refused_before_evaluation(tmp_path, monkeypatch):
    paths = _patch_artifact_paths(tmp_path, monkeypatch)
    paths["RESULTS_PATH"].write_text("already evaluated")
    monkeypatch.setattr(
        evaluation,
        "run_final_evaluation",
        lambda: pytest.fail("o teste não pode ser aberto novamente"),
    )
    with pytest.raises(evaluation.FinalEvaluationError, match="não permite sobrescrita"):
        evaluation.run_and_write_final_evaluation()


def test_versioned_final_outputs_pass_offline_contract():
    payload = evaluation.check_outputs()
    assert payload["protocol"]["seed"] == 42
    assert payload["protocol"]["multiple_seeds"] is False
    assert payload["test_access_count"] == 1
    assert payload["success_criterion"]["met"] is True
    assert payload["limitations"] == {
        "single_seed_only": True,
        "seed_variability_estimated": False,
    }
