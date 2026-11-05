from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from torch import nn

from docs.projects.regression.code.model import MLPConfig, MLPRegressor
from docs.projects.regression.code.search import (
    EXPERIMENTS,
    ExperimentConfig,
    SearchContractError,
    check_outputs,
    render_table,
    run_search,
    study_payload,
    write_outputs,
)
from docs.projects.regression.code.train import EpochMetrics


def test_model_regularization_order_and_validation():
    model = MLPRegressor(
        MLPConfig(
            input_dim=4,
            hidden_dims=(3,),
            dropout=0.2,
            normalization="layer",
        )
    )
    assert [type(layer) for layer in model.network] == [
        nn.Linear,
        nn.LayerNorm,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
    ]
    with pytest.raises(ValueError, match="dropout"):
        MLPConfig(dropout=1.0)
    with pytest.raises(ValueError, match="normalization"):
        MLPConfig(normalization="batch")  # type: ignore[arg-type]


def test_declared_grid_has_twelve_unique_configs_and_required_techniques():
    assert len(EXPERIMENTS) == 12
    identifiers = {item.experiment_id for item in EXPERIMENTS}
    assert len(identifiers) == 12
    assert "reference" in identifiers
    assert any(item.dropout > 0 for item in EXPERIMENTS)
    assert any(item.weight_decay > 0 for item in EXPERIMENTS)
    assert any(item.normalization == "layer" for item in EXPERIMENTS)
    assert {item.build_configs(420)[1].seed for item in EXPERIMENTS} == {42}


def _fake_training_result(model_config, training_config, mae):
    metrics = {"mae_usd": mae, "rmse_usd": mae * 1.5, "r2": 0.8}
    history = (
        EpochMetrics(1, 0.2, 0.3, mae + 10),
        EpochMetrics(2, 0.1, 0.2, mae),
    )
    return SimpleNamespace(
        model=MLPRegressor(model_config),
        target_transformer=None,
        history=history,
        best_epoch=2,
        stopped_epoch=2,
        best_metrics_usd=metrics,
        parameter_count=sum(
            parameter.numel()
            for parameter in MLPRegressor(model_config).parameters()
        ),
        training_rows=10,
        model_config=model_config,
        training_config=training_config,
    )


def _study(monkeypatch, *, fail_id=None):
    prepared = SimpleNamespace(X_train=SimpleNamespace(shape=(10, 4)))
    order = {item.experiment_id: position for position, item in enumerate(EXPERIMENTS)}

    def fake_train(received_prepared, model_config, training_config):
        assert received_prepared is prepared
        experiment = next(
            item
            for item in EXPERIMENTS
            if item.hidden_dims == model_config.hidden_dims
            and item.dropout == model_config.dropout
            and item.normalization == model_config.normalization
            and item.learning_rate == training_config.learning_rate
            and item.weight_decay == training_config.weight_decay
        )
        if experiment.experiment_id == fail_id:
            raise RuntimeError("falha simulada")
        return _fake_training_result(
            model_config,
            training_config,
            mae=6_000.0 + order[experiment.experiment_id],
        )

    monkeypatch.setattr("docs.projects.regression.code.search.train_model", fake_train)
    ticks = iter(float(value) for value in range(100))
    return run_search(prepared, clock=lambda: next(ticks))


def test_search_orders_selects_by_mae_and_keeps_failures(monkeypatch):
    study = _study(monkeypatch, fail_id="low_lr_regularized")
    assert study.winner_id == "reference"
    assert [item.mae_usd for item in study.results[:-1]] == sorted(
        item.mae_usd for item in study.results[:-1]
    )
    failed = study.results[-1]
    assert failed.status == "failed"
    assert failed.error_type == "RuntimeError"
    assert failed.error_message == "falha simulada"


def test_grid_validation(monkeypatch):
    with pytest.raises(ValueError, match="oito"):
        run_search(SimpleNamespace(X_train=SimpleNamespace(shape=(10, 4))), EXPERIMENTS[:7])
    duplicated = (*EXPERIMENTS[:8], replace(EXPERIMENTS[0]))
    with pytest.raises(ValueError, match="únicos"):
        run_search(SimpleNamespace(X_train=SimpleNamespace(shape=(10, 4))), duplicated)


def test_search_prepares_data_once_and_breaks_mae_ties_by_id(monkeypatch):
    prepared = SimpleNamespace(X_train=SimpleNamespace(shape=(10, 4)))
    calls = {"load": 0, "prepare": 0, "train": 0}

    def fake_load():
        calls["load"] += 1
        return object()

    def fake_prepare(splits):
        assert splits is not None
        calls["prepare"] += 1
        return prepared

    def fake_train(received, model_config, training_config):
        assert received is prepared
        calls["train"] += 1
        return _fake_training_result(model_config, training_config, mae=6_000.0)

    monkeypatch.setattr("docs.projects.regression.code.search.load_development_splits", fake_load)
    monkeypatch.setattr("docs.projects.regression.code.search.prepare_development_data", fake_prepare)
    monkeypatch.setattr("docs.projects.regression.code.search.train_model", fake_train)
    ticks = iter(float(value) for value in range(100))
    experiments = EXPERIMENTS[:8]
    study = run_search(experiments=experiments, clock=lambda: next(ticks))
    assert calls == {"load": 1, "prepare": 1, "train": 8}
    assert study.winner_id == min(item.experiment_id for item in experiments)


def test_artifacts_are_validation_only_and_detect_divergence(tmp_path, monkeypatch):
    study = _study(monkeypatch)
    payload = study_payload(study)
    assert payload["selection_split"] == "validation"
    assert payload["test_accessed"] is False
    assert len(payload["declared_grid"]) == 12
    assert payload["winner"]["experiment_id"] == "reference"
    assert render_table(payload).count("\n") == 14

    results_path = tmp_path / "search.json"
    table_path = tmp_path / "search.md"
    figure_path = tmp_path / "search.png"
    monkeypatch.setattr("docs.projects.regression.code.search.RESULTS_PATH", results_path)
    monkeypatch.setattr("docs.projects.regression.code.search.TABLE_PATH", table_path)
    monkeypatch.setattr("docs.projects.regression.code.search.FIGURE_PATH", figure_path)
    write_outputs(study)
    check_outputs(study)
    assert figure_path.stat().st_size > 0

    observed = json.loads(results_path.read_text(encoding="utf-8"))
    observed["test_accessed"] = True
    results_path.write_text(json.dumps(observed), encoding="utf-8")
    with pytest.raises(SearchContractError, match="Busca divergente"):
        check_outputs(study)


def test_duration_is_recorded_but_excluded_from_deterministic_check(
    tmp_path, monkeypatch
):
    study = _study(monkeypatch)
    results_path = tmp_path / "search.json"
    table_path = tmp_path / "search.md"
    figure_path = tmp_path / "search.png"
    monkeypatch.setattr("docs.projects.regression.code.search.RESULTS_PATH", results_path)
    monkeypatch.setattr("docs.projects.regression.code.search.TABLE_PATH", table_path)
    monkeypatch.setattr("docs.projects.regression.code.search.FIGURE_PATH", figure_path)
    write_outputs(study)
    observed = json.loads(results_path.read_text(encoding="utf-8"))
    observed["results"][0]["duration_seconds"] = 999.0
    results_path.write_text(json.dumps(observed), encoding="utf-8")
    table_path.write_text(render_table(observed), encoding="utf-8")
    check_outputs(study)


def test_experiment_validation():
    with pytest.raises(ValueError, match="experiment_id"):
        ExperimentConfig("")
    with pytest.raises(ValueError, match="learning_rate"):
        ExperimentConfig("invalid", learning_rate=0)
