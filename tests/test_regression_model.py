from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy import sparse
from torch import nn

from docs.projects.regression.code.model import (
    MLPConfig,
    MLPRegressor,
    count_trainable_parameters,
)
from docs.projects.regression.code.train import (
    TargetTransformer,
    TrainingConfig,
    check_outputs,
    predict_usd,
    result_payload,
    train_model,
    write_outputs,
)


@pytest.mark.parametrize("mode", ["identity", "log1p"])
def test_target_transformer_round_trip(mode):
    values = np.asarray([10_000.0, 40_000.0, 100_000.0])
    transformer = TargetTransformer(mode).fit(values)
    restored = transformer.inverse_transform(transformer.transform(values))
    assert restored == pytest.approx(values, rel=1e-6)


def test_target_transformer_rejects_invalid_values():
    with pytest.raises(ValueError, match="não negativo"):
        TargetTransformer("log1p").fit(np.asarray([-1.0, 2.0]))
    with pytest.raises(ValueError, match="desvio positivo"):
        TargetTransformer("identity").fit(np.asarray([2.0, 2.0]))
    with pytest.raises(RuntimeError, match="não foi ajustado"):
        TargetTransformer().transform(np.asarray([1.0]))


def test_reference_model_shape_parameters_and_no_regularization():
    config = MLPConfig(input_dim=420, hidden_dims=(128, 64))
    torch.manual_seed(42)
    model = MLPRegressor(config)
    assert model(torch.zeros(7, 420)).shape == (7, 1)
    assert count_trainable_parameters(model) == 62_209
    assert not any(
        isinstance(layer, (nn.Dropout, nn.BatchNorm1d, nn.LayerNorm))
        for layer in model.modules()
    )


def test_initialization_is_reproducible_and_biases_are_zero():
    config = MLPConfig(input_dim=5, hidden_dims=(4, 2))
    torch.manual_seed(42)
    first = MLPRegressor(config)
    torch.manual_seed(42)
    second = MLPRegressor(config)
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters()
    ):
        assert torch.equal(first_parameter, second_parameter)
    linear_layers = [layer for layer in first.modules() if isinstance(layer, nn.Linear)]
    assert all(torch.count_nonzero(layer.bias) == 0 for layer in linear_layers)


def _tiny_prepared():
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(64, 4)).astype(np.float32)
    X_validation = rng.normal(size=(24, 4)).astype(np.float32)
    train_signal = 40_000 + 3_000 * X_train[:, 0] - 1_500 * X_train[:, 1]
    validation_signal = (
        40_000 + 3_000 * X_validation[:, 0] - 1_500 * X_validation[:, 1]
    )
    return SimpleNamespace(
        X_train=sparse.csr_matrix(X_train),
        X_validation=sparse.csr_matrix(X_validation),
        y_train=train_signal.astype(np.float32),
        y_validation=validation_signal.astype(np.float32),
    )


def _tiny_configs():
    return (
        MLPConfig(input_dim=4, hidden_dims=(8, 4)),
        TrainingConfig(
            seed=42,
            learning_rate=5e-3,
            batch_size=16,
            validation_batch_size=24,
            max_epochs=8,
            patience=3,
            min_delta_mae_usd=0.0,
            device="cpu",
        ),
    )


def test_short_training_is_reproducible_and_restores_best_epoch():
    prepared = _tiny_prepared()
    model_config, training_config = _tiny_configs()
    first = train_model(prepared, model_config, training_config)
    second = train_model(prepared, model_config, training_config)
    assert first.history == second.history
    assert first.best_epoch == second.best_epoch
    assert first.best_metrics_usd == pytest.approx(second.best_metrics_usd)
    assert first.stopped_epoch <= training_config.max_epochs
    best_recorded = min(item.validation_mae_usd for item in first.history)
    assert first.best_metrics_usd["mae_usd"] == pytest.approx(best_recorded)
    first_predictions = predict_usd(
        first.model,
        prepared.X_validation,
        first.target_transformer,
    )
    second_predictions = predict_usd(
        second.model,
        prepared.X_validation,
        second.target_transformer,
    )
    assert first_predictions == pytest.approx(second_predictions)


def test_reference_training_config_has_no_regularization():
    config = TrainingConfig()
    assert config.weight_decay == 0.0
    assert config.target_mode == "log1p"
    assert config.device == "cpu"


def test_payload_is_validation_only_and_uses_dollar_metrics(monkeypatch):
    prepared = _tiny_prepared()
    model_config, training_config = _tiny_configs()
    result = train_model(prepared, model_config, training_config)
    monkeypatch.setattr(
        "docs.projects.regression.code.train._baseline_reference",
        lambda: {
            "best_baseline_mae_usd": 7_813.89,
            "mlp_success_threshold_mae_usd": 6_641.81,
        },
    )
    payload = result_payload(result)
    assert payload["selection_split"] == "validation"
    assert payload["test_accessed"] is False
    assert payload["model"]["dropout"] == 0.0
    assert payload["training"]["weight_decay"] == 0.0
    assert payload["best_validation_metrics_usd"]["mae_usd"] > 0


def test_check_outputs_detects_divergence(tmp_path, monkeypatch):
    prepared = _tiny_prepared()
    model_config, training_config = _tiny_configs()
    result = train_model(prepared, model_config, training_config)
    baseline = {
        "best_baseline_mae_usd": 7_813.89,
        "mlp_success_threshold_mae_usd": 6_641.81,
    }
    monkeypatch.setattr(
        "docs.projects.regression.code.train._baseline_reference", lambda: baseline
    )
    results_path = tmp_path / "reference.json"
    figure_path = tmp_path / "figure.png"
    monkeypatch.setattr("docs.projects.regression.code.train.RESULTS_PATH", results_path)
    monkeypatch.setattr("docs.projects.regression.code.train.FIGURE_PATH", figure_path)
    write_outputs(result)
    assert figure_path.exists() and figure_path.stat().st_size > 0
    check_outputs(result)
    payload = results_path.read_text(encoding="utf-8").replace(
        '"test_accessed": false', '"test_accessed": true'
    )
    results_path.write_text(payload, encoding="utf-8")
    with pytest.raises(RuntimeError, match="Treinamento divergente"):
        check_outputs(result)


def test_invalid_training_configuration_and_input_width():
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA"):
            TrainingConfig(device="cuda")
    prepared = _tiny_prepared()
    with pytest.raises(ValueError, match="dados têm 4 colunas"):
        train_model(
            prepared,
            MLPConfig(input_dim=5, hidden_dims=(4,)),
            replace(TrainingConfig(), max_epochs=1),
        )
