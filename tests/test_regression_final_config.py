from __future__ import annotations

import json
from pathlib import Path

import pytest

from docs.projects.regression.code.final_config import (
    EXPECTED_WINNER_ID,
    FINAL_CONFIG_PATH,
    SEARCH_RESULTS_PATH,
    SPLIT_MANIFEST_PATH,
    FrozenConfigError,
    build_frozen_payload,
    load_frozen_config,
    write_frozen_config,
)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_versioned_config_matches_winning_search_row():
    search = _read(SEARCH_RESULTS_PATH)
    manifest = _read(SPLIT_MANIFEST_PATH)
    observed = _read(FINAL_CONFIG_PATH)
    assert observed == build_frozen_payload(search, manifest)
    assert observed["selection"]["source_rank"] == 1
    assert observed["selection"]["experiment_id"] == EXPECTED_WINNER_ID
    assert observed["selection"]["validation_metrics_usd"] == {
        "mae_usd": 6265.05,
        "rmse_usd": 9821.37,
        "r2": 0.87271,
    }


def test_loader_builds_ready_model_and_training_configs():
    frozen = load_frozen_config()
    assert frozen.selected_experiment_id == "low_lr_regularized"
    assert frozen.model_config.input_dim == 420
    assert frozen.model_config.hidden_dims == (128, 64)
    assert frozen.model_config.dropout == 0.1
    assert frozen.model_config.normalization == "none"
    assert frozen.training_config.seed == 42
    assert frozen.training_config.learning_rate == 3e-4
    assert frozen.training_config.weight_decay == 1e-5
    assert frozen.training_config.batch_size == 256
    assert frozen.training_config.validation_batch_size == 2048
    assert frozen.training_config.max_epochs == 200
    assert frozen.training_config.patience == 15
    assert frozen.training_config.min_delta_mae_usd == 1.0
    assert frozen.training_config.huber_delta == 1.0
    assert frozen.training_config.target_mode == "log1p"
    assert frozen.training_config.device == "cpu"
    assert frozen.best_epoch_evidence == 31
    assert frozen.stopped_epoch_evidence == 46


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(test_accessed=True), "teste"),
        (lambda payload: payload.update(selection_split="test"), "validação"),
        (
            lambda payload: payload["winner"].update(experiment_id="reference"),
            "Vencedor divergente",
        ),
        (
            lambda payload: payload["winner"].update(mae_usd=1.0),
            "MAE do vencedor",
        ),
    ],
)
def test_rejects_invalid_search_provenance(mutation, message):
    search = _read(SEARCH_RESULTS_PATH)
    mutation(search)
    with pytest.raises(FrozenConfigError, match=message):
        build_frozen_payload(search, _read(SPLIT_MANIFEST_PATH))


def test_loader_detects_tampered_frozen_file(tmp_path):
    frozen_path = tmp_path / "final.json"
    write_frozen_config(frozen_path)
    payload = _read(frozen_path)
    payload["training"]["learning_rate"] = 0.1
    _write(frozen_path, payload)
    with pytest.raises(FrozenConfigError, match="Configuração final divergente"):
        load_frozen_config(frozen_path)


def test_frozen_artifact_contains_no_test_results_or_identifiers():
    payload = _read(FINAL_CONFIG_PATH)
    serialized = json.dumps(payload).lower()
    assert "test_metrics" not in serialized
    assert "test_predictions" not in serialized
    assert "program_id" not in serialized
    assert payload["selection"]["test_accessed"] is False
    assert payload["data_contract"]["test_locked"] is True
    assert payload["data_contract"]["test_transformed"] is False
    assert "final_test_results.json" in payload["data_contract"]["state_scope"]


def test_seed_cannot_be_silently_overridden(tmp_path):
    frozen_path = tmp_path / "final.json"
    write_frozen_config(frozen_path)
    payload = _read(frozen_path)
    payload["training"]["seed"] = 43
    _write(frozen_path, payload)
    with pytest.raises(FrozenConfigError, match="Configuração final divergente"):
        load_frozen_config(frozen_path)
