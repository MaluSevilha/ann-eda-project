"""Congela e valida a configuração final escolhida somente na validação.

Execute a partir da raiz do repositório:

    python -m docs.projects.regression.code.final_config --check
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docs.projects.regression.code.model import (
    MLPConfig,
    MLPRegressor,
    count_trainable_parameters,
)
from docs.projects.regression.code.search import EXPERIMENTS, ExperimentConfig
from docs.projects.regression.code.train import TrainingConfig

REGRESSION_DIR = Path(__file__).resolve().parents[1]
FINAL_CONFIG_PATH = REGRESSION_DIR / "final_config.json"
SEARCH_RESULTS_PATH = REGRESSION_DIR / "hyperparameter_search.json"
SPLIT_MANIFEST_PATH = REGRESSION_DIR / "split_manifest.json"

EXPECTED_WINNER_ID = "low_lr_regularized"


class FrozenConfigError(RuntimeError):
    """Indica que a configuração final ou sua procedência divergiu."""


@dataclass(frozen=True)
class FrozenRegressionConfig:
    """Configuração pronta para ser entregue a ``train_model``."""

    selected_experiment_id: str
    model_config: MLPConfig
    training_config: TrainingConfig
    validation_metrics_usd: dict[str, float]
    best_epoch_evidence: int
    stopped_epoch_evidence: int


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FrozenConfigError(f"{label} ausente em {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenConfigError(f"{label} inválido em {path}: {error}") from error
    if not isinstance(payload, dict):
        raise FrozenConfigError(f"{label} deve ser um objeto JSON")
    return payload


def _selected_experiment() -> ExperimentConfig:
    return next(
        experiment
        for experiment in EXPERIMENTS
        if experiment.experiment_id == EXPECTED_WINNER_ID
    )


def _experiment_payload(experiment: ExperimentConfig) -> dict[str, Any]:
    return {
        "experiment_id": experiment.experiment_id,
        "hidden_dims": list(experiment.hidden_dims),
        "learning_rate": experiment.learning_rate,
        "dropout": experiment.dropout,
        "weight_decay": experiment.weight_decay,
        "normalization": experiment.normalization,
    }


def _validate_search(search: dict[str, Any]) -> dict[str, Any]:
    if search.get("schema_version") != 1:
        raise FrozenConfigError("Schema da busca incompatível")
    if search.get("selection_split") != "validation":
        raise FrozenConfigError("A configuração final deve ser escolhida na validação")
    if search.get("selection_metric") != "mae_usd":
        raise FrozenConfigError("A busca deve selecionar por mae_usd")
    if search.get("test_accessed") is not False:
        raise FrozenConfigError("A busca acessou ou não bloqueou explicitamente o teste")

    winner = search.get("winner", {})
    if winner.get("experiment_id") != EXPECTED_WINNER_ID:
        raise FrozenConfigError(
            f"Vencedor divergente: esperado {EXPECTED_WINNER_ID!r}, "
            f"observado {winner.get('experiment_id')!r}"
        )
    results = search.get("results")
    if not isinstance(results, list) or not results:
        raise FrozenConfigError("A busca não contém resultados")
    selected = results[0]
    if selected.get("status") != "success":
        raise FrozenConfigError("A primeira linha da busca não terminou com sucesso")
    experiment = _selected_experiment()
    if selected.get("experiment") != _experiment_payload(experiment):
        raise FrozenConfigError("A configuração da primeira linha diverge da grade declarada")
    metrics = selected.get("validation_metrics_usd", {})
    if winner.get("mae_usd") != metrics.get("mae_usd"):
        raise FrozenConfigError("O MAE do vencedor diverge da primeira linha da busca")

    expected_protocol = {
        "seed": 42,
        "target_mode": "log1p",
        "loss": "Huber(delta=1)",
        "optimizer": "Adam",
        "batch_size": 256,
        "max_epochs": 200,
        "patience": 15,
        "device": "cpu",
    }
    if search.get("fixed_protocol") != expected_protocol:
        raise FrozenConfigError("O protocolo fixo da busca divergiu")
    return selected


def _validate_manifest(manifest: dict[str, Any]) -> int:
    preprocessing = manifest.get("preprocessing", {})
    test_policy = manifest.get("test_policy", {})
    if preprocessing.get("fit_split") != "train":
        raise FrozenConfigError("O pré-processador não está contratado para ajuste no treino")
    feature_count = preprocessing.get("feature_count")
    if feature_count != 420:
        raise FrozenConfigError(f"Número de features divergente: {feature_count}")
    if test_policy != {
        "locked": True,
        "transformed": False,
        "target_statistics_computed": False,
        "predictive_metrics_computed": False,
    }:
        raise FrozenConfigError("O manifesto não mantém o teste integralmente bloqueado")
    return feature_count


def build_frozen_payload(
    search: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    """Deriva o contrato final da linha vencedora e do manifesto dos dados."""

    selected = _validate_search(search)
    input_dim = _validate_manifest(manifest)
    experiment = _selected_experiment()
    model_config, training_config = experiment.build_configs(input_dim)
    parameter_count = count_trainable_parameters(MLPRegressor(model_config))
    if selected.get("parameter_count") != parameter_count:
        raise FrozenConfigError("A contagem de parâmetros diverge da linha vencedora")

    metrics = selected["validation_metrics_usd"]
    return {
        "schema_version": 1,
        "status": "frozen",
        "selection": {
            "source_artifact": "hyperparameter_search.json",
            "source_schema_version": search["schema_version"],
            "source_rank": 1,
            "experiment_id": experiment.experiment_id,
            "selection_split": "validation",
            "selection_metric": "mae_usd",
            "test_accessed": False,
            "validation_metrics_usd": {
                "mae_usd": metrics["mae_usd"],
                "rmse_usd": metrics["rmse_usd"],
                "r2": metrics["r2"],
            },
            "best_epoch_evidence": selected["best_epoch"],
            "stopped_epoch_evidence": selected["stopped_epoch"],
        },
        "data_contract": {
            "input_features": input_dim,
            "preprocessor_fit_split": "train",
            "target_statistics_fit_split": "train",
            "test_locked": True,
            "test_transformed": False,
            "state_scope": (
                "Estado no congelamento, antes da avaliação final; "
                "os acessos ao teste estão em final_test_results.json."
            ),
        },
        "model": {
            "input_dim": model_config.input_dim,
            "hidden_dims": list(model_config.hidden_dims),
            "activation": "ReLU",
            "dropout": model_config.dropout,
            "normalization": model_config.normalization,
            "output": "1 linear unit",
            "hidden_initialization": "kaiming_normal",
            "output_initialization": "xavier_uniform",
            "parameter_count": parameter_count,
        },
        "target": {
            "transformation": training_config.target_mode,
            "standardized": True,
            "fit_split": "train",
            "metrics_scale": "original_usd",
        },
        "training": {
            "loss": "Huber",
            "huber_delta": training_config.huber_delta,
            "optimizer": "Adam",
            "learning_rate": training_config.learning_rate,
            "weight_decay": training_config.weight_decay,
            "batch_size": training_config.batch_size,
            "validation_batch_size": training_config.validation_batch_size,
            "max_epochs": training_config.max_epochs,
            "patience": training_config.patience,
            "min_delta_mae_usd": training_config.min_delta_mae_usd,
            "seed": training_config.seed,
            "device": training_config.device,
            "deterministic_algorithms": True,
            "early_stopping_metric": "validation_mae_usd",
            "restore_best_weights": True,
        },
    }


def _to_frozen_config(payload: dict[str, Any]) -> FrozenRegressionConfig:
    model = payload["model"]
    training = payload["training"]
    selection = payload["selection"]
    return FrozenRegressionConfig(
        selected_experiment_id=selection["experiment_id"],
        model_config=MLPConfig(
            input_dim=model["input_dim"],
            hidden_dims=tuple(model["hidden_dims"]),
            dropout=model["dropout"],
            normalization=model["normalization"],
        ),
        training_config=TrainingConfig(
            seed=training["seed"],
            learning_rate=training["learning_rate"],
            batch_size=training["batch_size"],
            validation_batch_size=training["validation_batch_size"],
            max_epochs=training["max_epochs"],
            patience=training["patience"],
            min_delta_mae_usd=training["min_delta_mae_usd"],
            huber_delta=training["huber_delta"],
            weight_decay=training["weight_decay"],
            target_mode=payload["target"]["transformation"],
            device=training["device"],
        ),
        validation_metrics_usd=dict(selection["validation_metrics_usd"]),
        best_epoch_evidence=selection["best_epoch_evidence"],
        stopped_epoch_evidence=selection["stopped_epoch_evidence"],
    )


def expected_payload(
    search_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    search_path = SEARCH_RESULTS_PATH if search_path is None else search_path
    manifest_path = SPLIT_MANIFEST_PATH if manifest_path is None else manifest_path
    return build_frozen_payload(
        _read_json(search_path, "Resultado da busca"),
        _read_json(manifest_path, "Manifesto dos dados"),
    )


def write_frozen_config(
    path: Path | None = None,
    search_path: Path | None = None,
    manifest_path: Path | None = None,
) -> None:
    path = FINAL_CONFIG_PATH if path is None else path
    payload = expected_payload(search_path, manifest_path)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_frozen_config(
    path: Path | None = None,
    search_path: Path | None = None,
    manifest_path: Path | None = None,
) -> FrozenRegressionConfig:
    """Carrega a fonte única de configuração e verifica sua procedência."""

    path = FINAL_CONFIG_PATH if path is None else path
    observed = _read_json(path, "Configuração final")
    expected = expected_payload(search_path, manifest_path)
    if observed != expected:
        raise FrozenConfigError(
            f"Configuração final divergente em {path}. "
            "Use --write-config somente após revisar a causa."
        )
    return _to_frozen_config(observed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-config", action="store_true")
    action.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.write_config:
        write_frozen_config()
        print(f"Configuração final congelada em {FINAL_CONFIG_PATH}")
        return
    frozen = load_frozen_config()
    print(
        f"Configuração final válida: experimento={frozen.selected_experiment_id}, "
        f"seed={frozen.training_config.seed}, "
        f"MAE de seleção=US$ {frozen.validation_metrics_usd['mae_usd']:,.2f}"
    )


if __name__ == "__main__":
    main()
