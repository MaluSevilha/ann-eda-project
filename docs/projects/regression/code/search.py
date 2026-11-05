"""Busca reproduzível de regularização e hiperparâmetros da MLP.

Execute a partir da raiz do repositório:

    python -m docs.projects.regression.code.search --check
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ann-eda-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from docs.projects.regression.code.data import (
    PreparedDevelopmentData,
    load_development_splits,
    prepare_development_data,
)
from docs.projects.regression.code.model import MLPConfig
from docs.projects.regression.code.train import (
    TrainingConfig,
    TrainingResult,
    train_model,
)

REGRESSION_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = REGRESSION_DIR / "hyperparameter_search.json"
TABLE_PATH = REGRESSION_DIR / "tables" / "hyperparameter_search.md"
FIGURE_PATH = REGRESSION_DIR / "figures" / "fig02_regularization_search.png"


class SearchContractError(RuntimeError):
    """Indica divergência entre a busca e os artefatos versionados."""


@dataclass(frozen=True)
class ExperimentConfig:
    """Uma configuração declarada antes da busca."""

    experiment_id: str
    hidden_dims: tuple[int, ...] = (128, 64)
    learning_rate: float = 1e-3
    dropout: float = 0.0
    weight_decay: float = 0.0
    normalization: Literal["none", "layer"] = "none"

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id não pode ser vazio")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate deve ser positivo e weight_decay não negativo")
        MLPConfig(
            input_dim=1,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            normalization=self.normalization,
        )

    def build_configs(self, input_dim: int) -> tuple[MLPConfig, TrainingConfig]:
        return (
            MLPConfig(
                input_dim=input_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
                normalization=self.normalization,
            ),
            TrainingConfig(
                seed=42,
                learning_rate=self.learning_rate,
                batch_size=256,
                max_epochs=200,
                patience=15,
                weight_decay=self.weight_decay,
                target_mode="log1p",
                device="cpu",
            ),
        )


EXPERIMENTS: tuple[ExperimentConfig, ...] = (
    ExperimentConfig("reference"),
    ExperimentConfig("weight_decay_1e-5", weight_decay=1e-5),
    ExperimentConfig("weight_decay_1e-4", weight_decay=1e-4),
    ExperimentConfig("dropout_0.10", dropout=0.10),
    ExperimentConfig("dropout_0.20", dropout=0.20),
    ExperimentConfig("dropout_0.10_wd_1e-5", dropout=0.10, weight_decay=1e-5),
    ExperimentConfig("layernorm", normalization="layer"),
    ExperimentConfig("layernorm_wd_1e-5", normalization="layer", weight_decay=1e-5),
    ExperimentConfig(
        "narrow_regularized",
        hidden_dims=(64, 32),
        dropout=0.10,
        weight_decay=1e-5,
    ),
    ExperimentConfig(
        "wide_regularized",
        hidden_dims=(256, 128),
        dropout=0.10,
        weight_decay=1e-5,
    ),
    ExperimentConfig(
        "deep_regularized",
        hidden_dims=(128, 64, 32),
        dropout=0.10,
        weight_decay=1e-5,
    ),
    ExperimentConfig(
        "low_lr_regularized",
        learning_rate=3e-4,
        dropout=0.10,
        weight_decay=1e-5,
    ),
)


@dataclass
class SearchResult:
    experiment: ExperimentConfig
    status: Literal["success", "failed"]
    duration_seconds: float
    training_result: TrainingResult | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def mae_usd(self) -> float:
        if self.training_result is None:
            return float("inf")
        return self.training_result.best_metrics_usd["mae_usd"]


@dataclass(frozen=True)
class HyperparameterStudy:
    results: tuple[SearchResult, ...]
    winner_id: str

    @property
    def winner(self) -> SearchResult:
        return next(
            item for item in self.results if item.experiment.experiment_id == self.winner_id
        )


def _validate_experiments(experiments: Sequence[ExperimentConfig]) -> None:
    if len(experiments) < 8:
        raise ValueError("A busca exige pelo menos oito configurações")
    identifiers = [item.experiment_id for item in experiments]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Os experiment_id da busca devem ser únicos")


def run_search(
    prepared: PreparedDevelopmentData | None = None,
    experiments: Sequence[ExperimentConfig] = EXPERIMENTS,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> HyperparameterStudy:
    """Executa a grade no mesmo desenvolvimento e seleciona apenas pela validação."""

    _validate_experiments(experiments)
    if prepared is None:
        prepared = prepare_development_data(load_development_splits())
    input_dim = prepared.X_train.shape[1]
    results: list[SearchResult] = []
    for experiment in experiments:
        started = clock()
        try:
            model_config, training_config = experiment.build_configs(input_dim)
            training_result = train_model(prepared, model_config, training_config)
            results.append(
                SearchResult(
                    experiment=experiment,
                    status="success",
                    duration_seconds=round(clock() - started, 3),
                    training_result=training_result,
                )
            )
        except Exception as error:  # a falha precisa permanecer no artefato
            results.append(
                SearchResult(
                    experiment=experiment,
                    status="failed",
                    duration_seconds=round(clock() - started, 3),
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            )

    successes = [item for item in results if item.status == "success"]
    if len(successes) < 8:
        raise RuntimeError(
            f"A busca produziu apenas {len(successes)} configurações válidas; são exigidas 8"
        )
    ordered = sorted(
        successes,
        key=lambda item: (item.mae_usd, item.experiment.experiment_id),
    ) + sorted(
        (item for item in results if item.status == "failed"),
        key=lambda item: item.experiment.experiment_id,
    )
    return HyperparameterStudy(
        results=tuple(ordered),
        winner_id=ordered[0].experiment.experiment_id,
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


def _result_payload(result: SearchResult) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment": _experiment_payload(result.experiment),
        "status": result.status,
        "duration_seconds": result.duration_seconds,
    }
    if result.training_result is None:
        payload["error"] = {
            "type": result.error_type,
            "message": result.error_message,
        }
        return payload

    training = result.training_result
    payload.update(
        {
            "parameter_count": training.parameter_count,
            "best_epoch": training.best_epoch,
            "stopped_epoch": training.stopped_epoch,
            "validation_metrics_usd": {
                "mae_usd": round(training.best_metrics_usd["mae_usd"], 2),
                "rmse_usd": round(training.best_metrics_usd["rmse_usd"], 2),
                "r2": round(training.best_metrics_usd["r2"], 6),
            },
            "history": [
                {
                    "epoch": item.epoch,
                    "train_loss": round(item.train_loss, 8),
                    "validation_loss": round(item.validation_loss, 8),
                    "validation_mae_usd": round(item.validation_mae_usd, 2),
                }
                for item in training.history
            ],
        }
    )
    return payload


def study_payload(study: HyperparameterStudy) -> dict[str, Any]:
    winner = study.winner
    if winner.training_result is None:
        raise RuntimeError("O vencedor da busca não possui resultado de treinamento")
    return {
        "schema_version": 1,
        "selection_split": "validation",
        "selection_metric": "mae_usd",
        "test_accessed": False,
        "fixed_protocol": {
            "seed": 42,
            "target_mode": "log1p",
            "loss": "Huber(delta=1)",
            "optimizer": "Adam",
            "batch_size": 256,
            "max_epochs": 200,
            "patience": 15,
            "device": "cpu",
        },
        "declared_grid": [_experiment_payload(item) for item in EXPERIMENTS],
        "results": [_result_payload(item) for item in study.results],
        "winner": {
            "experiment_id": study.winner_id,
            "mae_usd": round(winner.training_result.best_metrics_usd["mae_usd"], 2),
        },
    }


def _money(value: float) -> str:
    formatted = f"{value:,.2f}"
    return "US$ " + formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def render_table(payload: dict[str, Any]) -> str:
    lines = [
        "| Ordem | Configuração | Status | MAE | RMSE | R² | Parâmetros | Melhor época | Parada | Tempo (s) |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for position, result in enumerate(payload["results"], start=1):
        if result["status"] == "failed":
            error = result["error"]
            lines.append(
                f"| {position} | `{result['experiment']['experiment_id']}` | "
                f"falhou: {error['type']}: {error['message']} | — | — | — | — | — | — | "
                f"{result['duration_seconds']:.3f} |"
            )
            continue
        metrics = result["validation_metrics_usd"]
        parameters = f"{result['parameter_count']:,}".replace(",", ".")
        lines.append(
            f"| {position} | `{result['experiment']['experiment_id']}` | ok | "
            f"{_money(metrics['mae_usd'])} | {_money(metrics['rmse_usd'])} | "
            f"{metrics['r2']:.6f} | {parameters}"
            + f" | {result['best_epoch']} | {result['stopped_epoch']} | "
            f"{result['duration_seconds']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def save_figure(payload: dict[str, Any], path: Path) -> None:
    successful = [item for item in payload["results"] if item["status"] == "success"]
    identifiers = [item["experiment"]["experiment_id"] for item in successful]
    maes = [item["validation_metrics_usd"]["mae_usd"] for item in successful]
    colors = [
        "#b2182b" if identifier == payload["winner"]["experiment_id"] else "#2166ac"
        for identifier in identifiers
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    positions = range(len(identifiers))
    axes[0].barh(list(positions), maes, color=colors)
    axes[0].set_yticks(list(positions), labels=identifiers)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="MAE de validação (US$)", title="Busca ordenada pela métrica principal")
    for position, value in zip(positions, maes):
        axes[0].text(value + 8, position, f"{value:,.0f}", va="center", fontsize=8)

    representatives = {
        "reference": "Sem regularização",
        "weight_decay_1e-4": "Weight decay",
        "dropout_0.10": "Dropout",
        "layernorm": "LayerNorm",
    }
    by_id = {item["experiment"]["experiment_id"]: item for item in successful}
    for identifier, label in representatives.items():
        if identifier not in by_id:
            continue
        history = by_id[identifier]["history"]
        axes[1].plot(
            [item["epoch"] for item in history],
            [item["validation_mae_usd"] for item in history],
            label=label,
        )
    axes[1].set(
        xlabel="Época",
        ylabel="MAE de validação (US$)",
        title="Comparação direta das regularizações",
    )
    axes[1].legend()
    fig.suptitle("Figura 2: Regularização e busca de hiperparâmetros", fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_outputs(study: HyperparameterStudy) -> None:
    payload = study_payload(study)
    RESULTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(render_table(payload), encoding="utf-8")
    save_figure(payload, FIGURE_PATH)


def _without_durations(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = json.loads(json.dumps(payload))
    for result in cleaned["results"]:
        result.pop("duration_seconds", None)
    return cleaned


def check_outputs(study: HyperparameterStudy) -> None:
    if not RESULTS_PATH.exists():
        raise SearchContractError(f"Resultado ausente em {RESULTS_PATH}. Use --write-results.")
    observed = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    expected = study_payload(study)
    for result in observed.get("results", []):
        duration = result.get("duration_seconds")
        if not isinstance(duration, (int, float)) or duration < 0:
            raise SearchContractError("duration_seconds ausente ou inválido")
    if _without_durations(observed) != _without_durations(expected):
        raise SearchContractError(
            f"Busca divergente em {RESULTS_PATH}. Use --write-results somente após revisar a causa."
        )
    if not TABLE_PATH.exists() or TABLE_PATH.read_text(encoding="utf-8") != render_table(observed):
        raise SearchContractError(f"Tabela ausente ou divergente em {TABLE_PATH}")
    if not FIGURE_PATH.exists() or FIGURE_PATH.stat().st_size == 0:
        raise SearchContractError(f"Figura ausente ou vazia em {FIGURE_PATH}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-results", action="store_true")
    action.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    study = run_search()
    if args.write_results:
        write_outputs(study)
        print(f"Busca gravada em {RESULTS_PATH}, {TABLE_PATH} e {FIGURE_PATH}")
    else:
        check_outputs(study)
        print(
            f"Busca válida: vencedor={study.winner_id}, "
            f"MAE=US$ {study.winner.mae_usd:,.2f}"
        )


if __name__ == "__main__":
    main()
