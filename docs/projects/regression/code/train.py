"""Treinamento reproduzível da MLP de referência, sem tocar no teste.

Execute a partir da raiz do repositório:

    python -m docs.projects.regression.code.train --check
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ann-eda-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from docs.projects.regression.code.baselines import (
    RESULTS_PATH as BASELINE_RESULTS_PATH,
    regression_metrics,
)
from docs.projects.regression.code.data import (
    PreparedDevelopmentData,
    load_development_splits,
    prepare_development_data,
)
from docs.projects.regression.code.model import (
    MLPConfig,
    MLPRegressor,
    count_trainable_parameters,
)

REGRESSION_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = REGRESSION_DIR / "reference_training.json"
FIGURE_PATH = REGRESSION_DIR / "figures/fig01_training_curves.png"


class TrainingContractError(RuntimeError):
    """Indica divergência entre a execução e os artefatos versionados."""


class TargetTransformer:
    """Transforma o alvo e padroniza usando apenas estatísticas do treino."""

    def __init__(self, mode: Literal["identity", "log1p"] = "log1p"):
        if mode not in {"identity", "log1p"}:
            raise ValueError(f"Modo de alvo inválido: {mode}")
        self.mode = mode

    def _base_transform(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if not np.isfinite(array).all():
            raise ValueError("O alvo deve conter apenas valores finitos")
        if self.mode == "log1p":
            if np.any(array < 0):
                raise ValueError("log1p exige alvo não negativo")
            return np.log1p(array)
        return array

    def fit(self, values: np.ndarray) -> "TargetTransformer":
        transformed = self._base_transform(values)
        if transformed.size == 0:
            raise ValueError("Não é possível ajustar o alvo vazio")
        self.mean_ = float(np.mean(transformed))
        self.scale_ = float(np.std(transformed))
        if not np.isfinite(self.scale_) or self.scale_ <= 0:
            raise ValueError("O alvo transformado precisa ter desvio positivo")
        return self

    def _check_fitted(self) -> None:
        if not hasattr(self, "mean_") or not hasattr(self, "scale_"):
            raise RuntimeError("TargetTransformer ainda não foi ajustado")

    def transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        transformed = self._base_transform(values)
        return ((transformed - self.mean_) / self.scale_).astype(np.float32)

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        self._check_fitted()
        standardized = np.asarray(values, dtype=np.float64)
        transformed = standardized * self.scale_ + self.mean_
        restored = np.expm1(transformed) if self.mode == "log1p" else transformed
        return restored.astype(np.float64)


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    learning_rate: float = 1e-3
    batch_size: int = 256
    validation_batch_size: int = 2048
    max_epochs: int = 200
    patience: int = 15
    min_delta_mae_usd: float = 1.0
    huber_delta: float = 1.0
    weight_decay: float = 0.0
    target_mode: Literal["identity", "log1p"] = "log1p"
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.batch_size <= 0:
            raise ValueError("learning_rate e batch_size devem ser positivos")
        if self.validation_batch_size <= 0 or self.max_epochs <= 0:
            raise ValueError("validation_batch_size e max_epochs devem ser positivos")
        if self.patience <= 0 or self.min_delta_mae_usd < 0:
            raise ValueError("patience deve ser positiva e min_delta não negativa")
        if self.huber_delta <= 0 or self.weight_decay < 0:
            raise ValueError("huber_delta deve ser positivo e weight_decay não negativo")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device deve ser 'cpu' ou 'cuda'")
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA foi solicitada, mas não está disponível")


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_mae_usd: float


@dataclass
class TrainingResult:
    model: MLPRegressor
    target_transformer: TargetTransformer
    history: tuple[EpochMetrics, ...]
    best_epoch: int
    stopped_epoch: int
    best_metrics_usd: dict[str, float]
    parameter_count: int
    training_rows: int
    model_config: MLPConfig
    training_config: TrainingConfig


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _dense_tensor(matrix: sparse.csr_matrix) -> torch.Tensor:
    return torch.from_numpy(matrix.toarray().astype(np.float32, copy=False))


def _predict_standardized(
    model: nn.Module,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    loader = DataLoader(
        TensorDataset(features),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    with torch.no_grad():
        for (batch,) in loader:
            predictions.append(model(batch.to(device)).squeeze(1).cpu().numpy())
    return np.concatenate(predictions)


def predict_usd(
    model: nn.Module,
    features: sparse.csr_matrix | torch.Tensor,
    target_transformer: TargetTransformer,
    *,
    device: str = "cpu",
    batch_size: int = 2048,
) -> np.ndarray:
    """Prediz e sempre retorna a escala original em dólares."""

    tensor = _dense_tensor(features) if sparse.issparse(features) else features.float()
    standardized = _predict_standardized(
        model,
        tensor,
        device=torch.device(device),
        batch_size=batch_size,
    )
    return target_transformer.inverse_transform(standardized)


def train_model(
    prepared: PreparedDevelopmentData,
    model_config: MLPConfig,
    training_config: TrainingConfig,
) -> TrainingResult:
    """Treina no subtreino e restaura a melhor época pelo MAE de validação."""

    if prepared.X_train.shape[1] != model_config.input_dim:
        raise ValueError(
            f"input_dim={model_config.input_dim}, mas os dados têm "
            f"{prepared.X_train.shape[1]} colunas"
        )
    set_reproducible_seed(training_config.seed)
    device = torch.device(training_config.device)
    X_train = _dense_tensor(prepared.X_train)
    X_validation = _dense_tensor(prepared.X_validation)
    transformer = TargetTransformer(training_config.target_mode).fit(prepared.y_train)
    y_train = torch.from_numpy(transformer.transform(prepared.y_train)).reshape(-1, 1)
    y_validation = torch.from_numpy(
        transformer.transform(prepared.y_validation)
    ).reshape(-1, 1)

    generator = torch.Generator().manual_seed(training_config.seed)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=training_config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        TensorDataset(X_validation, y_validation),
        batch_size=training_config.validation_batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = MLPRegressor(model_config).to(device)
    loss_fn = nn.HuberLoss(delta=training_config.huber_delta, reduction="mean")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    history: list[EpochMetrics] = []
    best_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    wait = 0

    for epoch in range(1, training_config.max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_rows = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(features), targets)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(features)
            train_rows += len(features)

        model.eval()
        validation_loss_sum = 0.0
        validation_rows = 0
        with torch.no_grad():
            for features, targets in validation_loader:
                features = features.to(device)
                targets = targets.to(device)
                loss = loss_fn(model(features), targets)
                validation_loss_sum += loss.item() * len(features)
                validation_rows += len(features)
        predictions_usd = predict_usd(
            model,
            X_validation,
            transformer,
            device=training_config.device,
            batch_size=training_config.validation_batch_size,
        )
        validation_mae = regression_metrics(
            prepared.y_validation, predictions_usd
        )["mae_usd"]
        history.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss_sum / train_rows,
                validation_loss=validation_loss_sum / validation_rows,
                validation_mae_usd=validation_mae,
            )
        )

        if validation_mae < best_mae - training_config.min_delta_mae_usd:
            best_mae = validation_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= training_config.patience:
                break

    if best_state is None:
        raise RuntimeError("O treinamento terminou sem registrar uma melhor época")
    model.load_state_dict(best_state)
    final_predictions = predict_usd(
        model,
        X_validation,
        transformer,
        device=training_config.device,
        batch_size=training_config.validation_batch_size,
    )
    return TrainingResult(
        model=model,
        target_transformer=transformer,
        history=tuple(history),
        best_epoch=best_epoch,
        stopped_epoch=history[-1].epoch,
        best_metrics_usd=regression_metrics(prepared.y_validation, final_predictions),
        parameter_count=count_trainable_parameters(model),
        training_rows=len(prepared.y_train),
        model_config=model_config,
        training_config=training_config,
    )


def canonical_configs(input_dim: int = 420) -> tuple[MLPConfig, TrainingConfig]:
    return MLPConfig(input_dim=input_dim, hidden_dims=(128, 64)), TrainingConfig()


def _baseline_reference() -> dict[str, float]:
    payload = json.loads(BASELINE_RESULTS_PATH.read_text(encoding="utf-8"))
    return {
        "best_baseline_mae_usd": float(payload["best_baseline"]["mae_usd"]),
        "mlp_success_threshold_mae_usd": float(
            payload["mlp_success_threshold_mae_usd"]
        ),
    }


def result_payload(result: TrainingResult) -> dict[str, Any]:
    baseline = _baseline_reference()
    return {
        "schema_version": 1,
        "selection_split": "validation",
        "test_accessed": False,
        "device": result.training_config.device,
        "model": {
            "input_dim": result.model_config.input_dim,
            "hidden_dims": list(result.model_config.hidden_dims),
            "activation": "ReLU",
            "output": "1 linear unit",
            "hidden_initialization": "kaiming_normal",
            "output_initialization": "xavier_uniform",
            "parameter_count": result.parameter_count,
            "training_rows": result.training_rows,
            "parameters_per_training_row": round(
                result.parameter_count / result.training_rows, 6
            ),
            "dropout": 0.0,
            "normalization": None,
        },
        "training": asdict(result.training_config),
        "target_transformer": {
            "mode": result.target_transformer.mode,
            "mean_train_only": round(result.target_transformer.mean_, 10),
            "scale_train_only": round(result.target_transformer.scale_, 10),
        },
        "best_epoch": result.best_epoch,
        "stopped_epoch": result.stopped_epoch,
        "best_validation_metrics_usd": {
            "mae_usd": round(result.best_metrics_usd["mae_usd"], 2),
            "rmse_usd": round(result.best_metrics_usd["rmse_usd"], 2),
            "r2": round(result.best_metrics_usd["r2"], 6),
        },
        "baseline_reference": baseline,
        "history": [
            {
                "epoch": item.epoch,
                "train_loss": round(item.train_loss, 8),
                "validation_loss": round(item.validation_loss, 8),
                "validation_mae_usd": round(item.validation_mae_usd, 2),
            }
            for item in result.history
        ],
    }


def save_training_figure(payload: dict[str, Any], path: Path) -> None:
    history = payload["history"]
    epochs = [item["epoch"] for item in history]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(epochs, [item["train_loss"] for item in history], label="Treino")
    axes[0].plot(
        epochs,
        [item["validation_loss"] for item in history],
        label="Validação",
    )
    axes[0].axvline(
        payload["best_epoch"], color="#b2182b", linestyle="--", label="Melhor época"
    )
    axes[0].set(
        xlabel="Época",
        ylabel="Huber loss",
        title="Perda na escala log1p padronizada",
    )
    axes[0].legend()

    axes[1].plot(
        epochs,
        [item["validation_mae_usd"] for item in history],
        label="MLP na validação",
    )
    axes[1].axhline(
        payload["baseline_reference"]["best_baseline_mae_usd"],
        color="#2166ac",
        linestyle=":",
        label="Ridge",
    )
    axes[1].axhline(
        payload["baseline_reference"]["mlp_success_threshold_mae_usd"],
        color="#1b7837",
        linestyle="--",
        label="Critério de sucesso",
    )
    axes[1].axvline(payload["best_epoch"], color="#b2182b", linestyle="--")
    axes[1].set(
        xlabel="Época",
        ylabel="MAE de validação (US$)",
        title="Erro na escala original",
    )
    axes[1].legend()
    fig.suptitle("Figura 1: Treinamento da MLP de referência", fontweight="bold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_outputs(result: TrainingResult) -> None:
    payload = result_payload(result)
    RESULTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_training_figure(payload, FIGURE_PATH)


def check_outputs(result: TrainingResult) -> None:
    if not RESULTS_PATH.exists():
        raise TrainingContractError(
            f"Resultado ausente em {RESULTS_PATH}. Use --write-reference."
        )
    observed = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    expected = result_payload(result)
    if observed != expected:
        raise TrainingContractError(
            f"Treinamento divergente em {RESULTS_PATH}. "
            "Use --write-reference somente após revisar a causa."
        )
    if not FIGURE_PATH.exists() or FIGURE_PATH.stat().st_size == 0:
        raise TrainingContractError(f"Figura ausente ou vazia em {FIGURE_PATH}")


def run_canonical_training() -> TrainingResult:
    prepared = prepare_development_data(load_development_splits())
    model_config, training_config = canonical_configs(prepared.X_train.shape[1])
    return train_model(prepared, model_config, training_config)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-reference", action="store_true")
    action.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_canonical_training()
    if args.write_reference:
        write_outputs(result)
        print(f"Referência gravada em {RESULTS_PATH} e {FIGURE_PATH}")
    else:
        check_outputs(result)
        metrics = result.best_metrics_usd
        print(
            f"Referência válida: melhor época={result.best_epoch}, "
            f"parada={result.stopped_epoch}, MAE=US$ {metrics['mae_usd']:,.2f}"
        )


if __name__ == "__main__":
    main()
