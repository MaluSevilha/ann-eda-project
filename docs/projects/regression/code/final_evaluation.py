"""Avaliação final no teste com a configuração congelada e três seeds.

``--upgrade-multiseed`` migra os artefatos antigos de seed única. ``--check``
é offline: valida somente os artefatos versionados.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ann-eda-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge

from docs.projects.eda.code.pipeline import GROUP_COLUMN, MODEL_FEATURES, TARGET
from docs.projects.regression.code.baselines import regression_metrics
from docs.projects.regression.code.data import (
    PreparedDevelopmentData,
    _load_locked_test_for_final_evaluation,
    load_development_splits,
    prepare_development_data,
)
from docs.projects.regression.code.final_config import load_frozen_config
from docs.projects.regression.code.train import TrainingResult, predict_usd, train_model

REGRESSION_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = REGRESSION_DIR / "final_test_results.json"
PREDICTIONS_PATH = REGRESSION_DIR / "final_test_predictions.csv"
TABLE_PATH = REGRESSION_DIR / "tables" / "final_test_metrics.md"
PREDICTED_FIGURE_PATH = REGRESSION_DIR / "figures" / "fig03_predicted_vs_actual.png"
RESIDUAL_FIGURE_PATH = REGRESSION_DIR / "figures" / "fig04_residuals_vs_predicted.png"
BASELINE_RESULTS_PATH = REGRESSION_DIR / "baseline_results.json"
SPLIT_MANIFEST_PATH = REGRESSION_DIR / "split_manifest.json"

FINAL_SEEDS = (42, 43, 44)
REFERENCE_SEED = 42
HISTORICAL_TEST_ACCESS_COUNT = 2
SUCCESS_FRACTION = 0.85
CONTEXT_COLUMNS = ("program_id", GROUP_COLUMN, "credential_name", "institution_control")


class FinalEvaluationError(RuntimeError):
    """Indica quebra do contrato da avaliação final."""


@dataclass
class FinalEvaluationResult:
    mlp_trainings: dict[int, TrainingResult]
    predictions: pd.DataFrame
    metrics: dict[str, dict[str, float]]
    test_identity: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _program_id_sha256(values: pd.Series) -> str:
    normalized = "\n".join(sorted(values.astype(str))) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise FinalEvaluationError(f"{label} ausente em {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalEvaluationError(f"{label} inválido em {path}: {error}") from error
    if not isinstance(payload, dict):
        raise FinalEvaluationError(f"{label} deve ser um objeto JSON")
    return payload


def _selected_baseline_contract() -> dict[str, Any]:
    payload = _read_json(BASELINE_RESULTS_PATH, "Resultados dos baselines")
    if (
        payload.get("selection_split") != "validation"
        or payload.get("test_accessed") is not False
    ):
        raise FinalEvaluationError(
            "A seleção dos baselines não foi isolada no desenvolvimento"
        )
    best = payload.get("best_baseline")
    if (
        not isinstance(best, dict)
        or best.get("model") != "Ridge"
        or best.get("configuration") != "alpha=1"
    ):
        raise FinalEvaluationError("O baseline congelado deve ser Ridge com alpha=1")
    return best


def _fit_baselines(prepared: PreparedDevelopmentData) -> tuple[DummyRegressor, Ridge]:
    _selected_baseline_contract()
    dummy = DummyRegressor(strategy="mean")
    ridge = Ridge(alpha=1.0, solver="lsqr", max_iter=10_000, tol=1e-6)
    dummy.fit(prepared.X_train, prepared.y_train)
    ridge.fit(prepared.X_train, prepared.y_train)
    return dummy, ridge


def _sort_test(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.assign(_program_sort=frame["program_id"].astype(str))
        .sort_values("_program_sort", kind="stable")
        .drop(columns="_program_sort")
        .reset_index(drop=True)
    )


def _transform_test(
    prepared: PreparedDevelopmentData, test: pd.DataFrame
) -> sparse.csr_matrix:
    transformed = prepared.preprocessor.transform(test[MODEL_FEATURES])
    matrix = (
        transformed.tocsr()
        if sparse.issparse(transformed)
        else sparse.csr_matrix(transformed)
    )
    matrix = matrix.astype(np.float32)
    if matrix.shape[1] != prepared.X_train.shape[1]:
        raise FinalEvaluationError(
            f"Teste com {matrix.shape[1]} features; "
            f"treino com {prepared.X_train.shape[1]}"
        )
    if np.isnan(matrix.data).any():
        raise FinalEvaluationError("O teste transformado contém NaN")
    return matrix


def run_final_evaluation() -> FinalEvaluationResult:
    """Treina todos os modelos e só então abre o teste uma única vez."""

    prepared = prepare_development_data(load_development_splits())
    frozen = load_frozen_config()
    if frozen.training_config.seed != REFERENCE_SEED:
        raise FinalEvaluationError(
            f"A configuração congelada deve usar a seed-base {REFERENCE_SEED}"
        )
    mlp_trainings = {
        seed: train_model(
            prepared,
            frozen.model_config,
            replace(frozen.training_config, seed=seed),
        )
        for seed in FINAL_SEEDS
    }
    dummy, ridge = _fit_baselines(prepared)

    test = _sort_test(_load_locked_test_for_final_evaluation())
    X_test = _transform_test(prepared, test)
    y_test = test[TARGET].to_numpy(dtype=np.float64)
    mlp_predictions = {
        seed: predict_usd(
            training.model,
            X_test,
            training.target_transformer,
            device=frozen.training_config.device,
            batch_size=frozen.training_config.validation_batch_size,
        )
        for seed, training in mlp_trainings.items()
    }
    dummy_prediction = dummy.predict(X_test).astype(np.float64)
    ridge_prediction = ridge.predict(X_test).astype(np.float64)

    predictions = test.loc[:, CONTEXT_COLUMNS].copy()
    predictions["actual_usd"] = y_test
    for seed, prediction in mlp_predictions.items():
        predictions[f"mlp_seed_{seed}_usd"] = prediction
    predictions["ridge_usd"] = ridge_prediction
    predictions["dummy_usd"] = dummy_prediction
    reference_prediction = mlp_predictions[REFERENCE_SEED]
    predictions["mlp_residual_usd"] = y_test - reference_prediction
    predictions["mlp_absolute_error_usd"] = np.abs(y_test - reference_prediction)
    metrics = {
        **{
            f"mlp_seed_{seed}": regression_metrics(y_test, prediction)
            for seed, prediction in mlp_predictions.items()
        },
        "ridge": regression_metrics(y_test, ridge_prediction),
        "dummy": regression_metrics(y_test, dummy_prediction),
    }
    return FinalEvaluationResult(
        mlp_trainings=mlp_trainings,
        predictions=predictions,
        metrics=metrics,
        test_identity={
            "rows": len(test),
            "institutions": int(test[GROUP_COLUMN].nunique()),
            "program_id_sha256": _program_id_sha256(test["program_id"]),
        },
    )


def _rounded_metrics(metrics: dict[str, float]) -> dict[str, float]:
    return {
        "mae_usd": round(metrics["mae_usd"], 2),
        "rmse_usd": round(metrics["rmse_usd"], 2),
        "r2": round(metrics["r2"], 6),
    }


def _build_core_payload(
    metrics: dict[str, dict[str, float]],
    test_identity: dict[str, Any],
    epochs: dict[int, tuple[int, int]],
) -> dict[str, Any]:
    rounded = {name: _rounded_metrics(value) for name, value in metrics.items()}
    baseline_name = min(
        ("ridge", "dummy"), key=lambda name: rounded[name]["mae_usd"]
    )
    baseline_mae = rounded[baseline_name]["mae_usd"]
    aggregate: dict[str, dict[str, float | int]] = {}
    for metric, digits in (("mae_usd", 2), ("rmse_usd", 2), ("r2", 6)):
        values = np.asarray(
            [metrics[f"mlp_seed_{seed}"][metric] for seed in FINAL_SEEDS],
            dtype=np.float64,
        )
        aggregate[metric] = {
            "mean": round(float(values.mean()), digits),
            "std": round(float(values.std(ddof=1)), digits),
            "ddof": 1,
        }
    mlp_mae = float(aggregate["mae_usd"]["mean"])
    mlp_mae_std = float(aggregate["mae_usd"]["std"])
    threshold = round(SUCCESS_FRACTION * baseline_mae, 2)
    improvement = round(100.0 * (baseline_mae - mlp_mae) / baseline_mae, 2)
    gap = round(baseline_mae - mlp_mae, 2)
    return {
        "schema_version": 2,
        "evaluation_split": "test",
        "test_accessed": True,
        "test_access_count": HISTORICAL_TEST_ACCESS_COUNT,
        "selection_completed_before_test": True,
        "protocol": {
            "seeds": list(FINAL_SEEDS),
            "reference_seed": REFERENCE_SEED,
            "multiple_seeds": True,
            "aggregation": "sample_mean_and_sample_std_ddof_1",
            "current_run_test_load_count": 1,
            "training_split": "train",
            "early_stopping_split": "validation",
            "frozen_config": "final_config.json",
            "baseline_selection": "baseline_results.json",
        },
        "test_identity": test_identity,
        "models": {
            **{
                f"mlp_seed_{seed}": {
                    "configuration": "low_lr_regularized",
                    "best_epoch": epochs[seed][0],
                    "stopped_epoch": epochs[seed][1],
                    **rounded[f"mlp_seed_{seed}"],
                }
                for seed in FINAL_SEEDS
            },
            "mlp_aggregate": {
                "configuration": "low_lr_regularized",
                "seed_count": len(FINAL_SEEDS),
                **aggregate,
            },
            "ridge": {"configuration": "alpha=1", **rounded["ridge"]},
            "dummy": {"configuration": "strategy=mean", **rounded["dummy"]},
        },
        "success_criterion": {
            "required_improvement_pct": 15.0,
            "best_baseline": baseline_name,
            "best_baseline_mae_usd": baseline_mae,
            "maximum_mlp_mae_usd": threshold,
            "mlp_mae_usd": mlp_mae,
            "mlp_mae_std_usd": mlp_mae_std,
            "observed_improvement_pct": improvement,
            "met": mlp_mae <= threshold,
            "gap_to_baseline_usd": gap,
            "gap_exceeds_seed_std": gap > mlp_mae_std,
        },
        "limitations": {
            "test_reopened_for_multiseed_correction": True,
            "configuration_changed_after_first_test_access": False,
        },
    }


def _core_payload(result: FinalEvaluationResult) -> dict[str, Any]:
    return _build_core_payload(
        result.metrics,
        result.test_identity,
        {
            seed: (training.best_epoch, training.stopped_epoch)
            for seed, training in result.mlp_trainings.items()
        },
    )


def _format_number(value: float, digits: int) -> str:
    formatted = f"{value:,.{digits}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def render_table(payload: dict[str, Any]) -> str:
    lines = [
        "| Modelo | Configuração | MAE (US$) | RMSE (US$) | R² |",
        "|---|---|---:|---:|---:|",
    ]
    labels = {
        "mlp_seed_42": "MLP (seed 42)",
        "mlp_seed_43": "MLP (seed 43)",
        "mlp_seed_44": "MLP (seed 44)",
        "ridge": "Ridge",
        "dummy": "DummyRegressor",
    }
    aggregate = payload["models"]["mlp_aggregate"]
    lines.append(
        "| MLP (média ± desvio) | `low_lr_regularized`; 3 seeds | "
        f"{_format_number(aggregate['mae_usd']['mean'], 2)} ± "
        f"{_format_number(aggregate['mae_usd']['std'], 2)} | "
        f"{_format_number(aggregate['rmse_usd']['mean'], 2)} ± "
        f"{_format_number(aggregate['rmse_usd']['std'], 2)} | "
        f"{_format_number(aggregate['r2']['mean'], 6)} ± "
        f"{_format_number(aggregate['r2']['std'], 6)} |"
    )
    for key in ("mlp_seed_42", "mlp_seed_43", "mlp_seed_44", "ridge", "dummy"):
        model = payload["models"][key]
        lines.append(
            f"| {labels[key]} | `{model['configuration']}` | "
            f"{_format_number(model['mae_usd'], 2)} | "
            f"{_format_number(model['rmse_usd'], 2)} | "
            f"{_format_number(model['r2'], 6)} |"
        )
    return "\n".join(lines) + "\n"


def _save_figures(predictions: pd.DataFrame) -> None:
    actual = predictions["actual_usd"].to_numpy()
    predicted = predictions["mlp_seed_42_usd"].to_numpy()
    lower = float(min(actual.min(), predicted.min()))
    upper = float(max(actual.max(), predicted.max()))

    fig, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(actual, predicted, alpha=0.25, s=10, color="#2166ac")
    axis.plot(
        [lower, upper],
        [lower, upper],
        "--",
        color="#b2182b",
        label="Previsão ideal",
    )
    axis.set(
        xlabel="Renda real (US$)",
        ylabel="Renda prevista (US$)",
        title="Figura 3: MLP seed 42 — previsto × real no teste",
    )
    axis.legend()
    fig.tight_layout()
    PREDICTED_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        PREDICTED_FIGURE_PATH, dpi=180, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)

    residuals = actual - predicted
    fig, axis = plt.subplots(figsize=(8, 7))
    axis.scatter(predicted, residuals, alpha=0.25, s=10, color="#2166ac")
    axis.axhline(0.0, linestyle="--", color="#b2182b", label="Resíduo zero")
    axis.set(
        xlabel="Renda prevista (US$)",
        ylabel="Resíduo: real − previsto (US$)",
        title="Figura 4: MLP seed 42 — resíduos no teste",
    )
    axis.legend()
    fig.tight_layout()
    fig.savefig(
        RESIDUAL_FIGURE_PATH, dpi=180, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def _artifact_paths() -> tuple[Path, ...]:
    return (
        RESULTS_PATH,
        PREDICTIONS_PATH,
        TABLE_PATH,
        PREDICTED_FIGURE_PATH,
        RESIDUAL_FIGURE_PATH,
    )


def ensure_outputs_absent() -> None:
    existing = [str(path) for path in _artifact_paths() if path.exists()]
    if existing:
        raise FinalEvaluationError(
            "A avaliação final não permite sobrescrita; artefatos existentes: "
            + ", ".join(existing)
        )


def ensure_single_seed_upgrade_source() -> None:
    """Aceita somente a migração auditável dos artefatos antigos conhecidos."""

    payload = _read_json(RESULTS_PATH, "Resultado final anterior")
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol", {}).get("seed") != REFERENCE_SEED
        or payload.get("test_access_count") != 1
        or payload.get("limitations", {}).get("single_seed_only") is not True
    ):
        raise FinalEvaluationError("A origem não é a avaliação antiga de seed única")
    path_map = {
        "predictions": PREDICTIONS_PATH,
        "table": TABLE_PATH,
        "predicted_vs_actual_figure": PREDICTED_FIGURE_PATH,
        "residual_figure": RESIDUAL_FIGURE_PATH,
    }
    for name, path in path_map.items():
        if payload.get("artifacts", {}).get(name, {}).get("sha256") != _sha256_file(path):
            raise FinalEvaluationError(f"Artefato anterior divergente: {name}")


def write_outputs(result: FinalEvaluationResult) -> None:
    core = _core_payload(result)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.to_csv(
        PREDICTIONS_PATH, index=False, float_format="%.10f"
    )
    _save_figures(result.predictions)
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(render_table(core), encoding="utf-8")
    core["artifacts"] = {
        "predictions": {
            "path": PREDICTIONS_PATH.name,
            "sha256": _sha256_file(PREDICTIONS_PATH),
        },
        "table": {
            "path": f"tables/{TABLE_PATH.name}",
            "sha256": _sha256_file(TABLE_PATH),
        },
        "predicted_vs_actual_figure": {
            "path": f"figures/{PREDICTED_FIGURE_PATH.name}",
            "sha256": _sha256_file(PREDICTED_FIGURE_PATH),
        },
        "residual_figure": {
            "path": f"figures/{RESIDUAL_FIGURE_PATH.name}",
            "sha256": _sha256_file(RESIDUAL_FIGURE_PATH),
        },
    }
    RESULTS_PATH.write_text(
        json.dumps(core, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_and_write_final_evaluation() -> FinalEvaluationResult:
    ensure_outputs_absent()
    result = run_final_evaluation()
    write_outputs(result)
    return result


def upgrade_and_write_multiseed_evaluation() -> FinalEvaluationResult:
    ensure_single_seed_upgrade_source()
    result = run_final_evaluation()
    write_outputs(result)
    return result


def _metrics_from_predictions(
    predictions: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    actual = predictions["actual_usd"].to_numpy(dtype=np.float64)
    return {
        **{
            f"mlp_seed_{seed}": regression_metrics(
                actual,
                predictions[f"mlp_seed_{seed}_usd"].to_numpy(dtype=np.float64),
            )
            for seed in FINAL_SEEDS
        },
        "ridge": regression_metrics(
            actual, predictions["ridge_usd"].to_numpy(dtype=np.float64)
        ),
        "dummy": regression_metrics(
            actual, predictions["dummy_usd"].to_numpy(dtype=np.float64)
        ),
    }


def check_outputs() -> dict[str, Any]:
    """Valida os artefatos sem carregar ou transformar o teste original."""

    payload = _read_json(RESULTS_PATH, "Resultado final")
    frozen = load_frozen_config()
    if frozen.training_config.seed != REFERENCE_SEED:
        raise FinalEvaluationError("Seed congelada divergente")
    required = _artifact_paths()[1:]
    missing = [
        str(path) for path in required if not path.exists() or path.stat().st_size == 0
    ]
    if missing:
        raise FinalEvaluationError(
            "Artefatos finais ausentes ou vazios: " + ", ".join(missing)
        )
    predictions = pd.read_csv(PREDICTIONS_PATH)
    expected_columns = {
        *CONTEXT_COLUMNS,
        "actual_usd",
        *(f"mlp_seed_{seed}_usd" for seed in FINAL_SEEDS),
        "ridge_usd",
        "dummy_usd",
        "mlp_residual_usd",
        "mlp_absolute_error_usd",
    }
    if set(predictions.columns) != expected_columns:
        raise FinalEvaluationError("Colunas das previsões finais divergentes")
    if predictions["program_id"].duplicated().any():
        raise FinalEvaluationError("program_id duplicado nas previsões finais")
    manifest = _read_json(SPLIT_MANIFEST_PATH, "Manifesto dos dados")
    expected_identity = manifest["splits"]["test_locked"]
    observed_identity = {
        "rows": len(predictions),
        "institutions": int(predictions[GROUP_COLUMN].nunique()),
        "program_id_sha256": _program_id_sha256(predictions["program_id"]),
    }
    if (
        observed_identity != expected_identity
        or payload.get("test_identity") != expected_identity
    ):
        raise FinalEvaluationError("Identidade das previsões de teste divergente")
    residual = predictions["actual_usd"] - predictions["mlp_seed_42_usd"]
    if not np.allclose(residual, predictions["mlp_residual_usd"], atol=1e-6):
        raise FinalEvaluationError("Resíduos salvos divergentes")
    if not np.allclose(
        np.abs(residual), predictions["mlp_absolute_error_usd"], atol=1e-6
    ):
        raise FinalEvaluationError("Erros absolutos salvos divergentes")
    observed_metrics = _metrics_from_predictions(predictions)
    for name, metrics in observed_metrics.items():
        stored = {
            key: payload["models"][name][key]
            for key in ("mae_usd", "rmse_usd", "r2")
        }
        if stored != _rounded_metrics(metrics):
            raise FinalEvaluationError(f"Métricas divergentes para {name}")
    expected_core = _build_core_payload(
        observed_metrics,
        observed_identity,
        {
            seed: (
                payload["models"][f"mlp_seed_{seed}"]["best_epoch"],
                payload["models"][f"mlp_seed_{seed}"]["stopped_epoch"],
            )
            for seed in FINAL_SEEDS
        },
    )
    observed_core = dict(payload)
    artifacts = observed_core.pop("artifacts", None)
    if observed_core != expected_core:
        raise FinalEvaluationError("Contrato ou critério de sucesso divergente")
    if not isinstance(artifacts, dict):
        raise FinalEvaluationError("Hashes dos artefatos ausentes")
    path_map = {
        "predictions": PREDICTIONS_PATH,
        "table": TABLE_PATH,
        "predicted_vs_actual_figure": PREDICTED_FIGURE_PATH,
        "residual_figure": RESIDUAL_FIGURE_PATH,
    }
    for name, path in path_map.items():
        if artifacts.get(name, {}).get("sha256") != _sha256_file(path):
            raise FinalEvaluationError(f"Hash divergente para {name}")
    if TABLE_PATH.read_text(encoding="utf-8") != render_table(payload):
        raise FinalEvaluationError("Tabela final divergente")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-results", action="store_true")
    action.add_argument("--upgrade-multiseed", action="store_true")
    action.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.write_results:
        result = run_and_write_final_evaluation()
        metrics = result.metrics["mlp_seed_42"]
        print(
            f"Avaliação final gravada: MLP seed 42, "
            f"MAE=US$ {metrics['mae_usd']:,.2f}"
        )
        return
    if args.upgrade_multiseed:
        result = upgrade_and_write_multiseed_evaluation()
        values = [result.metrics[f"mlp_seed_{seed}"]["mae_usd"] for seed in FINAL_SEEDS]
        print(
            "Avaliação multiseed gravada: "
            f"MAE=US$ {np.mean(values):,.2f} ± {np.std(values, ddof=1):,.2f}"
        )
        return
    payload = check_outputs()
    criterion = payload["success_criterion"]
    print(
        f"Avaliação final válida: MAE=US$ {criterion['mlp_mae_usd']:,.2f}, "
        f"critério de 15% atingido={criterion['met']}"
    )


if __name__ == "__main__":
    main()
