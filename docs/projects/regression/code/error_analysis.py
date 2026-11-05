"""Análise de erros finais e importância usando apenas desenvolvimento.

Os erros de teste vêm do CSV final já versionado. Novos treinamentos e
permutation importance usam exclusivamente treino e validação.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ann-eda-matplotlib")
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer

from docs.projects.eda.code.pipeline import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
)
from docs.projects.regression.code.baselines import regression_metrics
from docs.projects.regression.code.data import (
    RARE_CATEGORY_MIN_COUNT,
    DevelopmentSplits,
    PreparedDevelopmentData,
    load_development_splits,
    prepare_development_data,
)
from docs.projects.regression.code.final_config import load_frozen_config
from docs.projects.regression.code.model import MLPConfig
from docs.projects.regression.code.train import predict_usd, train_model

REGRESSION_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = REGRESSION_DIR / "error_analysis.json"
PREDICTIONS_PATH = REGRESSION_DIR / "final_test_predictions.csv"
FINAL_RESULTS_PATH = REGRESSION_DIR / "final_test_results.json"
FIGURE_GROUPS_PATH = REGRESSION_DIR / "figures" / "fig05_error_subgroups.png"
FIGURE_IMPORTANCE_PATH = (
    REGRESSION_DIR / "figures" / "fig06_permutation_importance.png"
)
TABLE_PATHS = {
    "credential": REGRESSION_DIR / "tables" / "error_by_credential.md",
    "control": REGRESSION_DIR / "tables" / "error_by_institution_control.md",
    "target_band": REGRESSION_DIR / "tables" / "error_by_target_band.md",
    "worst": REGRESSION_DIR / "tables" / "worst_test_errors.md",
    "importance": REGRESSION_DIR / "tables" / "permutation_importance.md",
    "ablation": REGRESSION_DIR / "tables" / "earnings_1yr_ablation.md",
}

PERMUTATION_REPEATS = 3
ANALYSIS_SEED = 42
SMALL_GROUP_THRESHOLD = 100
ABLATION_FEATURE = "median_earnings_1yr_usd"
TARGET_BAND_EDGES = (-np.inf, 46_709.0, 58_878.0, 77_121.0, np.inf)
TARGET_BAND_LABELS = (
    "até US$ 46.709",
    "US$ 46.709–58.878",
    "US$ 58.878–77.121",
    "acima de US$ 77.121",
)


class ErrorAnalysisContractError(RuntimeError):
    """Indica divergência nos dados ou artefatos da análise."""


@dataclass(frozen=True)
class ErrorGroupResult:
    group: str
    count: int
    mae_usd: float
    rmse_usd: float
    median_absolute_error_usd: float
    mean_residual_usd: float
    small_group: bool


@dataclass(frozen=True)
class PermutationImportanceResult:
    feature: str
    delta_mae_mean_usd: float
    delta_mae_std_usd: float
    delta_mae_repeats_usd: tuple[float, ...]


@dataclass(frozen=True)
class AblationResult:
    removed_feature: str
    full_input_dim: int
    ablated_input_dim: int
    full_mae_usd: float
    ablated_mae_usd: float
    delta_mae_usd: float
    relative_change_pct: float
    ablated_rmse_usd: float
    ablated_r2: float
    best_epoch: int
    stopped_epoch: int


@dataclass(frozen=True)
class ErrorAnalysisStudy:
    source_sha256: str
    source_rows: int
    credential_groups: tuple[ErrorGroupResult, ...]
    control_groups: tuple[ErrorGroupResult, ...]
    target_bands: tuple[ErrorGroupResult, ...]
    worst_examples: tuple[dict[str, Any], ...]
    baseline_validation_mae_usd: float
    permutation_importance: tuple[PermutationImportanceResult, ...]
    ablation: AblationResult


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise ErrorAnalysisContractError(f"{label} ausente em {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ErrorAnalysisContractError(f"{label} inválido: {error}") from error
    if not isinstance(payload, dict):
        raise ErrorAnalysisContractError(f"{label} deve ser um objeto JSON")
    return payload


def load_final_predictions() -> tuple[pd.DataFrame, str]:
    """Lê somente o artefato de previsões; nunca abre o dataset original."""

    final_results = _read_json(FINAL_RESULTS_PATH, "Resultado final")
    if (
        final_results.get("test_accessed") is not True
        or final_results.get("test_access_count") != 1
    ):
        raise ErrorAnalysisContractError("Recibo da avaliação final inválido")
    expected_hash = (
        final_results.get("artifacts", {}).get("predictions", {}).get("sha256")
    )
    observed_hash = _sha256_file(PREDICTIONS_PATH)
    if expected_hash != observed_hash:
        raise ErrorAnalysisContractError("Hash das previsões finais divergente")
    predictions = pd.read_csv(PREDICTIONS_PATH)
    required = {
        "program_id",
        "credential_name",
        "institution_control",
        "actual_usd",
        "mlp_seed_42_usd",
        "mlp_residual_usd",
        "mlp_absolute_error_usd",
    }
    if not required.issubset(predictions.columns):
        raise ErrorAnalysisContractError("Previsões finais sem colunas obrigatórias")
    if predictions["program_id"].duplicated().any():
        raise ErrorAnalysisContractError("program_id duplicado nas previsões finais")
    if len(predictions) != final_results.get("test_identity", {}).get("rows"):
        raise ErrorAnalysisContractError("Quantidade de previsões finais divergente")
    return predictions, observed_hash


def _group_metrics(
    frame: pd.DataFrame, group_column: str
) -> tuple[ErrorGroupResult, ...]:
    results: list[ErrorGroupResult] = []
    for value, group in frame.groupby(group_column, observed=True, dropna=False):
        actual = group["actual_usd"].to_numpy(dtype=np.float64)
        predicted = group["mlp_seed_42_usd"].to_numpy(dtype=np.float64)
        residual = actual - predicted
        results.append(
            ErrorGroupResult(
                group=str(value),
                count=len(group),
                mae_usd=float(np.mean(np.abs(residual))),
                rmse_usd=float(np.sqrt(np.mean(np.square(residual)))),
                median_absolute_error_usd=float(np.median(np.abs(residual))),
                mean_residual_usd=float(np.mean(residual)),
                small_group=len(group) < SMALL_GROUP_THRESHOLD,
            )
        )
    return tuple(sorted(results, key=lambda item: (-item.mae_usd, item.group)))


def _target_band_metrics(predictions: pd.DataFrame) -> tuple[ErrorGroupResult, ...]:
    banded = predictions.copy()
    banded["target_band"] = pd.cut(
        banded["actual_usd"],
        bins=TARGET_BAND_EDGES,
        labels=TARGET_BAND_LABELS,
        right=True,
        include_lowest=True,
        ordered=True,
    )
    if banded["target_band"].isna().any():
        raise ErrorAnalysisContractError("Há alvos fora das faixas documentadas")
    by_label = {item.group: item for item in _group_metrics(banded, "target_band")}
    return tuple(by_label[label] for label in TARGET_BAND_LABELS)


def _worst_examples(predictions: pd.DataFrame) -> tuple[dict[str, Any], ...]:
    ordered = predictions.sort_values(
        ["mlp_absolute_error_usd", "program_id"],
        ascending=[False, True],
        kind="stable",
    ).head(10)
    return tuple(
        {
            "program_id": str(row.program_id),
            "credential_name": str(row.credential_name),
            "institution_control": str(row.institution_control),
            "actual_usd": float(row.actual_usd),
            "predicted_usd": float(row.mlp_seed_42_usd),
            "residual_usd": float(row.mlp_residual_usd),
            "absolute_error_usd": float(row.mlp_absolute_error_usd),
        }
        for row in ordered.itertuples(index=False)
    )


def _as_csr_float32(values: Any) -> sparse.csr_matrix:
    matrix = values.tocsr() if sparse.issparse(values) else sparse.csr_matrix(values)
    return matrix.astype(np.float32)


def _permutation_importance(
    splits: DevelopmentSplits,
    prepared: PreparedDevelopmentData,
    model: Any,
    target_transformer: Any,
) -> tuple[float, tuple[PermutationImportanceResult, ...]]:
    baseline_prediction = predict_usd(
        model,
        prepared.X_validation,
        target_transformer,
        device="cpu",
        batch_size=2048,
    )
    baseline_mae = regression_metrics(
        prepared.y_validation, baseline_prediction
    )["mae_usd"]
    results: list[PermutationImportanceResult] = []
    validation_features = splits.validation.loc[:, MODEL_FEATURES]
    for feature_index, feature in enumerate(MODEL_FEATURES):
        deltas: list[float] = []
        values = validation_features[feature].to_numpy(copy=True)
        for repeat_index in range(PERMUTATION_REPEATS):
            rng = np.random.default_rng(
                ANALYSIS_SEED + feature_index * 1_000 + repeat_index
            )
            permuted = validation_features.copy()
            permuted[feature] = values[rng.permutation(len(values))]
            matrix = _as_csr_float32(prepared.preprocessor.transform(permuted))
            prediction = predict_usd(
                model,
                matrix,
                target_transformer,
                device="cpu",
                batch_size=2048,
            )
            permuted_mae = regression_metrics(
                prepared.y_validation, prediction
            )["mae_usd"]
            deltas.append(permuted_mae - baseline_mae)
        results.append(
            PermutationImportanceResult(
                feature=feature,
                delta_mae_mean_usd=float(np.mean(deltas)),
                delta_mae_std_usd=float(np.std(deltas, ddof=1)),
                delta_mae_repeats_usd=tuple(deltas),
            )
        )
    ordered = tuple(
        sorted(results, key=lambda item: (-item.delta_mae_mean_usd, item.feature))
    )
    return baseline_mae, ordered


def _build_ablation_preprocessor() -> ColumnTransformer:
    base = build_preprocessor(min_category_frequency=RARE_CATEGORY_MIN_COUNT)
    transformers = []
    for name, transformer, columns in base.transformers:
        reduced = [column for column in columns if column != ABLATION_FEATURE]
        transformers.append((name, clone(transformer), reduced))
    return ColumnTransformer(
        transformers=transformers,
        sparse_threshold=base.sparse_threshold,
        verbose_feature_names_out=True,
    )


def _prepare_ablation(splits: DevelopmentSplits) -> PreparedDevelopmentData:
    preprocessor = _build_ablation_preprocessor()
    reduced_features = [feature for feature in MODEL_FEATURES if feature != ABLATION_FEATURE]
    X_train = _as_csr_float32(
        preprocessor.fit_transform(splits.train.loc[:, reduced_features])
    )
    X_validation = _as_csr_float32(
        preprocessor.transform(splits.validation.loc[:, reduced_features])
    )
    names = tuple(map(str, preprocessor.get_feature_names_out()))
    if any(ABLATION_FEATURE in name for name in names):
        raise ErrorAnalysisContractError("A feature removida persiste na ablação")
    return PreparedDevelopmentData(
        X_train=X_train,
        X_validation=X_validation,
        y_train=splits.train[TARGET].to_numpy(dtype=np.float32),
        y_validation=splits.validation[TARGET].to_numpy(dtype=np.float32),
        feature_names=names,
        preprocessor=preprocessor,
    )


def _run_ablation(
    splits: DevelopmentSplits,
    full_input_dim: int,
    full_mae_usd: float,
) -> AblationResult:
    frozen = load_frozen_config()
    ablated = _prepare_ablation(splits)
    model_config = replace(
        frozen.model_config,
        input_dim=ablated.X_train.shape[1],
    )
    result = train_model(ablated, model_config, frozen.training_config)
    metrics = result.best_metrics_usd
    delta = metrics["mae_usd"] - full_mae_usd
    return AblationResult(
        removed_feature=ABLATION_FEATURE,
        full_input_dim=full_input_dim,
        ablated_input_dim=ablated.X_train.shape[1],
        full_mae_usd=full_mae_usd,
        ablated_mae_usd=metrics["mae_usd"],
        delta_mae_usd=delta,
        relative_change_pct=100.0 * delta / full_mae_usd,
        ablated_rmse_usd=metrics["rmse_usd"],
        ablated_r2=metrics["r2"],
        best_epoch=result.best_epoch,
        stopped_epoch=result.stopped_epoch,
    )


def run_error_analysis() -> ErrorAnalysisStudy:
    predictions, source_hash = load_final_predictions()
    credential = _group_metrics(predictions, "credential_name")
    control = _group_metrics(predictions, "institution_control")
    target_bands = _target_band_metrics(predictions)
    worst = _worst_examples(predictions)

    splits = load_development_splits()
    prepared = prepare_development_data(splits)
    frozen = load_frozen_config()
    full_result = train_model(
        prepared,
        frozen.model_config,
        frozen.training_config,
    )
    baseline_mae, importance = _permutation_importance(
        splits,
        prepared,
        full_result.model,
        full_result.target_transformer,
    )
    expected_mae = frozen.validation_metrics_usd["mae_usd"]
    if round(baseline_mae, 2) != round(expected_mae, 2):
        raise ErrorAnalysisContractError(
            f"MAE base divergente: esperado={expected_mae:.2f}, "
            f"observado={baseline_mae:.2f}"
        )
    ablation = _run_ablation(
        splits,
        prepared.X_train.shape[1],
        baseline_mae,
    )
    return ErrorAnalysisStudy(
        source_sha256=source_hash,
        source_rows=len(predictions),
        credential_groups=credential,
        control_groups=control,
        target_bands=target_bands,
        worst_examples=worst,
        baseline_validation_mae_usd=baseline_mae,
        permutation_importance=importance,
        ablation=ablation,
    )


def _group_payload(result: ErrorGroupResult) -> dict[str, Any]:
    return {
        "group": result.group,
        "count": result.count,
        "mae_usd": round(result.mae_usd, 2),
        "rmse_usd": round(result.rmse_usd, 2),
        "median_absolute_error_usd": round(result.median_absolute_error_usd, 2),
        "mean_residual_usd": round(result.mean_residual_usd, 2),
        "small_group": result.small_group,
    }


def study_payload(study: ErrorAnalysisStudy) -> dict[str, Any]:
    ablation = study.ablation
    return {
        "schema_version": 1,
        "test_source": {
            "artifact": PREDICTIONS_PATH.name,
            "sha256": study.source_sha256,
            "rows": study.source_rows,
            "raw_test_reopened": False,
        },
        "group_policy": {"small_group_threshold": SMALL_GROUP_THRESHOLD},
        "target_band_policy": {
            "source": "quartis do treino registrados na EDA",
            "cut_points_usd": [46_709, 58_878, 77_121],
            "labels": list(TARGET_BAND_LABELS),
        },
        "error_by_credential": [_group_payload(item) for item in study.credential_groups],
        "error_by_institution_control": [
            _group_payload(item) for item in study.control_groups
        ],
        "error_by_target_band": [_group_payload(item) for item in study.target_bands],
        "worst_examples": [
            {
                **item,
                "actual_usd": round(item["actual_usd"], 2),
                "predicted_usd": round(item["predicted_usd"], 2),
                "residual_usd": round(item["residual_usd"], 2),
                "absolute_error_usd": round(item["absolute_error_usd"], 2),
            }
            for item in study.worst_examples
        ],
        "permutation_importance": {
            "split": "validation",
            "metric": "increase_in_mae_usd",
            "seed": ANALYSIS_SEED,
            "repeats": PERMUTATION_REPEATS,
            "baseline_mae_usd": round(study.baseline_validation_mae_usd, 2),
            "features": [
                {
                    "feature": item.feature,
                    "delta_mae_mean_usd": round(item.delta_mae_mean_usd, 2),
                    "delta_mae_std_usd": round(item.delta_mae_std_usd, 2),
                    "delta_mae_repeats_usd": [
                        round(value, 2) for value in item.delta_mae_repeats_usd
                    ],
                }
                for item in study.permutation_importance
            ],
        },
        "ablation": {
            "split": "validation",
            "removed_feature": ablation.removed_feature,
            "full_input_dim": ablation.full_input_dim,
            "ablated_input_dim": ablation.ablated_input_dim,
            "full_mae_usd": round(ablation.full_mae_usd, 2),
            "ablated_mae_usd": round(ablation.ablated_mae_usd, 2),
            "delta_mae_usd": round(ablation.delta_mae_usd, 2),
            "relative_change_pct": round(ablation.relative_change_pct, 2),
            "ablated_rmse_usd": round(ablation.ablated_rmse_usd, 2),
            "ablated_r2": round(ablation.ablated_r2, 6),
            "best_epoch": ablation.best_epoch,
            "stopped_epoch": ablation.stopped_epoch,
        },
    }


def _number(value: float, digits: int = 2) -> str:
    formatted = f"{value:,.{digits}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def _render_group_table(rows: Sequence[dict[str, Any]], label: str) -> str:
    lines = [
        f"| {label} | n | MAE (US$) | RMSE (US$) | Mediana erro abs. | Resíduo médio | Grupo pequeno |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['group']} | {row['count']} | {_number(row['mae_usd'])} | "
            f"{_number(row['rmse_usd'])} | "
            f"{_number(row['median_absolute_error_usd'])} | "
            f"{_number(row['mean_residual_usd'])} | "
            f"{'sim' if row['small_group'] else 'não'} |"
        )
    return "\n".join(lines) + "\n"


def render_tables(payload: dict[str, Any]) -> dict[str, str]:
    worst_lines = [
        "| Ordem | program_id | Credencial | Controle | Real (US$) | Previsto (US$) | Resíduo | Erro absoluto |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for position, row in enumerate(payload["worst_examples"], start=1):
        worst_lines.append(
            f"| {position} | `{row['program_id']}` | {row['credential_name']} | "
            f"{row['institution_control']} | {_number(row['actual_usd'])} | "
            f"{_number(row['predicted_usd'])} | {_number(row['residual_usd'])} | "
            f"{_number(row['absolute_error_usd'])} |"
        )
    importance_lines = [
        "| Posição | Feature original | Δ MAE médio (US$) | Desvio (US$) | Repetições (US$) |",
        "|---:|---|---:|---:|---|",
    ]
    for position, row in enumerate(
        payload["permutation_importance"]["features"], start=1
    ):
        repeats = "; ".join(_number(value) for value in row["delta_mae_repeats_usd"])
        importance_lines.append(
            f"| {position} | `{row['feature']}` | "
            f"{_number(row['delta_mae_mean_usd'])} | "
            f"{_number(row['delta_mae_std_usd'])} | {repeats} |"
        )
    ablation = payload["ablation"]
    ablation_lines = [
        "| Modelo | Features | MAE (US$) | RMSE (US$) | R² | Δ MAE | Variação |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| Completo | {ablation['full_input_dim']} | {_number(ablation['full_mae_usd'])} | — | — | — | — |",
        f"| Sem `{ablation['removed_feature']}` | {ablation['ablated_input_dim']} | "
        f"{_number(ablation['ablated_mae_usd'])} | {_number(ablation['ablated_rmse_usd'])} | "
        f"{_number(ablation['ablated_r2'], 6)} | {_number(ablation['delta_mae_usd'])} | "
        f"{_number(ablation['relative_change_pct'])}% |",
    ]
    return {
        "credential": _render_group_table(
            payload["error_by_credential"], "Credencial"
        ),
        "control": _render_group_table(
            payload["error_by_institution_control"], "Controle institucional"
        ),
        "target_band": _render_group_table(
            payload["error_by_target_band"], "Faixa do alvo"
        ),
        "worst": "\n".join(worst_lines) + "\n",
        "importance": "\n".join(importance_lines) + "\n",
        "ablation": "\n".join(ablation_lines) + "\n",
    }


def _save_figures(payload: dict[str, Any]) -> None:
    groups = [
        ("Credencial", payload["error_by_credential"]),
        ("Controle", payload["error_by_institution_control"]),
        ("Faixa do alvo", payload["error_by_target_band"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(19, 7))
    for axis, (title, rows) in zip(axes, groups):
        ordered = sorted(rows, key=lambda row: row["mae_usd"])
        axis.barh(
            [row["group"] for row in ordered],
            [row["mae_usd"] for row in ordered],
            color="#2166ac",
        )
        for index, row in enumerate(ordered):
            axis.text(row["mae_usd"] + 40, index, f"n={row['count']}", va="center", fontsize=8)
        axis.set(title=title, xlabel="MAE no teste (US$)")
    fig.suptitle("Figura 5: Erro da MLP por subgrupo", fontweight="bold")
    fig.tight_layout()
    FIGURE_GROUPS_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_GROUPS_PATH, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    top = payload["permutation_importance"]["features"][:15]
    ordered = list(reversed(top))
    fig, axis = plt.subplots(figsize=(10, 7))
    axis.barh(
        [row["feature"] for row in ordered],
        [row["delta_mae_mean_usd"] for row in ordered],
        xerr=[row["delta_mae_std_usd"] for row in ordered],
        color="#1b7837",
        alpha=0.85,
    )
    axis.axvline(0.0, color="#b2182b", linestyle="--")
    axis.set(
        xlabel="Aumento do MAE de validação (US$)",
        title="Figura 6: Importância por permutação — top 15 features",
    )
    fig.tight_layout()
    fig.savefig(
        FIGURE_IMPORTANCE_PATH, dpi=180, bbox_inches="tight", facecolor="white"
    )
    plt.close(fig)


def write_outputs(study: ErrorAnalysisStudy) -> None:
    payload = study_payload(study)
    RESULTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tables = render_tables(payload)
    for name, path in TABLE_PATHS.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(tables[name], encoding="utf-8")
    _save_figures(payload)


def check_outputs(study: ErrorAnalysisStudy) -> None:
    observed = _read_json(RESULTS_PATH, "Análise de erros")
    expected = study_payload(study)
    if observed != expected:
        raise ErrorAnalysisContractError("Análise de erros versionada divergente")
    tables = render_tables(expected)
    for name, path in TABLE_PATHS.items():
        if not path.exists() or path.read_text(encoding="utf-8") != tables[name]:
            raise ErrorAnalysisContractError(f"Tabela divergente em {path}")
    for path in (FIGURE_GROUPS_PATH, FIGURE_IMPORTANCE_PATH):
        if not path.exists() or path.stat().st_size == 0:
            raise ErrorAnalysisContractError(f"Figura ausente ou vazia em {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-analysis", action="store_true")
    action.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    study = run_error_analysis()
    if args.write_analysis:
        write_outputs(study)
        print(f"Análise de erros gravada em {RESULTS_PATH}")
        return
    check_outputs(study)
    print(
        f"Análise válida: {len(study.permutation_importance)} features, "
        f"Δ MAE da ablação=US$ {study.ablation.delta_mae_usd:,.2f}"
    )


if __name__ == "__main__":
    main()
