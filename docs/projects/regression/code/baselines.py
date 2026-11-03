"""Baselines obrigatórios escolhidos exclusivamente pela validação.

Execute a partir da raiz do repositório:

    python -m docs.projects.regression.code.baselines --check

O módulo usa apenas a interface pública de desenvolvimento. O conjunto de
teste não é carregado, transformado nem avaliado.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from docs.projects.regression.code.data import (
    PreparedDevelopmentData,
    load_development_splits,
    prepare_development_data,
)

REGRESSION_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = REGRESSION_DIR / "baseline_results.json"
TABLE_PATH = REGRESSION_DIR / "tables/baseline_validation.md"

RIDGE_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
SUCCESS_IMPROVEMENT = 0.15


class BaselineContractError(RuntimeError):
    """Indica divergência nos artefatos versionados dos baselines."""


@dataclass(frozen=True)
class BaselineResult:
    """Métricas de uma configuração na validação."""

    model: str
    configuration: str
    mae_usd: float
    rmse_usd: float
    r2: float


@dataclass(frozen=True)
class BaselineStudy:
    """Comparação ordenada e critério numérico para a futura MLP."""

    results: tuple[BaselineResult, ...]
    best_baseline: BaselineResult
    mlp_success_threshold_mae_usd: float


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula MAE, RMSE e R², rejeitando entradas inválidas."""

    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    predicted = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.size == 0 or true.shape != predicted.shape:
        raise ValueError(
            f"Shapes inválidos: y_true={true.shape}, y_pred={predicted.shape}"
        )
    if not (np.isfinite(true).all() and np.isfinite(predicted).all()):
        raise ValueError("Métricas exigem valores finitos")
    return {
        "mae_usd": float(mean_absolute_error(true, predicted)),
        "rmse_usd": float(np.sqrt(mean_squared_error(true, predicted))),
        "r2": float(r2_score(true, predicted)),
    }


def _evaluate(model: Any, configuration: str, prepared: PreparedDevelopmentData) -> BaselineResult:
    model.fit(prepared.X_train, prepared.y_train)
    metrics = regression_metrics(
        prepared.y_validation, model.predict(prepared.X_validation)
    )
    return BaselineResult(
        model=type(model).__name__,
        configuration=configuration,
        **metrics,
    )


def run_baselines(
    prepared: PreparedDevelopmentData | None = None,
) -> BaselineStudy:
    """Executa Dummy e Ridge no mesmo desenvolvimento pré-processado."""

    if prepared is None:
        prepared = prepare_development_data(load_development_splits())

    results = [
        _evaluate(
            DummyRegressor(strategy="mean"),
            "strategy=mean",
            prepared,
        )
    ]
    for alpha in RIDGE_ALPHAS:
        results.append(
            _evaluate(
                Ridge(
                    alpha=alpha,
                    solver="lsqr",
                    max_iter=10_000,
                    tol=1e-6,
                ),
                f"alpha={alpha:g}",
                prepared,
            )
        )

    ordered = tuple(
        sorted(results, key=lambda item: (item.mae_usd, item.model, item.configuration))
    )
    best = ordered[0]
    return BaselineStudy(
        results=ordered,
        best_baseline=best,
        mlp_success_threshold_mae_usd=(1.0 - SUCCESS_IMPROVEMENT) * best.mae_usd,
    )


def _rounded_result(result: BaselineResult) -> dict[str, Any]:
    return {
        "model": result.model,
        "configuration": result.configuration,
        "mae_usd": round(result.mae_usd, 2),
        "rmse_usd": round(result.rmse_usd, 2),
        "r2": round(result.r2, 6),
    }


def study_payload(study: BaselineStudy) -> dict[str, Any]:
    """Serializa apenas resultados estáveis e necessários ao relatório."""

    return {
        "schema_version": 1,
        "selection_split": "validation",
        "test_accessed": False,
        "primary_metric": "mae_usd",
        "ridge_alphas": list(RIDGE_ALPHAS),
        "results": [_rounded_result(result) for result in study.results],
        "best_baseline": _rounded_result(study.best_baseline),
        "mlp_success_improvement_pct": SUCCESS_IMPROVEMENT * 100,
        "mlp_success_threshold_mae_usd": round(
            study.mlp_success_threshold_mae_usd, 2
        ),
    }


def render_markdown_table(payload: dict[str, Any]) -> str:
    """Produz tabela determinística já ordenada pela métrica principal."""

    def format_decimal(value: float, digits: int) -> str:
        formatted = f"{value:,.{digits}f}"
        return formatted.replace(",", "_").replace(".", ",").replace("_", ".")

    lines = [
        "| Posição | Modelo | Configuração | MAE (US$) | RMSE (US$) | R² |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for position, result in enumerate(payload["results"], start=1):
        lines.append(
            f"| {position} | {result['model']} | `{result['configuration']}` | "
            f"{format_decimal(result['mae_usd'], 2)} | "
            f"{format_decimal(result['rmse_usd'], 2)} | "
            f"{format_decimal(result['r2'], 6)} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    study: BaselineStudy,
    *,
    results_path: Path = RESULTS_PATH,
    table_path: Path = TABLE_PATH,
) -> None:
    payload = study_payload(study)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    table_path.write_text(render_markdown_table(payload), encoding="utf-8")


def check_outputs(
    study: BaselineStudy,
    *,
    results_path: Path = RESULTS_PATH,
    table_path: Path = TABLE_PATH,
) -> None:
    """Confere separadamente JSON e tabela para indicar a divergência exata."""

    missing = [str(path) for path in (results_path, table_path) if not path.exists()]
    if missing:
        raise BaselineContractError(
            "Artefatos ausentes: " + ", ".join(missing) + ". Use --write-results."
        )
    expected_payload = study_payload(study)
    observed_payload = json.loads(results_path.read_text(encoding="utf-8"))
    if observed_payload != expected_payload:
        raise BaselineContractError(
            f"Resultados divergentes em {results_path}. Use --write-results após revisar a causa."
        )
    expected_table = render_markdown_table(expected_payload)
    observed_table = table_path.read_text(encoding="utf-8")
    if observed_table != expected_table:
        raise BaselineContractError(
            f"Tabela divergente em {table_path}. Use --write-results após revisar a causa."
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--write-results", action="store_true", help="regenera JSON e tabela"
    )
    action.add_argument(
        "--check", action="store_true", help="confere os artefatos versionados"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    study = run_baselines()
    if args.write_results:
        write_outputs(study)
        print(f"Resultados gravados em {RESULTS_PATH} e {TABLE_PATH}")
    else:
        check_outputs(study)
        best = study.best_baseline
        print(
            f"Baselines válidos: melhor={best.model} ({best.configuration}), "
            f"MAE=US$ {best.mae_usd:,.2f}, "
            f"limite MLP=US$ {study.mlp_success_threshold_mae_usd:,.2f}"
        )


if __name__ == "__main__":
    main()
