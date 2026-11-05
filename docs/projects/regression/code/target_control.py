"""Controle: quanto do ganho da MLP vem apenas de treinar em ``log1p``?

Execute a partir da raiz do repositório:

    python -m docs.projects.regression.code.target_control --check

O Ridge oficial aprende em dólares e a MLP em ``log1p``. Este módulo repete a
mesma grade de ``alpha`` com o alvo transformado como na MLP e mede o erro em
dólares. Usa somente treino e validação e não altera o baseline oficial nem o
critério de sucesso registrado antes do treinamento.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.linear_model import Ridge

from docs.projects.regression.code.baselines import (
    RESULTS_PATH as BASELINE_RESULTS_PATH,
    RIDGE_ALPHAS,
    regression_metrics,
)
from docs.projects.regression.code.data import (
    PreparedDevelopmentData,
    load_development_splits,
    prepare_development_data,
)
from docs.projects.regression.code.final_config import FINAL_CONFIG_PATH
from docs.projects.regression.code.train import TargetTransformer

REGRESSION_DIR = Path(__file__).resolve().parents[1]
RESULTS_PATH = REGRESSION_DIR / "target_control.json"
TABLE_PATH = REGRESSION_DIR / "tables" / "target_control_validation.md"

# O solver iterativo do Ridge varia alguns centavos entre ambientes; a
# conclusão do controle não depende dessa ordem de grandeza.
USD_TOLERANCE = 1.0
R2_TOLERANCE = 1e-4
DERIVED_KEYS = (
    "mae_gain_official_ridge_to_ridge_log1p_usd",
    "share_of_mlp_gain_matched_by_log1p_pct",
)


class TargetControlError(RuntimeError):
    """Indica divergência nos artefatos versionados do controle."""


@dataclass(frozen=True)
class ControlResult:
    alpha: float
    mae_usd: float
    rmse_usd: float
    r2: float


def run_control(prepared: PreparedDevelopmentData) -> tuple[ControlResult, ...]:
    """Ajusta a grade em ``log1p`` e ordena pelo MAE de validação em dólares."""

    transformer = TargetTransformer("log1p").fit(prepared.y_train)
    y_train = transformer.transform(prepared.y_train)
    results = []
    for alpha in RIDGE_ALPHAS:
        model = Ridge(alpha=alpha, solver="lsqr", max_iter=10_000, tol=1e-6)
        model.fit(prepared.X_train, y_train)
        predicted = transformer.inverse_transform(model.predict(prepared.X_validation))
        results.append(
            ControlResult(
                alpha=alpha,
                **regression_metrics(prepared.y_validation, predicted),
            )
        )
    return tuple(sorted(results, key=lambda item: (item.mae_usd, item.alpha)))


def _reference_metrics() -> tuple[dict[str, Any], dict[str, Any]]:
    """Lê do versionado o Ridge oficial e a MLP final, ambos na validação."""

    baseline = json.loads(BASELINE_RESULTS_PATH.read_text(encoding="utf-8"))
    config = json.loads(FINAL_CONFIG_PATH.read_text(encoding="utf-8"))
    return baseline["best_baseline"], config["selection"]["validation_metrics_usd"]


def build_payload(results: tuple[ControlResult, ...]) -> dict[str, Any]:
    official, mlp = _reference_metrics()
    best = results[0]
    total_gain = official["mae_usd"] - mlp["mae_usd"]
    target_gain = official["mae_usd"] - round(best.mae_usd, 2)
    return {
        "schema_version": 1,
        "selection_split": "validation",
        "test_accessed": False,
        "changes_official_baseline_or_success_criterion": False,
        "official_ridge_dollars": official,
        "mlp_final": mlp,
        "ridge_log1p_grid": [
            {
                "alpha": item.alpha,
                "mae_usd": round(item.mae_usd, 2),
                "rmse_usd": round(item.rmse_usd, 2),
                "r2": round(item.r2, 6),
            }
            for item in results
        ],
        "mae_gain_official_ridge_to_mlp_usd": round(total_gain, 2),
        "mae_gain_official_ridge_to_ridge_log1p_usd": round(target_gain, 2),
        "share_of_mlp_gain_matched_by_log1p_pct": round(
            100.0 * target_gain / total_gain, 2
        ),
    }


def _number(value: float, digits: int = 2) -> str:
    formatted = f"{value:,.{digits}f}"
    return formatted.replace(",", "_").replace(".", ",").replace("_", ".")


def render_table(payload: dict[str, Any]) -> str:
    best = payload["ridge_log1p_grid"][0]
    rows = (
        ("Ridge oficial", "dólares", "`alpha=1`", payload["official_ridge_dollars"]),
        ("Ridge de controle", "`log1p`", f"`alpha={best['alpha']:g}`", best),
        ("MLP final", "`log1p`", "`low_lr_regularized`", payload["mlp_final"]),
    )
    lines = [
        "| Modelo | Alvo no treino | Configuração | MAE (US$) | RMSE (US$) | R² |",
        "|---|---|---|---:|---:|---:|",
    ]
    for model, target, configuration, metrics in rows:
        lines.append(
            f"| {model} | {target} | {configuration} | "
            f"{_number(metrics['mae_usd'])} | {_number(metrics['rmse_usd'])} | "
            f"{_number(metrics['r2'], 6)} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(payload: dict[str, Any]) -> None:
    RESULTS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    TABLE_PATH.write_text(render_table(payload), encoding="utf-8")


def _row_matches(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
    return (
        expected["alpha"] == observed["alpha"]
        and abs(expected["mae_usd"] - observed["mae_usd"]) <= USD_TOLERANCE
        and abs(expected["rmse_usd"] - observed["rmse_usd"]) <= USD_TOLERANCE
        and abs(expected["r2"] - observed["r2"]) <= R2_TOLERANCE
    )


def _payload_matches(expected: dict[str, Any], observed: dict[str, Any]) -> bool:
    def stable(payload: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in payload.items()
            if key != "ridge_log1p_grid" and key not in DERIVED_KEYS
        }

    expected_grid = expected["ridge_log1p_grid"]
    observed_grid = observed["ridge_log1p_grid"]
    return (
        stable(expected) == stable(observed)
        and len(expected_grid) == len(observed_grid)
        and all(map(_row_matches, expected_grid, observed_grid))
        and all(
            abs(expected[key] - observed[key]) <= USD_TOLERANCE
            for key in DERIVED_KEYS
        )
    )


def check_outputs(expected: dict[str, Any]) -> None:
    if not RESULTS_PATH.exists() or not TABLE_PATH.exists():
        raise TargetControlError("Artefatos ausentes. Use --write-results.")
    observed = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if not _payload_matches(expected, observed):
        raise TargetControlError(f"Controle divergente em {RESULTS_PATH}")
    if TABLE_PATH.read_text(encoding="utf-8") != render_table(observed):
        raise TargetControlError(f"Tabela divergente em {TABLE_PATH}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write-results", action="store_true")
    action.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_payload(run_control(prepare_development_data(load_development_splits())))
    if args.write_results:
        write_outputs(payload)
        print(f"Controle gravado em {RESULTS_PATH} e {TABLE_PATH}")
        return
    check_outputs(payload)
    best = payload["ridge_log1p_grid"][0]
    print(
        f"Controle válido: Ridge log1p alpha={best['alpha']:g}, "
        f"MAE=US$ {best['mae_usd']:,.2f}"
    )


if __name__ == "__main__":
    main()
