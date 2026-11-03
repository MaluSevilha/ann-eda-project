"""Contrato de dados da regressão, sem acesso público ao conjunto de teste.

Execute a partir da raiz do repositório:

    python -m docs.projects.regression.code.data --check

O módulo reproduz o split externo da EDA, separa validação a partir do treino
e ajusta o pipeline apenas no treino interno. Nenhuma métrica preditiva é
calculada aqui.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit

from docs.projects.eda.code.pipeline import (
    GROUP_COLUMN,
    MODEL_FEATURES,
    TARGET,
    build_preprocessor,
)

ROOT = Path(__file__).resolve().parents[4]
DATA_PATH = ROOT / "data/raw/college_majors_2026.csv"
MANIFEST_PATH = Path(__file__).resolve().parents[1] / "split_manifest.json"

EDA_SPLIT_SEED = 42
VALIDATION_SPLIT_SEED = 42
EDA_TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
RARE_CATEGORY_MIN_COUNT = 20


class DataContractError(RuntimeError):
    """Indica que os dados atuais não respeitam o contrato versionado."""


@dataclass(frozen=True)
class DevelopmentSplits:
    """Frames de desenvolvimento; deliberadamente não contém o teste."""

    train: pd.DataFrame
    validation: pd.DataFrame


@dataclass(frozen=True)
class PreparedDevelopmentData:
    """Dados transformados pelo pré-processador ajustado apenas no treino."""

    X_train: sparse.csr_matrix
    X_validation: sparse.csr_matrix
    y_train: np.ndarray
    y_validation: np.ndarray
    feature_names: tuple[str, ...]
    preprocessor: ColumnTransformer


def _require_dataset(path: Path) -> None:
    if not path.exists():
        raise DataContractError(
            f"Dataset ausente em {path}. Execute primeiro: python download_data.py"
        )


def _load_modeling_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carrega a população modelável definida na EDA."""

    _require_dataset(path)
    raw = pd.read_csv(path, low_memory=False)
    required = {"program_id", "institution_control", GROUP_COLUMN, TARGET, *MODEL_FEATURES}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise DataContractError(f"Colunas obrigatórias ausentes: {missing}")
    if raw["program_id"].duplicated().any():
        raise DataContractError("program_id deve ser único antes dos splits")
    return raw.loc[
        raw[TARGET].notna() & raw["institution_control"].ne("Foreign")
    ].copy()


def _split_all(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retorna treino, validação e teste somente para validação do contrato."""

    outer = GroupShuffleSplit(
        n_splits=1, test_size=EDA_TEST_SIZE, random_state=EDA_SPLIT_SEED
    )
    development_idx, test_idx = next(outer.split(data, groups=data[GROUP_COLUMN]))
    development = data.iloc[development_idx].copy()
    locked_test = data.iloc[test_idx].copy()

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=VALIDATION_SPLIT_SEED,
    )
    train_idx, validation_idx = next(
        inner.split(development, groups=development[GROUP_COLUMN])
    )
    train = development.iloc[train_idx].copy()
    validation = development.iloc[validation_idx].copy()
    return train, validation, locked_test


def load_development_splits(path: Path = DATA_PATH) -> DevelopmentSplits:
    """Reproduz treino e validação sem expor o teste aos experimentos."""

    train, validation, _locked_test = _split_all(_load_modeling_data(path))
    return DevelopmentSplits(train=train, validation=validation)


def _as_float32_csr(values: Any) -> sparse.csr_matrix:
    matrix = values.tocsr() if sparse.issparse(values) else sparse.csr_matrix(values)
    return matrix.astype(np.float32)


def _count_nan(matrix: sparse.csr_matrix) -> int:
    return int(np.isnan(matrix.data).sum())


def prepare_development_data(
    splits: DevelopmentSplits,
    *,
    min_category_frequency: int = RARE_CATEGORY_MIN_COUNT,
) -> PreparedDevelopmentData:
    """Ajusta o pipeline no treino e apenas transforma a validação."""

    preprocessor = build_preprocessor(
        min_category_frequency=min_category_frequency
    )
    X_train = _as_float32_csr(
        preprocessor.fit_transform(splits.train[MODEL_FEATURES])
    )
    X_validation = _as_float32_csr(
        preprocessor.transform(splits.validation[MODEL_FEATURES])
    )
    feature_names = tuple(map(str, preprocessor.get_feature_names_out()))

    errors: list[str] = []
    if X_train.shape[1] != X_validation.shape[1]:
        errors.append(
            f"larguras diferentes: treino={X_train.shape[1]}, "
            f"validação={X_validation.shape[1]}"
        )
    if len(feature_names) != X_train.shape[1]:
        errors.append(
            f"nomes={len(feature_names)}, colunas transformadas={X_train.shape[1]}"
        )
    if len(set(feature_names)) != len(feature_names):
        errors.append("nomes de features duplicados")
    train_nan = _count_nan(X_train)
    validation_nan = _count_nan(X_validation)
    if train_nan or validation_nan:
        errors.append(f"NaN no treino={train_nan}, validação={validation_nan}")
    if errors:
        raise DataContractError("Pré-processamento inválido: " + "; ".join(errors))

    return PreparedDevelopmentData(
        X_train=X_train,
        X_validation=X_validation,
        y_train=splits.train[TARGET].to_numpy(dtype=np.float32),
        y_validation=splits.validation[TARGET].to_numpy(dtype=np.float32),
        feature_names=feature_names,
        preprocessor=preprocessor,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _program_id_sha256(frame: pd.DataFrame) -> str:
    normalized = "\n".join(sorted(frame["program_id"].astype(str))) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _split_identity(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": len(frame),
        "institutions": int(frame[GROUP_COLUMN].nunique()),
        "program_id_sha256": _program_id_sha256(frame),
    }


def build_manifest(path: Path = DATA_PATH) -> dict[str, Any]:
    """Calcula identidades dos splits e invariantes do desenvolvimento."""

    data = _load_modeling_data(path)
    train, validation, locked_test = _split_all(data)
    prepared = prepare_development_data(
        DevelopmentSplits(train=train, validation=validation)
    )
    groups = {
        "train": set(train[GROUP_COLUMN]),
        "validation": set(validation[GROUP_COLUMN]),
        "test": set(locked_test[GROUP_COLUMN]),
    }
    return {
        "schema_version": 1,
        "dataset": {
            "path": "data/raw/college_majors_2026.csv",
            "sha256": _sha256_file(path),
            "modeling_rows": len(data),
            "filter": (
                "median_earnings_4yr_usd not null and "
                "institution_control != Foreign"
            ),
        },
        "split_parameters": {
            "group_column": GROUP_COLUMN,
            "eda_test_size": EDA_TEST_SIZE,
            "eda_seed": EDA_SPLIT_SEED,
            "validation_size_within_eda_train": VALIDATION_SIZE,
            "validation_seed": VALIDATION_SPLIT_SEED,
        },
        "splits": {
            "train": _split_identity(train),
            "validation": _split_identity(validation),
            "test_locked": _split_identity(locked_test),
        },
        "institution_overlaps": {
            "train_validation": len(groups["train"] & groups["validation"]),
            "train_test": len(groups["train"] & groups["test"]),
            "validation_test": len(groups["validation"] & groups["test"]),
        },
        "preprocessing": {
            "fit_split": "train",
            "transformed_splits": ["train", "validation"],
            "rare_category_min_count": RARE_CATEGORY_MIN_COUNT,
            "train_shape": list(prepared.X_train.shape),
            "validation_shape": list(prepared.X_validation.shape),
            "matrix_format": "csr",
            "dtype": str(prepared.X_train.dtype),
            "feature_count": len(prepared.feature_names),
            "unique_feature_names": len(set(prepared.feature_names)),
            "train_nan": _count_nan(prepared.X_train),
            "validation_nan": _count_nan(prepared.X_validation),
        },
        "test_policy": {
            "locked": True,
            "transformed": False,
            "target_statistics_computed": False,
            "predictive_metrics_computed": False,
        },
    }


def _manifest_differences(expected: Any, observed: Any, path: str = "manifest") -> list[str]:
    if type(expected) is not type(observed):
        return [f"{path}: tipo esperado={type(expected).__name__}, observado={type(observed).__name__}"]
    if isinstance(expected, dict):
        differences: list[str] = []
        for key in sorted(set(expected) | set(observed)):
            if key not in expected:
                differences.append(f"{path}.{key}: campo inesperado")
            elif key not in observed:
                differences.append(f"{path}.{key}: campo ausente")
            else:
                differences.extend(
                    _manifest_differences(expected[key], observed[key], f"{path}.{key}")
                )
        return differences
    if isinstance(expected, list):
        if len(expected) != len(observed):
            return [f"{path}: tamanho esperado={len(expected)}, observado={len(observed)}"]
        differences = []
        for index, (expected_item, observed_item) in enumerate(zip(expected, observed)):
            differences.extend(
                _manifest_differences(expected_item, observed_item, f"{path}[{index}]")
            )
        return differences
    return [] if expected == observed else [f"{path}: esperado={expected!r}, observado={observed!r}"]


def check_manifest(observed: dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    """Compara o contrato calculado com o manifesto versionado."""

    if not path.exists():
        raise DataContractError(
            f"Manifesto ausente em {path}. Use --write-manifest para criá-lo."
        )
    expected = json.loads(path.read_text(encoding="utf-8"))
    differences = _manifest_differences(expected, observed)
    if differences:
        details = "\n- ".join(differences)
        raise DataContractError(f"Contrato de dados divergente:\n- {details}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--check", action="store_true", help="compara os dados com o manifesto"
    )
    action.add_argument(
        "--write-manifest",
        action="store_true",
        help="grava o contrato calculado no manifesto versionado",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_manifest()
    if args.write_manifest:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Manifesto gravado em {MANIFEST_PATH}")
    else:
        check_manifest(manifest)
        print(
            "Contrato válido: "
            f"treino={manifest['splits']['train']['rows']}, "
            f"validação={manifest['splits']['validation']['rows']}, "
            f"teste bloqueado={manifest['splits']['test_locked']['rows']}"
        )


if __name__ == "__main__":
    main()
