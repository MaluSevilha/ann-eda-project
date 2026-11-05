from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from docs.projects.eda.code.pipeline import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
)
from docs.projects.regression.code import error_analysis as analysis
from docs.projects.regression.code.data import DevelopmentSplits, PreparedDevelopmentData


def _prediction_frame():
    frame = pd.DataFrame(
        {
            "program_id": ["a", "b", "c", "d"],
            "credential_name": ["Bachelor", "Bachelor", "Master", "Rare"],
            "institution_control": ["Public", "Public", "Private", "Private"],
            "actual_usd": [40_000.0, 50_000.0, 70_000.0, 100_000.0],
            "mlp_seed_42_usd": [39_000.0, 52_000.0, 60_000.0, 80_000.0],
        }
    )
    frame["mlp_residual_usd"] = frame["actual_usd"] - frame["mlp_seed_42_usd"]
    frame["mlp_absolute_error_usd"] = frame["mlp_residual_usd"].abs()
    return frame


def test_group_metrics_include_size_bias_and_small_group_flag():
    results = analysis._group_metrics(_prediction_frame(), "credential_name")
    by_name = {item.group: item for item in results}
    bachelor = by_name["Bachelor"]
    assert bachelor.count == 2
    assert bachelor.mae_usd == pytest.approx(1_500.0)
    assert bachelor.rmse_usd == pytest.approx(np.sqrt(2_500_000.0))
    assert bachelor.median_absolute_error_usd == pytest.approx(1_500.0)
    assert bachelor.mean_residual_usd == pytest.approx(-500.0)
    assert bachelor.small_group is True


def test_target_bands_cover_every_row_once_in_documented_order():
    results = analysis._target_band_metrics(_prediction_frame())
    assert [item.group for item in results] == list(analysis.TARGET_BAND_LABELS)
    assert sum(item.count for item in results) == 4
    assert [item.count for item in results] == [1, 1, 1, 1]


def test_worst_examples_are_sorted_and_do_not_include_institution_id():
    worst = analysis._worst_examples(_prediction_frame())
    errors = [item["absolute_error_usd"] for item in worst]
    assert errors == sorted(errors, reverse=True)
    assert worst[0]["program_id"] == "d"
    assert all("opeid6" not in item for item in worst)


def _development_splits(rows=40):
    rng = np.random.default_rng(7)

    def frame(size):
        values = {
            feature: rng.normal(size=size) for feature in NUMERIC_FEATURES
        }
        values.update(
            {
                feature: np.where(np.arange(size) % 2, "category_a", "category_b")
                for feature in CATEGORICAL_FEATURES
            }
        )
        values[TARGET] = 50_000 + rng.normal(0, 2_000, size=size)
        return pd.DataFrame(values)

    return DevelopmentSplits(train=frame(rows), validation=frame(rows // 2))


def test_ablation_removes_feature_and_missing_indicator():
    splits = _development_splits()
    splits.train.loc[0, analysis.ABLATION_FEATURE] = np.nan
    prepared = analysis._prepare_ablation(splits)
    assert prepared.X_train.shape[0] == len(splits.train)
    assert prepared.X_validation.shape[0] == len(splits.validation)
    assert all(analysis.ABLATION_FEATURE not in name for name in prepared.feature_names)
    assert analysis.ABLATION_FEATURE not in analysis._build_ablation_preprocessor().transformers[0][2]


def test_permutation_importance_has_33_features_and_three_repeats(monkeypatch):
    splits = _development_splits(rows=8)
    prepared = SimpleNamespace(
        X_validation=sparse.csr_matrix(np.ones((4, 2), dtype=np.float32)),
        y_validation=np.asarray([1.0, 2.0, 3.0, 4.0]),
        preprocessor=SimpleNamespace(
            transform=lambda frame: sparse.csr_matrix(
                np.ones((len(frame), 2), dtype=np.float32)
            )
        ),
    )
    calls = {"count": 0}

    def fake_predict(*_args, **_kwargs):
        calls["count"] += 1
        return np.asarray([1.0, 2.0, 3.0, 4.0]) + calls["count"] * 0.01

    monkeypatch.setattr(analysis, "predict_usd", fake_predict)
    baseline, results = analysis._permutation_importance(
        splits, prepared, object(), object()
    )
    assert baseline > 0
    assert len(results) == len(MODEL_FEATURES) == 33
    assert {item.feature for item in results} == set(MODEL_FEATURES)
    assert all(len(item.delta_mae_repeats_usd) == 3 for item in results)
    assert calls["count"] == 1 + 33 * 3


def _synthetic_study():
    group = analysis.ErrorGroupResult(
        group="example",
        count=100,
        mae_usd=1_000.0,
        rmse_usd=1_200.0,
        median_absolute_error_usd=900.0,
        mean_residual_usd=100.0,
        small_group=False,
    )
    bands = tuple(
        analysis.ErrorGroupResult(
            group=label,
            count=25,
            mae_usd=1_000.0 + index,
            rmse_usd=1_200.0,
            median_absolute_error_usd=900.0,
            mean_residual_usd=100.0,
            small_group=True,
        )
        for index, label in enumerate(analysis.TARGET_BAND_LABELS)
    )
    importance = tuple(
        analysis.PermutationImportanceResult(
            feature=feature,
            delta_mae_mean_usd=float(33 - index),
            delta_mae_std_usd=1.0,
            delta_mae_repeats_usd=(1.0, 2.0, 3.0),
        )
        for index, feature in enumerate(MODEL_FEATURES)
    )
    return analysis.ErrorAnalysisStudy(
        source_sha256="abc",
        source_rows=100,
        credential_groups=(group,),
        control_groups=(group,),
        target_bands=bands,
        worst_examples=(
            {
                "program_id": "p1",
                "credential_name": "Bachelor",
                "institution_control": "Public",
                "actual_usd": 10_000.0,
                "predicted_usd": 8_000.0,
                "residual_usd": 2_000.0,
                "absolute_error_usd": 2_000.0,
            },
        ),
        baseline_validation_mae_usd=1_000.0,
        permutation_importance=importance,
        ablation=analysis.AblationResult(
            removed_feature=analysis.ABLATION_FEATURE,
            full_input_dim=420,
            ablated_input_dim=418,
            full_mae_usd=1_000.0,
            ablated_mae_usd=1_100.0,
            delta_mae_usd=100.0,
            relative_change_pct=10.0,
            ablated_rmse_usd=1_300.0,
            ablated_r2=0.8,
            best_epoch=4,
            stopped_epoch=8,
        ),
    )


def test_artifacts_are_deterministic_and_detect_divergence(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "RESULTS_PATH", tmp_path / "analysis.json")
    monkeypatch.setattr(analysis, "FIGURE_GROUPS_PATH", tmp_path / "groups.png")
    monkeypatch.setattr(analysis, "FIGURE_IMPORTANCE_PATH", tmp_path / "importance.png")
    table_paths = {name: tmp_path / f"{name}.md" for name in analysis.TABLE_PATHS}
    monkeypatch.setattr(analysis, "TABLE_PATHS", table_paths)
    study = _synthetic_study()
    analysis.write_outputs(study)
    analysis.check_outputs(study)
    assert all(path.stat().st_size > 0 for path in table_paths.values())
    payload = json.loads(analysis.RESULTS_PATH.read_text())
    payload["test_source"]["raw_test_reopened"] = True
    analysis.RESULTS_PATH.write_text(json.dumps(payload))
    with pytest.raises(analysis.ErrorAnalysisContractError, match="divergente"):
        analysis.check_outputs(study)


def test_module_has_no_locked_test_loader_reference():
    source = (analysis.Path(analysis.__file__)).read_text(encoding="utf-8")
    assert "_load_locked_test_for_final_evaluation" not in source
