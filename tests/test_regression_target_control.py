from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from docs.projects.regression.code import target_control as control
from docs.projects.regression.code.data import (
    load_development_splits,
    prepare_development_data,
)


def _tiny_prepared():
    rng = np.random.default_rng(7)
    X_train = rng.normal(size=(200, 3)).astype(np.float32)
    X_validation = rng.normal(size=(50, 3)).astype(np.float32)
    weights = np.asarray([0.3, -0.2, 0.1])
    return SimpleNamespace(
        X_train=sparse.csr_matrix(X_train),
        X_validation=sparse.csr_matrix(X_validation),
        y_train=np.exp(10.5 + X_train @ weights).astype(np.float32),
        y_validation=np.exp(10.5 + X_validation @ weights).astype(np.float32),
    )


def test_control_covers_grid_sorted_and_recovers_log_linear_signal():
    results = control.run_control(_tiny_prepared())
    assert {item.alpha for item in results} == set(control.RIDGE_ALPHAS)
    maes = [item.mae_usd for item in results]
    assert maes == sorted(maes)
    assert results[0].r2 > 0.99


def test_gain_share_uses_official_ridge_and_final_mlp(monkeypatch):
    monkeypatch.setattr(
        control,
        "_reference_metrics",
        lambda: (
            {"mae_usd": 100.0, "rmse_usd": 150.0, "r2": 0.7},
            {"mae_usd": 80.0, "rmse_usd": 120.0, "r2": 0.8},
        ),
    )
    payload = control.build_payload(
        (control.ControlResult(alpha=1.0, mae_usd=95.0, rmse_usd=140.0, r2=0.75),)
    )
    assert payload["test_accessed"] is False
    assert payload["changes_official_baseline_or_success_criterion"] is False
    assert payload["mae_gain_official_ridge_to_mlp_usd"] == 20.0
    assert payload["mae_gain_official_ridge_to_ridge_log1p_usd"] == 5.0
    assert payload["share_of_mlp_gain_matched_by_log1p_pct"] == 25.0


def test_check_tolerates_solver_noise_but_detects_real_divergence(tmp_path, monkeypatch):
    monkeypatch.setattr(control, "RESULTS_PATH", tmp_path / "control.json")
    monkeypatch.setattr(control, "TABLE_PATH", tmp_path / "control.md")
    monkeypatch.setattr(
        control,
        "_reference_metrics",
        lambda: (
            {"mae_usd": 110.0, "rmse_usd": 150.0, "r2": 0.7},
            {"mae_usd": 80.0, "rmse_usd": 120.0, "r2": 0.8},
        ),
    )
    result = control.ControlResult(alpha=1.0, mae_usd=100.0, rmse_usd=140.0, r2=0.75)
    payload = control.build_payload((result,))
    control.write_outputs(payload)

    def recomputed(**changes):
        return control.build_payload(
            (replace(result, **changes),)
        )

    control.check_outputs(recomputed(mae_usd=100.12))
    for divergent in (
        recomputed(mae_usd=150.0),
        recomputed(alpha=0.1),
        recomputed(r2=0.70),
    ):
        with pytest.raises(control.TargetControlError, match="divergente"):
            control.check_outputs(divergent)


def test_module_has_no_locked_test_loader_reference():
    source = control.Path(control.__file__).read_text(encoding="utf-8")
    assert "_load_locked_test_for_final_evaluation" not in source


def test_versioned_outputs_match_recomputed_control():
    prepared = prepare_development_data(load_development_splits())
    control.check_outputs(control.build_payload(control.run_control(prepared)))
