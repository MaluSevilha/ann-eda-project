from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

from docs.projects.regression.code import data as regression_data


@pytest.fixture(scope="session")
def modeling_data():
    return regression_data._load_modeling_data()


@pytest.fixture(scope="session")
def all_splits(modeling_data):
    return regression_data._split_all(modeling_data)


@pytest.fixture(scope="session")
def prepared(all_splits):
    train, validation, _locked_test = all_splits
    splits = regression_data.DevelopmentSplits(train=train, validation=validation)
    return regression_data.prepare_development_data(splits)


@pytest.fixture(scope="session")
def generated_manifest():
    return regression_data.build_manifest()


def test_modeling_population_matches_eda(modeling_data):
    assert len(modeling_data) == 58_063
    assert modeling_data[regression_data.TARGET].notna().all()
    assert not modeling_data["institution_control"].eq("Foreign").any()
    assert modeling_data["program_id"].is_unique


def test_split_identities_and_no_overlap(all_splits):
    train, validation, locked_test = all_splits
    expected = {
        "train": (
            train,
            36_565,
            2_969,
            "387f1cc705fafc49ab5db640685be8888c235029105de1d170ff518a870ab2f9",
        ),
        "validation": (
            validation,
            9_758,
            743,
            "025e39c26cc44251af79b487b765f0d9757cecb579c96017235ad571f7556ffb",
        ),
        "test": (
            locked_test,
            11_740,
            928,
            "30d4a2c9a160f5b33fa20196cdfe69888beaa3f60fd7243b1f84b8586dc39e95",
        ),
    }
    for frame, rows, institutions, digest in expected.values():
        assert len(frame) == rows
        assert frame[regression_data.GROUP_COLUMN].nunique() == institutions
        assert regression_data._program_id_sha256(frame) == digest

    group_sets = [set(frame[regression_data.GROUP_COLUMN]) for frame, *_ in expected.values()]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])

    ids = [set(frame["program_id"]) for frame, *_ in expected.values()]
    assert ids[0].isdisjoint(ids[1])
    assert ids[0].isdisjoint(ids[2])
    assert ids[1].isdisjoint(ids[2])
    assert len(set.union(*ids)) == 58_063


def test_split_is_deterministic(modeling_data, all_splits):
    repeated = regression_data._split_all(modeling_data)
    for first, second in zip(all_splits, repeated):
        assert first["program_id"].tolist() == second["program_id"].tolist()


def test_public_loader_does_not_expose_test():
    splits = regression_data.load_development_splits()
    assert set(splits.__dataclass_fields__) == {"train", "validation"}


def test_prepare_calls_fit_only_for_train(monkeypatch):
    calls = []

    class SpyPreprocessor:
        def fit_transform(self, frame):
            calls.append(("fit_transform", frame.index.tolist()))
            return sparse.csr_matrix(np.ones((len(frame), 1)))

        def transform(self, frame):
            calls.append(("transform", frame.index.tolist()))
            return sparse.csr_matrix(np.ones((len(frame), 1)))

        def get_feature_names_out(self):
            return np.asarray(["feature"])

    monkeypatch.setattr(
        regression_data, "build_preprocessor", lambda **_kwargs: SpyPreprocessor()
    )
    columns = regression_data.MODEL_FEATURES + [regression_data.TARGET]
    train = regression_data.pd.DataFrame(
        [{column: 1 for column in columns}], index=[10]
    )
    validation = regression_data.pd.DataFrame(
        [{column: 2 for column in columns}], index=[20]
    )
    regression_data.prepare_development_data(
        regression_data.DevelopmentSplits(train=train, validation=validation)
    )
    assert calls == [("fit_transform", [10]), ("transform", [20])]


def test_prepared_development_contract(prepared):
    assert sparse.isspmatrix_csr(prepared.X_train)
    assert sparse.isspmatrix_csr(prepared.X_validation)
    assert prepared.X_train.shape == (36_565, 420)
    assert prepared.X_validation.shape == (9_758, 420)
    assert prepared.X_train.dtype == np.float32
    assert prepared.X_validation.dtype == np.float32
    assert len(prepared.feature_names) == 420
    assert len(set(prepared.feature_names)) == 420
    assert not np.isnan(prepared.X_train.data).any()
    assert not np.isnan(prepared.X_validation.data).any()
    assert prepared.y_train.shape == (36_565,)
    assert prepared.y_validation.shape == (9_758,)


def test_versioned_manifest_matches_data(generated_manifest):
    versioned = json.loads(
        regression_data.MANIFEST_PATH.read_text(encoding="utf-8")
    )
    assert generated_manifest == versioned
    regression_data.check_manifest(generated_manifest)


def test_torch_is_available():
    import torch

    assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:1])) >= (2,)
