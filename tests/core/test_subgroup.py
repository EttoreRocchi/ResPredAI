"""Tests for subgroup performance evaluation."""

import warnings

import numpy as np
import pandas as pd
import pytest

from respredai.core.subgroup import compute_subgroup_metrics, save_subgroup_metrics


@pytest.fixture
def binary_predictions():
    """Create simple binary classification predictions."""
    rng = np.random.RandomState(42)
    n = 100
    y_true = rng.randint(0, 2, size=n)
    y_prob = np.column_stack([1 - rng.rand(n) * 0.5, rng.rand(n) * 0.5 + 0.5])
    # Make predictions somewhat correlated with truth
    y_prob[y_true == 1, 1] += 0.3
    y_prob = np.clip(y_prob, 0, 1)
    y_prob[:, 0] = 1 - y_prob[:, 1]
    y_pred = (y_prob[:, 1] >= 0.5).astype(int)
    return y_true, y_pred, y_prob


class TestComputeSubgroupMetrics:
    def test_basic_subgroup_computation(self, binary_predictions):
        y_true, y_pred, y_prob = binary_predictions
        groups = np.array(["A"] * 50 + ["B"] * 50)

        df = compute_subgroup_metrics(y_true, y_pred, y_prob, groups, "test_col")

        assert len(df) == 2
        assert set(df["Subgroup"].values) == {"A", "B"}
        assert "N" in df.columns
        assert "Prevalence" in df.columns
        assert "AUROC" in df.columns
        assert "F1 (weighted)" in df.columns
        assert "MCC" in df.columns
        assert "Brier Score" in df.columns
        assert df["N"].sum() == 100

    def test_single_subgroup(self, binary_predictions):
        y_true, y_pred, y_prob = binary_predictions
        groups = np.array(["All"] * 100)

        df = compute_subgroup_metrics(y_true, y_pred, y_prob, groups, "test_col")

        assert len(df) == 1
        assert df.iloc[0]["N"] == 100

    def test_small_subgroup_warns(self, binary_predictions):
        y_true, y_pred, y_prob = binary_predictions
        groups = np.array(["Large"] * 95 + ["Small"] * 5)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            df = compute_subgroup_metrics(y_true, y_pred, y_prob, groups, "test_col")
            small_warnings = [x for x in w if "Small" in str(x.message)]
            assert len(small_warnings) == 1

        assert len(df) == 2

    def test_nan_subgroup_values(self, binary_predictions):
        y_true, y_pred, y_prob = binary_predictions
        groups = np.array(["A"] * 40 + ["B"] * 40 + [None] * 20, dtype=object)

        df = compute_subgroup_metrics(y_true, y_pred, y_prob, groups, "test_col")

        assert "Unknown" in df["Subgroup"].values
        assert len(df) == 3

    def test_prevalence_calculation(self):
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        y_prob = np.column_stack([1 - y_true * 0.9, y_true * 0.9])
        groups = np.array(["A"] * 5 + ["B"] * 5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = compute_subgroup_metrics(y_true, y_pred, y_prob, groups, "test_col")

        a_row = df[df["Subgroup"] == "A"].iloc[0]
        b_row = df[df["Subgroup"] == "B"].iloc[0]
        # A has [0,0,0,1,1] -> prevalence = 2/5 = 0.4
        assert abs(a_row["Prevalence"] - 0.4) < 1e-6
        # B has [1,1,1,1,1] -> prevalence = 5/5 = 1.0
        assert abs(b_row["Prevalence"] - 1.0) < 1e-6


class TestSaveSubgroupMetrics:
    def test_saves_csv(self, tmp_path, binary_predictions):
        y_true, y_pred, y_prob = binary_predictions
        groups = np.array(["A"] * 50 + ["B"] * 50)

        df = compute_subgroup_metrics(y_true, y_pred, y_prob, groups, "test_col")
        output_path = tmp_path / "subgroup" / "test_subgroup.csv"
        save_subgroup_metrics(df, output_path)

        assert output_path.exists()
        loaded = pd.read_csv(output_path)
        assert len(loaded) == 2
        assert "Subgroup" in loaded.columns

    def test_creates_parent_dirs(self, tmp_path, binary_predictions):
        y_true, y_pred, y_prob = binary_predictions
        groups = np.array(["A"] * 100)

        df = compute_subgroup_metrics(y_true, y_pred, y_prob, groups, "test_col")
        output_path = tmp_path / "deep" / "nested" / "dir" / "test.csv"
        save_subgroup_metrics(df, output_path)

        assert output_path.exists()
