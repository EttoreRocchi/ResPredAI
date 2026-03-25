"""Tests for signed feature importance direction."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from respredai.visualization.feature_importance import (
    _compute_directions_from_folds,
    compute_feature_direction,
    save_feature_importance_csv,
)


@pytest.fixture
def trained_lr():
    """Train a simple logistic regression model."""
    rng = np.random.RandomState(42)
    X = rng.randn(100, 3)
    # Feature 0 is risk (+), feature 1 is protective (-), feature 2 is weak
    y = ((X[:, 0] * 2 - X[:, 1] * 1.5 + rng.randn(100) * 0.3) > 0).astype(int)
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X, y)
    feature_names = ["risk_feat", "protect_feat", "weak_feat"]
    return model, X, y, feature_names


@pytest.fixture
def trained_rf():
    """Train a simple random forest model."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 3)
    y = ((X[:, 0] * 2 - X[:, 1] * 1.5 + rng.randn(200) * 0.3) > 0).astype(int)
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    feature_names = ["risk_feat", "protect_feat", "weak_feat"]
    return model, X, y, feature_names


class TestComputeFeatureDirection:
    def test_linear_model_returns_signed_series(self, trained_lr):
        model, X, _, feature_names = trained_lr
        result = compute_feature_direction(model, X, feature_names, model_name="LR")

        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == 3
        assert list(result.index) == feature_names

    def test_linear_model_correct_signs(self, trained_lr):
        model, X, _, feature_names = trained_lr
        result = compute_feature_direction(model, X, feature_names, model_name="LR")

        # risk_feat should have positive direction
        assert result["risk_feat"] > 0
        # protect_feat should have negative direction
        assert result["protect_feat"] < 0

    def test_tree_model_returns_signed_series(self, trained_rf):
        model, X, _, feature_names = trained_rf
        result = compute_feature_direction(model, X, feature_names, model_name="RF")

        assert result is not None
        assert isinstance(result, pd.Series)
        assert len(result) == 3

    def test_none_model_returns_none(self):
        result = compute_feature_direction(None, None, [], model_name="LR")
        assert result is None

    def test_none_data_returns_none(self, trained_lr):
        model, _, _, feature_names = trained_lr
        result = compute_feature_direction(model, None, feature_names, model_name="LR")
        assert result is None


class TestDirectionLabels:
    def test_compute_directions_from_folds(self, trained_lr):
        model, X, _, feature_names = trained_lr
        fold_models = [model, model]
        fold_test_data = [(X, feature_names), (X, feature_names)]
        fold_transformers = [None, None]

        result = _compute_directions_from_folds(
            fold_models,
            fold_test_data,
            fold_transformers,
            feature_names,
            "LR",
            seed=42,
        )

        assert result is not None
        assert set(result.values).issubset({"Risk (+)", "Protective (-)"})
        assert result["risk_feat"] == "Risk (+)"
        assert result["protect_feat"] == "Protective (-)"

    def test_empty_folds_returns_none(self):
        result = _compute_directions_from_folds(
            [None],
            [None],
            [None],
            ["a"],
            "LR",
            seed=42,
        )
        assert result is None


class TestDirectionInCSV:
    def test_csv_includes_direction_column(self, tmp_path, trained_lr):
        model, X, _, feature_names = trained_lr
        importances_df = pd.DataFrame({f: np.abs(np.random.randn(5)) for f in feature_names})
        directions = pd.Series(
            {"risk_feat": "Risk (+)", "protect_feat": "Protective (-)", "weak_feat": "Risk (+)"}
        )
        csv_path = tmp_path / "test_importance.csv"
        save_feature_importance_csv(
            importances_df, csv_path, method="native", directions=directions
        )

        loaded = pd.read_csv(csv_path)
        assert "Direction" in loaded.columns
        assert "Risk (+)" in loaded["Direction"].values
        assert "Protective (-)" in loaded["Direction"].values

    def test_csv_without_direction(self, tmp_path):
        importances_df = pd.DataFrame({"feat_a": [0.1, 0.2], "feat_b": [0.3, 0.4]})
        csv_path = tmp_path / "test_no_dir.csv"
        save_feature_importance_csv(importances_df, csv_path, method="native")

        loaded = pd.read_csv(csv_path)
        assert "Direction" not in loaded.columns
