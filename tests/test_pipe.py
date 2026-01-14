"""Unit tests for pipe.py - pipeline and imputation utilities."""

import numpy as np
import pandas as pd
import pytest
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer

from respredai.core.pipe import get_imputer, get_pipeline


class TestGetImputer:
    """Unit tests for get_imputer function."""

    def test_imputer_none_returns_none(self):
        """Test that method='none' returns None."""
        imputer = get_imputer(method="none")
        assert imputer is None

    def test_imputer_simple_mean(self):
        """Test SimpleImputer with mean strategy."""
        imputer = get_imputer(method="simple", strategy="mean")
        assert isinstance(imputer, SimpleImputer)
        assert imputer.strategy == "mean"

    def test_imputer_simple_median(self):
        """Test SimpleImputer with median strategy."""
        imputer = get_imputer(method="simple", strategy="median")
        assert isinstance(imputer, SimpleImputer)
        assert imputer.strategy == "median"

    def test_imputer_simple_most_frequent(self):
        """Test SimpleImputer with most_frequent strategy."""
        imputer = get_imputer(method="simple", strategy="most_frequent")
        assert isinstance(imputer, SimpleImputer)
        assert imputer.strategy == "most_frequent"

    def test_imputer_knn(self):
        """Test KNNImputer with custom n_neighbors."""
        imputer = get_imputer(method="knn", n_neighbors=3)
        assert isinstance(imputer, KNNImputer)
        assert imputer.n_neighbors == 3

    def test_imputer_knn_default_neighbors(self):
        """Test KNNImputer with default n_neighbors."""
        imputer = get_imputer(method="knn")
        assert isinstance(imputer, KNNImputer)
        assert imputer.n_neighbors == 5

    def test_imputer_iterative_bayesian_ridge(self):
        """Test IterativeImputer with BayesianRidge estimator."""
        imputer = get_imputer(method="iterative", estimator="bayesian_ridge")
        assert isinstance(imputer, IterativeImputer)

    def test_imputer_iterative_random_forest(self):
        """Test IterativeImputer with RandomForest estimator (MissForest-style)."""
        imputer = get_imputer(method="iterative", estimator="random_forest")
        assert isinstance(imputer, IterativeImputer)

    def test_imputer_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown imputation method"):
            get_imputer(method="unknown_method")

    def test_imputer_simple_actually_imputes(self):
        """Test that SimpleImputer actually fills missing values."""
        imputer = get_imputer(method="simple", strategy="mean")
        X = np.array([[1, 2], [np.nan, 3], [7, 6]])
        X_imputed = imputer.fit_transform(X)
        assert not np.isnan(X_imputed).any()
        # Mean of column 0 is (1+7)/2 = 4
        assert X_imputed[1, 0] == 4.0

    def test_imputer_knn_actually_imputes(self):
        """Test that KNNImputer actually fills missing values."""
        imputer = get_imputer(method="knn", n_neighbors=2)
        X = np.array([[1, 2], [np.nan, 3], [7, 6], [4, 5]])
        X_imputed = imputer.fit_transform(X)
        assert not np.isnan(X_imputed).any()


class TestGetPipeline:
    """Unit tests for get_pipeline function."""

    @pytest.mark.parametrize(
        "model_name",
        ["LR", "RF", "XGB", "MLP", "CatBoost", "RBF_SVC", "Linear_SVC", "KNN"],
    )
    def test_get_pipeline_returns_tuple(self, model_name):
        """Test that get_pipeline returns transformer and grid search."""
        transformer, grid = get_pipeline(
            model_name=model_name,
            continuous_cols=["col1", "col2"],
            inner_folds=2,
            n_jobs=1,
            rnd_state=42,
        )
        assert transformer is not None
        assert grid is not None

    def test_get_pipeline_invalid_model_raises(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Possible models are"):
            get_pipeline(
                model_name="InvalidModel",
                continuous_cols=["col1"],
                inner_folds=2,
                n_jobs=1,
                rnd_state=42,
            )

    def test_get_pipeline_with_imputation(self):
        """Test pipeline with imputation enabled."""
        transformer, grid = get_pipeline(
            model_name="LR",
            continuous_cols=["col1"],
            inner_folds=2,
            n_jobs=1,
            rnd_state=42,
            imputation_method="simple",
            imputation_strategy="mean",
        )
        # Transformer should have imputation in the pipeline
        assert transformer is not None

    def test_get_pipeline_uses_stratified_group_kfold(self):
        """Test that use_groups=True uses StratifiedGroupKFold."""
        from sklearn.model_selection import StratifiedGroupKFold

        _, grid = get_pipeline(
            model_name="LR",
            continuous_cols=["col1"],
            inner_folds=3,
            n_jobs=1,
            rnd_state=42,
            use_groups=True,
        )
        assert isinstance(grid.cv, StratifiedGroupKFold)

    def test_get_pipeline_uses_stratified_kfold(self):
        """Test that use_groups=False uses StratifiedKFold."""
        from sklearn.model_selection import StratifiedKFold

        _, grid = get_pipeline(
            model_name="LR",
            continuous_cols=["col1"],
            inner_folds=3,
            n_jobs=1,
            rnd_state=42,
            use_groups=False,
        )
        assert isinstance(grid.cv, StratifiedKFold)

    def test_transformer_scales_continuous_columns(self):
        """Test that transformer scales continuous columns."""
        transformer, _ = get_pipeline(
            model_name="LR",
            continuous_cols=["continuous"],
            inner_folds=2,
            n_jobs=1,
            rnd_state=42,
        )

        # Create test data
        df = pd.DataFrame({"continuous": [1.0, 2.0, 3.0, 4.0, 5.0], "other": [0, 1, 0, 1, 0]})

        transformed = transformer.fit_transform(df)
        # Check that continuous column is scaled (mean ~0, std ~1)
        cont_values = transformed["continuous"].values
        assert abs(cont_values.mean()) < 0.1
        assert abs(cont_values.std() - 1.0) < 0.1

    def test_transformer_passes_through_other_columns(self):
        """Test that non-continuous columns are passed through."""
        transformer, _ = get_pipeline(
            model_name="LR",
            continuous_cols=["continuous"],
            inner_folds=2,
            n_jobs=1,
            rnd_state=42,
        )

        df = pd.DataFrame({"continuous": [1.0, 2.0, 3.0], "categorical": [0, 1, 0]})

        transformed = transformer.fit_transform(df)
        # Categorical column should be unchanged
        assert "categorical" in transformed.columns
        assert list(transformed["categorical"]) == [0, 1, 0]

    def test_pipeline_with_knn_imputation(self):
        """Test pipeline with KNN imputation on data with missing values."""
        transformer, _ = get_pipeline(
            model_name="RF",
            continuous_cols=["col1", "col2"],
            inner_folds=2,
            n_jobs=1,
            rnd_state=42,
            imputation_method="knn",
            imputation_n_neighbors=2,
        )

        df = pd.DataFrame(
            {
                "col1": [1.0, np.nan, 3.0, 4.0],
                "col2": [2.0, 3.0, np.nan, 5.0],
                "other": [0, 1, 1, 0],
            }
        )

        transformed = transformer.fit_transform(df)
        # Should have no NaN values after transformation
        assert not transformed.isna().any().any()


class TestPipelineIntegration:
    """Integration tests for pipeline components."""

    def test_lr_pipeline_fits_and_predicts(self):
        """Test that LR pipeline can fit and predict on simple data."""
        transformer, grid = get_pipeline(
            model_name="LR",
            continuous_cols=["x1", "x2"],
            inner_folds=2,
            n_jobs=1,
            rnd_state=42,
        )

        # Create simple linearly separable data
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "x1": np.concatenate([np.random.randn(20) - 2, np.random.randn(20) + 2]),
                "x2": np.concatenate([np.random.randn(20) - 2, np.random.randn(20) + 2]),
            }
        )
        y = np.array([0] * 20 + [1] * 20)

        X_scaled = transformer.fit_transform(X)
        grid.fit(X_scaled, y)

        predictions = grid.predict(X_scaled)
        # Should achieve reasonable accuracy on linearly separable data
        accuracy = (predictions == y).mean()
        assert accuracy > 0.8

    def test_rf_pipeline_fits_and_predicts(self):
        """Test that RF pipeline can fit and predict."""
        transformer, grid = get_pipeline(
            model_name="RF",
            continuous_cols=["x1"],
            inner_folds=2,
            n_jobs=1,
            rnd_state=42,
        )

        X = pd.DataFrame({"x1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        X_scaled = transformer.fit_transform(X)
        grid.fit(X_scaled, y)

        predictions = grid.predict(X_scaled)
        assert len(predictions) == len(y)

    def test_knn_pipeline_fits_and_predicts(self):
        """Test that KNN pipeline can fit and predict."""
        import warnings

        transformer, grid = get_pipeline(
            model_name="KNN",
            continuous_cols=["x1", "x2"],
            inner_folds=2,
            n_jobs=1,
            rnd_state=42,
        )

        # Use larger sample size to accommodate KNN param grid (n_neighbors up to 21)
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "x1": np.concatenate([np.random.randn(50), np.random.randn(50) + 3]),
                "x2": np.concatenate([np.random.randn(50), np.random.randn(50) + 3]),
            }
        )
        y = np.array([0] * 50 + [1] * 50)

        X_scaled = transformer.fit_transform(X)

        # Suppress expected warnings from GridSearchCV when some n_neighbors values are too large
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            grid.fit(X_scaled, y)

        proba = grid.predict_proba(X_scaled)
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
