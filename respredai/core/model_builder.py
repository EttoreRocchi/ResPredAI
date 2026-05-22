"""Pipeline creation for different machine learning models."""

import logging
import os
from typing import Literal

import numpy as np
import torch
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from respredai.core.constants import AVAILABLE_MODELS
from respredai.core.params import PARAM_GRID

TABPFN_TOKEN_ENV = "TABPFN_TOKEN"


def ensure_tabpfn_available() -> None:
    """Verify that TabPFN is installed and an API token is configured.

    TabPFN is shipped as an optional extra (``pip install respredai[tabpfn]``)
    and the v3 model requires a PriorLabs API token, supplied via the
    ``TABPFN_TOKEN`` environment variable. Call this eagerly before starting
    any pipeline that includes ``TabPFN`` so the run fails fast on setup
    issues instead of mid-fold.

    Raises
    ------
    ImportError
        If the ``tabpfn`` package is not installed.
    RuntimeError
        If ``TABPFN_TOKEN`` is not set in the environment.
    """
    try:
        import tabpfn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "TabPFN is not installed. Install with `pip install respredai[tabpfn]` "
            "to enable the TabPFN model."
        ) from e
    if not os.environ.get(TABPFN_TOKEN_ENV):
        raise RuntimeError(
            "TabPFN v3 requires a PriorLabs API token. "
            f"Set the {TABPFN_TOKEN_ENV} environment variable "
            "(see https://priorlabs.ai/docs)."
        )


class _NaNSafeScaler(BaseEstimator, TransformerMixin):
    """StandardScaler that tolerates NaN values.

    Computes mean/std from non-NaN values during fit, and applies z-score
    normalization while preserving NaN positions. This enables distance-based
    imputers (e.g. KNNImputer) to operate on scale-normalized features,
    preventing features with larger magnitudes from dominating distances.
    """

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=np.float64)
        self.mean_ = np.nanmean(X_arr, axis=0)
        self.scale_ = np.nanstd(X_arr, axis=0)
        # Handle all-NaN columns (nanstd/nanmean return NaN)
        self.mean_[np.isnan(self.mean_)] = 0.0
        self.scale_[np.isnan(self.scale_)] = 1.0
        # Avoid division by zero for constant features
        self.scale_[self.scale_ == 0] = 1.0
        self.n_features_in_ = X_arr.shape[1]
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=np.float64)
        return (X_arr - self.mean_) / self.scale_

    def get_feature_names_out(self, input_features=None):
        """Support sklearn set_output API."""
        if input_features is not None:
            return np.array(input_features)
        return np.array([f"x{i}" for i in range(self.n_features_in_)])


def get_imputer(
    method: str,
    strategy: str = "mean",
    n_neighbors: int = 5,
    estimator: str = "bayesian_ridge",
    random_state: int = 42,
):
    """
    Get imputer based on configuration.

    Parameters
    ----------
    method : str
        Imputation method: "none", "simple", "knn", "iterative"
    strategy : str
        Strategy for SimpleImputer (mean, median, most_frequent)
    n_neighbors : int
        Number of neighbors for KNNImputer
    estimator : str
        Estimator for IterativeImputer: "bayesian_ridge" or "random_forest"
    random_state : int
        Random state for reproducibility

    Returns
    -------
    imputer or None
        Configured imputer or None if method is "none"
    """
    if method == "none":
        return None
    elif method == "simple":
        return SimpleImputer(strategy=strategy)
    elif method == "knn":
        return KNNImputer(n_neighbors=n_neighbors)
    elif method == "iterative":
        if estimator == "bayesian_ridge":
            est = BayesianRidge()
        else:  # random_forest (MissForest-style)
            est = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=1)
        return IterativeImputer(estimator=est, random_state=random_state, max_iter=10)
    else:
        raise ValueError(f"Unknown imputation method: {method}")


def _create_classifier(
    model_name: str,
    rnd_state: int,
    calibrate_probabilities: bool = False,
):
    """Instantiate the classifier for the given model name.

    Parameters
    ----------
    model_name : str
        One of LR, XGB, RF, MLP, CatBoost, TabPFN, RBF_SVC, Linear_SVC, KNN.
    rnd_state : int
        Random state for reproducibility.
    calibrate_probabilities : bool
        When True, SVC models disable internal Platt scaling to avoid
        double calibration with external CalibratedClassifierCV.
    """
    if model_name == "LR":
        return LogisticRegression(
            solver="saga",
            max_iter=5000,
            random_state=rnd_state,
            class_weight="balanced",
            n_jobs=1,
        )
    elif model_name == "XGB":
        return XGBClassifier(
            importance_type="gain",
            random_state=rnd_state,
            enable_categorical=True,
            n_jobs=1,
        )
    # Note: models in NO_CLASS_WEIGHT_MODELS do not support native
    # class weight balancing.
    elif model_name == "MLP":
        return MLPClassifier(
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=5000,
            shuffle=True,
            random_state=rnd_state,
        )
    elif model_name == "RF":
        return RandomForestClassifier(
            random_state=rnd_state,
            class_weight="balanced",
            n_jobs=1,
        )
    elif model_name == "CatBoost":
        return CatBoostClassifier(
            random_state=rnd_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            auto_class_weights="Balanced",
        )
    elif model_name == "TabPFN":
        ensure_tabpfn_available()
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion

        if not torch.cuda.is_available():
            logging.getLogger("respredai").warning("CUDA not available; TabPFN will use CPU.")
        return TabPFNClassifier.create_default_for_version(
            version=ModelVersion.V3,
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_estimators=8,
            random_state=rnd_state,
        )
    elif model_name == "RBF_SVC":
        # When external probability calibration is enabled, disable SVC's
        # internal Platt scaling to avoid double calibration.
        # CalibratedClassifierCV can calibrate from the decision function.
        return SVC(
            kernel="rbf",
            random_state=rnd_state,
            class_weight="balanced",
            probability=not calibrate_probabilities,
        )
    elif model_name == "Linear_SVC":
        return SVC(
            kernel="linear",
            random_state=rnd_state,
            class_weight="balanced",
            probability=not calibrate_probabilities,
        )
    elif model_name == "KNN":
        return KNeighborsClassifier(
            metric="euclidean",
            n_jobs=1,
        )
    else:
        raise ValueError(
            f"Possible models are {AVAILABLE_MODELS}. {model_name} was passed instead."
        )


def get_pipeline(
    model_name: Literal[
        "LR", "XGB", "RF", "MLP", "CatBoost", "TabPFN", "RBF_SVC", "Linear_SVC", "KNN"
    ],
    continuous_cols: list[str],
    inner_folds: int,
    n_jobs: int,
    rnd_state: int,
    use_groups: bool = False,
    imputation_method: str = "none",
    imputation_strategy: str = "mean",
    imputation_n_neighbors: int = 5,
    imputation_estimator: str = "bayesian_ridge",
    calibrate_probabilities: bool = False,
) -> tuple[ColumnTransformer, GridSearchCV]:
    """Get the sklearn pipeline with transformer and grid search.

    Parameters
    ----------
    model_name : str
        Name of the model to use. Options: LR, XGB, RF, MLP, CatBoost, TabPFN, RBF_SVC, Linear_SVC, KNN
    continuous_cols : list
        List of continuous column names for scaling
    inner_folds : int
        Number of folds for inner cross-validation
    n_jobs : int
        Number of parallel jobs
    rnd_state : int
        Random state for reproducibility
    use_groups : bool, optional
        Whether to use StratifiedGroupKFold instead of StratifiedKFold
    imputation_method : str, optional
        Method for missing data imputation (none, simple, knn, iterative)
    imputation_strategy : str, optional
        Strategy for SimpleImputer (mean, median, most_frequent)
    imputation_n_neighbors : int, optional
        Number of neighbors for KNNImputer
    imputation_estimator : str, optional
        Estimator for IterativeImputer (bayesian_ridge, random_forest)
    calibrate_probabilities : bool, optional
        Whether external probability calibration (CalibratedClassifierCV) will
        be applied. When True, SVC models are created with probability=False to
        avoid double calibration (SVC's internal Platt scaling + external
        calibration). CalibratedClassifierCV can calibrate directly from the
        decision function.

    Returns
    -------
    transformer : ColumnTransformer
        The transformer for scaling continuous features
    grid : GridSearchCV
        The grid search object with the model.
    """
    # Use StratifiedGroupKFold if groups are specified, otherwise StratifiedKFold
    if use_groups:
        inner_cv = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=rnd_state)
    else:
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=rnd_state)

    imputer = get_imputer(
        method=imputation_method,
        strategy=imputation_strategy,
        n_neighbors=imputation_n_neighbors,
        estimator=imputation_estimator,
        random_state=rnd_state,
    )

    # Build transformer with optional imputation
    if imputer is not None:
        # For distance-based imputers (KNN), pre-scale features so that
        # Euclidean distances are not dominated by high-magnitude features.
        if imputation_method == "knn":
            steps = [
                ("pre_scaler", _NaNSafeScaler()),
                ("imputer", imputer),
                ("scaler", StandardScaler()),
            ]
        else:
            steps = [("imputer", imputer), ("scaler", StandardScaler())]
        # When imputation is enabled, also impute categorical columns with
        # most-frequent strategy to prevent NaN propagation downstream.
        categorical_imputer = SimpleImputer(strategy="most_frequent")
        transformer = ColumnTransformer(
            transformers=[
                (
                    "continuous",
                    Pipeline(steps),
                    continuous_cols,
                )
            ],
            remainder=categorical_imputer,
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
    else:
        # Original transformer without imputation
        transformer = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), continuous_cols)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

    classifier = _create_classifier(model_name, rnd_state, calibrate_probabilities)

    return transformer, GridSearchCV(
        estimator=classifier,
        param_grid=PARAM_GRID[model_name],
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
        return_train_score=True,
    )
