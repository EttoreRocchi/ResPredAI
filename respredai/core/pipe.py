"""Pipeline creation for different machine learning models."""

from typing import List, Literal, Tuple

import torch
from catboost import CatBoostClassifier
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
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from xgboost import XGBClassifier

from respredai.core.params import PARAM_GRID


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


def get_pipeline(
    model_name: Literal[
        "LR", "XGB", "RF", "MLP", "CatBoost", "TabPFN", "RBF_SVC", "Linear_SVC", "KNN"
    ],
    continuous_cols: List[str],
    inner_folds: int,
    n_jobs: int,
    rnd_state: int,
    use_groups: bool = False,
    imputation_method: str = "none",
    imputation_strategy: str = "mean",
    imputation_n_neighbors: int = 5,
    imputation_estimator: str = "bayesian_ridge",
) -> Tuple[ColumnTransformer, GridSearchCV]:
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

    Returns
    -------
    transformer : ColumnTransformer
        The transformer for scaling continuous features
    grid : GridSearchCV
        The grid search object with the model
    """

    # Use StratifiedGroupKFold if groups are specified, otherwise StratifiedKFold
    if use_groups:
        inner_cv = StratifiedGroupKFold(n_splits=inner_folds, shuffle=True, random_state=rnd_state)
    else:
        inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=rnd_state)

    # Get imputer if imputation is enabled
    imputer = get_imputer(
        method=imputation_method,
        strategy=imputation_strategy,
        n_neighbors=imputation_n_neighbors,
        estimator=imputation_estimator,
        random_state=rnd_state,
    )

    # Build transformer with optional imputation
    if imputer is not None:
        # Create pipeline: impute then scale for continuous columns
        transformer = ColumnTransformer(
            transformers=[
                (
                    "continuous",
                    Pipeline([("imputer", imputer), ("scaler", StandardScaler())]),
                    continuous_cols,
                )
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")
    else:
        # Original transformer without imputation
        transformer = ColumnTransformer(
            transformers=[("scaler", StandardScaler(), continuous_cols)],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

    if model_name == "LR":
        classifier = LogisticRegression(
            solver="saga",
            max_iter=5000,
            random_state=rnd_state,
            class_weight="balanced",
            n_jobs=1,
        )
    elif model_name == "XGB":
        classifier = XGBClassifier(
            importance_type="gain",
            random_state=rnd_state,
            enable_categorical=True,
            n_jobs=1,
        )
    elif model_name == "MLP":
        classifier = MLPClassifier(
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=5000,
            shuffle=True,
            random_state=rnd_state,
        )
    elif model_name == "RF":
        classifier = RandomForestClassifier(
            random_state=rnd_state,
            class_weight="balanced",
            n_jobs=1,
        )
    elif model_name == "CatBoost":
        classifier = CatBoostClassifier(
            random_state=rnd_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=1,
            auto_class_weights="Balanced",
        )
    elif model_name == "TabPFN":
        classifier = TabPFNClassifier().create_default_for_version(
            version=ModelVersion.V2,
            device="cuda" if torch.cuda.is_available() else "cpu",
            n_estimators=8,
            random_state=rnd_state,
        )
    elif model_name == "RBF_SVC":
        classifier = SVC(
            kernel="rbf",
            random_state=rnd_state,
            class_weight="balanced",
            probability=True,
        )
    elif model_name == "Linear_SVC":
        classifier = SVC(
            kernel="linear",
            random_state=rnd_state,
            class_weight="balanced",
            probability=True,
        )
    elif model_name == "KNN":
        classifier = KNeighborsClassifier(
            metric="euclidean",
            n_jobs=1,
        )
    else:
        raise ValueError(
            f"Possible models are 'LR', 'XGB', 'RF', 'MLP', 'CatBoost', 'TabPFN', "
            f"'RBF_SVC', 'Linear_SVC', and 'KNN'. {model_name} was passed instead."
        )

    return transformer, GridSearchCV(
        estimator=classifier,
        param_grid=PARAM_GRID[model_name],
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
        return_train_score=True,
    )
