"""Pipeline creation for different machine learning models."""

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
import torch

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from typing import Literal
from .params import PARAM_GRID


def get_pipeline(
    model_name: Literal["LR", "XGB", "RF", "MLP", "CatBoost", "TabPFN", "RBF_SVC", "Linear_SVC"],
    continuous_cols: list,
    inner_folds: int,
    n_jobs: int,
    rnd_state: int
) -> tuple[ColumnTransformer, GridSearchCV]:
    """Get the sklearn pipeline with transformer and grid search.

    Parameters
    ----------
    model_name : str
        Name of the model to use. Options: LR, XGB, RF, MLP, CatBoost, TabPFN, RBF_SVC, Linear_SVC
    continuous_cols : list
        List of continuous column names for scaling
    inner_folds : int
        Number of folds for inner cross-validation
    n_jobs : int
        Number of parallel jobs
    rnd_state : int
        Random state for reproducibility

    Returns
    -------
    transformer : ColumnTransformer
        The transformer for scaling continuous features
    grid : GridSearchCV
        The grid search object with the model
    """

    inner_cv = StratifiedKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=rnd_state
    )

    transformer = ColumnTransformer(
        transformers=[
            (
                "scaler",
                StandardScaler(),
                continuous_cols
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    if model_name == "LR":
        classifier = LogisticRegression(
            solver="saga",
            max_iter=5000,
            random_state=rnd_state,
            class_weight="balanced",
            n_jobs=n_jobs,
        )
    elif model_name == "XGB":
        classifier = XGBClassifier(
            importance_type="gain",
            random_state=rnd_state,
            enable_categorical=True,
            n_jobs=n_jobs,
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
            n_jobs=n_jobs,
        )
    elif model_name == "CatBoost":
        classifier = CatBoostClassifier(
            random_state=rnd_state,
            verbose=False,
            allow_writing_files=False,
            thread_count=n_jobs,
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
            kernel='rbf',
            random_state=rnd_state,
            class_weight='balanced',
            probability=True,
        )
    elif model_name == "Linear_SVC":
        classifier = SVC(
            kernel='linear',
            random_state=rnd_state,
            class_weight='balanced',
            probability=True,
        )
    else:
        raise ValueError(
            f"Possible models are 'LR', 'XGB', 'RF', 'MLP', 'CatBoost', 'TabPFN', "
            f"'RBF_SVC', and 'Linear_SVC'. {model_name} was passed instead."
        )

    return transformer, GridSearchCV(
        estimator=classifier,
        param_grid=PARAM_GRID[model_name],
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
        return_train_score=True
    )