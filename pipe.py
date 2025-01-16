from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from typing import Literal

from params import PARAM_GRID


def get_pipeline(model_name: Literal["LR", "XGB", "RF", "MLP"], continuous_cols: list, inner_folds: int, n_jobs: int, rnd_state: int) -> tuple[ColumnTransformer, GridSearchCV]:
    """Get the `sklearn.pipeline`.

    Parameters
    ----------

    - 
    
    Returns
    -------
    
    pipeline = `sklearn.Pipeline`

        The pipeline composed of the scaler (if needed) and the model.
    """

    inner_cv = StratifiedKFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=rnd_state
    )

    transformer = ColumnTransformer(
        transformers = [
            (
                "scaler",
                StandardScaler(),
                continuous_cols
            )
        ],
        remainder = "passthrough",
        verbose_feature_names_out = False
    ).set_output(transform="pandas")

    if model_name == "LR":
        classifier = LogisticRegression(
            solver="saga",
            max_iter=5000,
            random_state = rnd_state,
            class_weight = "balanced",
            n_jobs=n_jobs
        )
    elif model_name == "XGB":
        classifier = XGBClassifier(
            importance_type = "gain",
            random_state = rnd_state,
            enable_categorical=True,
            n_jobs=n_jobs,
        )
    elif model_name == "MLP":
        classifier = MLPClassifier(
            solver = "adam",
            learning_rate = "adaptive",
            learning_rate_init = 0.001,
            max_iter = 5000,
            shuffle = True,
            random_state = rnd_state
        )
    else:
        raise ValueError(
            f"Possible models are 'LR', 'XGB', and 'MLP'. {model_name} is passed, instead. "
        )

    return transformer, GridSearchCV(
        estimator=classifier,
        param_grid=PARAM_GRID[model_name],
        cv=inner_cv,
        scoring="roc_auc",
        n_jobs=n_jobs,
        return_train_score=True
    )