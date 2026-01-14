"""Parameter grids for hyperparameter tuning."""

from typing import Dict, List, Union

import numpy as np

PARAM_GRID: Dict[str, Union[Dict, List[Dict]]] = {
    "LR": [
        {
            "penalty": [None],
        },
        {"penalty": ["l1", "l2"], "C": np.logspace(-2, 4, 10)},
        {
            "penalty": ["elasticnet"],
            "C": np.logspace(-2, 4, 10),
            "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
    ],
    "MLP": {
        "hidden_layer_sizes": [
            (64, 32, 16),
            (64, 32),
            (32, 16, 8),
            (32, 16),
            (16, 8),
            (64,),
            (32,),
        ],
        "activation": ["relu", "tanh", "logistic"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
    },
    "XGB": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7, 9],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.3],
    },
    "RF": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", None],
    },
    "CatBoost": {
        "iterations": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [3, 5, 7, 9],
        "l2_leaf_reg": [1, 3, 5, 7, 9],
        "border_count": [32, 64, 128],
        "min_data_in_leaf": [1, 5, 10],
    },
    "TabPFN": {},
    "RBF_SVC": {
        "C": np.logspace(-2, 4, 10),
        "gamma": ["scale", "auto"] + list(np.logspace(-4, 1, 8)),
    },
    "Linear_SVC": {"C": np.logspace(-2, 4, 10)},
    "KNN": {
        "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
    },
}
