"""Parameter grids for hyperparameter tuning."""

import numpy as np

PARAM_GRID = {
    "LR": [
        {
            "penalty": [None],
        },
        {
            "penalty": ["l1", "l2"],
            "C": np.logspace(-1, 4, 6)
        },
        {
            "penalty": ["elasticnet"],
            "C": np.logspace(-1, 4, 6),
            "l1_ratio": [0.5]
        }
    ],
    "MLP": {
        "hidden_layer_sizes": [(32, 16, 8), (32, 16), (16, 8)],
        "activation": ['relu', 'logistic'],
    },
    "XGB": {
        "n_estimators": [25, 50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5, 7]
    },
    "RF": {
        "n_estimators": [50, 100, 200],
        "max_depth": [3, 5, 7, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "CatBoost": {
        "iterations": [50, 100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "depth": [4, 6, 8],
        "l2_leaf_reg": [1, 3, 5]
    },
    "TabPFN": {
    },
    "RBF_SVC": {
        "C": np.logspace(-1, 3, 5),
        "gamma": ['scale', 'auto'] + list(np.logspace(-3, 0, 4))
    },
    "Linear_SVC": {
        "C": np.logspace(-1, 3, 5),
        "max_iter": [5000]
    }
}