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
        "activation": ['relu', 'logistic'] ,
    },
    "XGB": {
        "n_estimators": [25, 50, 100],
        "learning_rate": [0.01, 0.1],
        "max_depth": [3, 5, 7]
    }
}