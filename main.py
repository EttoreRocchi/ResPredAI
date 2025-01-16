__author__ = "Ettore Rocchi"
__email__ = "ettore.rocchi3@unibo.it"

import os

import numpy as np
import pandas as pd
import argparse

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    f1_score,
    roc_auc_score
)

from utils import (
    ConfigHandler,
    DataSetter
)
from pipe import get_pipeline
from cm import save_cm

def perform_pipeline(
    datasetter: DataSetter,
    models: list[str],
    config_handler: ConfigHandler,
):

    X, Y = datasetter.X, datasetter.Y
    if config_handler.verbosity:
        config_handler.logger.info(
            f"Data dimension: {X.shape} "
        )

    ohe = ColumnTransformer(
        transformers = [
            (
                "ohe",
                OneHotEncoder(drop="if_binary", sparse_output=False),
                X.columns.difference(datasetter.continuous_features, sort=False)
            )
        ],
        remainder = "passthrough",
        verbose_feature_names_out = False
    ).set_output(transform="pandas")
    X = ohe.fit_transform(X)
    if config_handler.verbosity:
        config_handler.logger.info(
            f"Data dimension after encoding of categorical variables: {X.shape} "
        )

    for model in models:
        if config_handler.verbosity:
            config_handler.logger.info(
                f"Starting analyses with model: {model}. "
            )

        transformer, grid = get_pipeline(
            model_name = model,
            continuous_cols = datasetter.continuous_features,
            inner_folds = config_handler.inner_folds,
            n_jobs = config_handler.n_jobs,
            rnd_state = config_handler.seed
        )

        outer_cv = StratifiedKFold(
            n_splits=config_handler.outer_folds,
            shuffle=True,
            random_state=config_handler.seed
        )

        f1scores, mccs, cms, average_cms, aurocs = {}, {}, {}, {}, {}
        for target in Y.columns:
            f1scores[target] = []
            mccs[target] = []
            cms[target] = []
            aurocs[target] = []

            y = Y[target]
            if config_handler.verbosity:
                config_handler.logger.info(
                    f"Starting training for target: {target}. "
                )
            for i, (train_set, test_set) in enumerate(outer_cv.split(X, y)):
                if config_handler.verbosity == 2:
                    config_handler.logger.info(
                        f"Starting iteration: {i+1}. "
                    )
                X_train, X_test = X.iloc[train_set], X.iloc[test_set]
                y_train, y_test = y.iloc[train_set], y.iloc[test_set]

                X_train_scaled = transformer.fit_transform(X_train)
                X_test_scaled = transformer.fit_transform(X_test)

                grid.fit(
                    X = X_train_scaled,
                    y = y_train,
                )
                if config_handler.verbosity == 2:
                    config_handler.logger.info(
                        f"Model {model} trained for iteration: {i+1}. "
                    )

                best_classifier = grid.best_estimator_
                y_pred = best_classifier.predict(X_test_scaled)
                y_score = best_classifier.predict_proba(X_test_scaled)[:, 1]
                
                f1scores[target].append(
                    f1_score(
                        y_true = y_test,
                        y_pred = y_pred,
                        average = "weighted"
                    )
                )
                mccs[target].append(
                    matthews_corrcoef(
                        y_true = y_test,
                        y_pred = y_pred
                    )
                )
                aurocs[target].append(
                    roc_auc_score(
                        y_true = y_test,
                        y_score = y_score
                    )
                )
                cms[target].append(
                    confusion_matrix(
                        y_true = y_test,
                        y_pred = y_pred,
                        normalize="true",
                        labels=[0, 1]
                    )
                )
            if config_handler.verbosity:
                config_handler.logger.info(
                    f"Completed training for target {target} with model {model}. "
                )

        average_cms = {
            target:
            pd.DataFrame(
                data = np.nanmean(cms[target], axis = 0),
                index = ["Susceptible", "Resistant"],
                columns = ["Susceptible", "Resistant"]
            )
            for target in Y.columns
        }

        save_cm(
            f1scores = f1scores,
            mccs = mccs,
            cms = average_cms,
            aurocs = aurocs,
            out_dir=config_handler.out_folder,
            model = model.replace(" ", "_")
        )
        if config_handler.verbosity:
            config_handler.logger.info(
                f"Completed model {model}. "
            )
    if config_handler.verbosity:
        config_handler.logger.info(
            f"Analysis completed. "
        )

def main():
    """Perform the analysis."""
    parser = argparse.ArgumentParser(
        description="Implementation of the pipeline described in the work"
        " 'Artificial intelligence model to predict resistances in Gram-negative bloodstream infections'"
        " by Bonazzetti et al."
    )
    parser.add_argument(
        "--config", "-c",
        dest="config_path",
        required=True, 
        help="Path to the config file"
    )
    args = parser.parse_args()
    config_handler = ConfigHandler(args.config_path)
    os.makedirs(config_handler.out_folder, exist_ok=True)

    datasetter = DataSetter(config_handler)

    models = config_handler.models

    perform_pipeline(
        datasetter=datasetter,
        models=models,
        config_handler=config_handler
    )

if __name__ == "__main__":
    main()