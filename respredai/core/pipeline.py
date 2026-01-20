"""Main pipeline execution for ResPredAI."""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, make_scorer, roc_curve
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    TunedThresholdClassifierCV,
    cross_val_predict,
)
from sklearn.preprocessing import OneHotEncoder

from respredai.core.metrics import metric_dict, save_metrics_summary
from respredai.core.models import generate_summary_report, get_model_path, load_models, save_models
from respredai.core.pipe import get_pipeline
from respredai.io.config import ConfigHandler, DataSetter
from respredai.visualization.confusion_matrix import save_cm
from respredai.visualization.html_report import generate_html_report


def perform_pipeline(
    datasetter: DataSetter, models: list[str], config_handler: ConfigHandler, progress_callback=None
):
    """
    Execute the machine learning pipeline with nested cross-validation.

    Parameters
    ----------
    datasetter : DataSetter
        Object containing the dataset and feature information
    models : list[str]
        List of model names to train
    config_handler : ConfigHandler
        Configuration handler with pipeline parameters
    progress_callback : TrainingProgressCallback, optional
        Callback object for progress updates
    """

    X, Y = datasetter.X, datasetter.Y
    if config_handler.verbosity:
        config_handler.logger.info(f"Data dimension: {X.shape}")

    # List of categorical columns (non-continuous)
    categorical_cols = [col for col in X.columns if col not in datasetter.continuous_features]

    # One-hot encoding of categorical features
    ohe_transformer = ColumnTransformer(
        transformers=[
            (
                "ohe",
                OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"),
                categorical_cols,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    # Apply one-hot encoding
    X = ohe_transformer.fit_transform(X)

    # Clean feature names for XGBoost compatibility (remove <, > characters)
    X.columns = X.columns.str.replace("<", "_lt_", regex=False)
    X.columns = X.columns.str.replace(">", "_gt_", regex=False)

    if config_handler.verbosity:
        config_handler.logger.info(
            f"After preprocessing, data dimension: {X.shape}. "
            f"Training on {len(models)} models: {models}."
        )

    # Start overall progress
    if progress_callback:
        total_work = len(models) * len(Y.columns) * config_handler.outer_folds
        progress_callback.start(total_work=total_work)

    for model in models:
        if config_handler.verbosity:
            config_handler.logger.info(f"Starting model: {model}")

        # Start model progress
        if progress_callback:
            total_work_for_model = len(Y.columns) * config_handler.outer_folds
            progress_callback.start_model(model, total_work=total_work_for_model)

        # Initialize pipeline
        try:
            transformer, grid = get_pipeline(
                model_name=model,
                continuous_cols=datasetter.continuous_features,
                inner_folds=config_handler.inner_folds,
                n_jobs=config_handler.n_jobs,
                rnd_state=config_handler.seed,
                use_groups=(datasetter.groups is not None),
                imputation_method=config_handler.imputation_method,
                imputation_strategy=config_handler.imputation_strategy,
                imputation_n_neighbors=config_handler.imputation_n_neighbors,
                imputation_estimator=config_handler.imputation_estimator,
            )
        except Exception as e:
            if config_handler.verbosity:
                config_handler.logger.error(f"Failed to initialize model {model}: {str(e)}")
            warnings.warn(f"Skipping model {model} due to initialization error: {str(e)}")
            if progress_callback:
                total_work_skipped = len(Y.columns) * config_handler.outer_folds
                progress_callback.skip_model(model, total_work_skipped, "initialization error")
            continue

        # Use StratifiedGroupKFold if groups are specified, otherwise StratifiedKFold
        if datasetter.groups is not None:
            outer_cv = StratifiedGroupKFold(
                n_splits=config_handler.outer_folds, shuffle=True, random_state=config_handler.seed
            )
        else:
            outer_cv = StratifiedKFold(
                n_splits=config_handler.outer_folds, shuffle=True, random_state=config_handler.seed
            )

        f1scores, mccs, cms, aurocs = {}, {}, {}, {}
        all_metrics = {}  # Store comprehensive metrics
        # Sample-level predictions for bootstrap CI
        all_y_true = {}
        all_y_pred = {}
        all_y_prob = {}

        for target in Y.columns:
            # Check for existing saved models
            model_path = get_model_path(config_handler.out_folder, model, target)

            # Try to load saved models
            model_data = None
            start_fold = 0
            fold_models = []
            fold_transformers = []
            fold_thresholds = []
            fold_hyperparams = []
            fold_test_data = []

            if config_handler.save_models_enable and model_path.exists():
                model_data = load_models(model_path)
                if model_data is not None:
                    completed_folds = model_data.get("completed_folds", 0)

                    # Check if all folds are completed
                    if completed_folds >= config_handler.outer_folds:
                        if config_handler.verbosity:
                            config_handler.logger.info(
                                f"All folds completed for {model} - {target}. Loading from saved models."
                            )

                        # Restore metrics from saved models
                        all_metrics[target] = model_data["metrics"].get("all_metrics", [])
                        f1scores[target] = model_data["metrics"].get("f1scores", [])
                        mccs[target] = model_data["metrics"].get("mccs", [])
                        cms[target] = model_data["metrics"].get("cms", [])
                        aurocs[target] = model_data["metrics"].get("aurocs", [])

                        # Restore sample-level predictions for bootstrap CI
                        all_y_true[target] = model_data["metrics"].get("all_y_true", [])
                        all_y_pred[target] = model_data["metrics"].get("all_y_pred", [])
                        all_y_prob[target] = model_data["metrics"].get("all_y_prob", [])

                        if progress_callback:
                            progress_callback.skip_target(
                                target, config_handler.outer_folds, "saved models"
                            )

                        continue
                    else:
                        # Resume from last completed fold
                        start_fold = completed_folds
                        fold_models = model_data.get("fold_models", [])
                        fold_transformers = model_data.get("fold_transformers", [])
                        fold_thresholds = model_data.get("fold_thresholds", [])
                        fold_hyperparams = model_data.get("fold_hyperparams", [])
                        fold_test_data = model_data.get("fold_test_data", [])

                        # Restore partial metrics
                        all_metrics[target] = model_data["metrics"].get("all_metrics", [])
                        f1scores[target] = model_data["metrics"].get("f1scores", [])
                        mccs[target] = model_data["metrics"].get("mccs", [])
                        cms[target] = model_data["metrics"].get("cms", [])
                        aurocs[target] = model_data["metrics"].get("aurocs", [])

                        # Restore partial sample-level predictions for bootstrap CI
                        all_y_true[target] = model_data["metrics"].get("all_y_true", [])
                        all_y_pred[target] = model_data["metrics"].get("all_y_pred", [])
                        all_y_prob[target] = model_data["metrics"].get("all_y_prob", [])

                        if config_handler.verbosity:
                            config_handler.logger.info(
                                f"Resuming {model} - {target} from fold {start_fold + 1}"
                            )

            # Initialize metrics storage if starting fresh
            if start_fold == 0:
                f1scores[target] = []
                mccs[target] = []
                cms[target] = []
                aurocs[target] = []
                all_metrics[target] = []
                # Sample-level predictions for bootstrap CI
                all_y_true[target] = []
                all_y_pred[target] = []
                all_y_prob[target] = []

            y = Y[target]
            if config_handler.verbosity:
                config_handler.logger.info(
                    f"Starting training for target: {target} (from fold {start_fold + 1})."
                )

            # Start target progress
            if progress_callback:
                progress_callback.start_target(
                    target, total_folds=config_handler.outer_folds, resumed_from=start_fold
                )

            # Pass groups to split if available
            split_args = [X, y]
            if datasetter.groups is not None:
                split_args.append(datasetter.groups)
            for i, (train_set, test_set) in enumerate(outer_cv.split(*split_args)):
                # Skip already completed folds
                if i < start_fold:
                    continue

                # Start fold progress
                if progress_callback:
                    progress_callback.start_fold(i + 1, config_handler.outer_folds)

                if config_handler.verbosity == 2:
                    config_handler.logger.info(f"Starting iteration: {i + 1}.")

                X_train, X_test = X.iloc[train_set], X.iloc[test_set]
                y_train, y_test = y.iloc[train_set], y.iloc[test_set]

                # Apply scaling
                X_train_scaled = transformer.fit_transform(X_train)
                X_test_scaled = transformer.transform(X_test)

                try:
                    # Pass groups to GridSearchCV if available
                    fit_params = {}
                    if datasetter.groups is not None:
                        fit_params["groups"] = datasetter.groups[train_set]

                    # Step 1: Hyperparameter tuning with GridSearchCV (optimizes ROC-AUC)
                    grid.fit(X=X_train_scaled, y=y_train, **fit_params)

                    if config_handler.verbosity == 2:
                        config_handler.logger.info(f"Model {model} trained for iteration: {i + 1}.")

                    # Step 2: Get best estimator and hyperparameters from GridSearchCV
                    best_estimator = grid.best_estimator_
                    best_params = grid.best_params_

                    # Step 3: Threshold calibration using Youden's J statistic (if enabled)
                    if config_handler.calibrate_threshold:
                        # Determine threshold calibration method
                        threshold_method = config_handler.threshold_method
                        if threshold_method == "auto":
                            # Auto: use OOF for small datasets, CV for large datasets
                            threshold_method = "oof" if len(y_train) < 1000 else "cv"

                        if threshold_method == "oof":
                            # Method 1: Out-of-Fold (OOF) predictions approach
                            # Use the same CV splitter as GridSearchCV
                            if datasetter.groups is not None:
                                inner_cv = StratifiedGroupKFold(
                                    n_splits=config_handler.inner_folds,
                                    shuffle=True,
                                    random_state=config_handler.seed,
                                )
                                cv_fit_params = {"groups": datasetter.groups[train_set]}
                            else:
                                inner_cv = StratifiedKFold(
                                    n_splits=config_handler.inner_folds,
                                    shuffle=True,
                                    random_state=config_handler.seed,
                                )
                                cv_fit_params = {}

                            # Get OOF probability predictions on training data
                            y_pred_proba_oof = cross_val_predict(
                                best_estimator,
                                X_train_scaled,
                                y_train,
                                cv=inner_cv,
                                method="predict_proba",
                                **cv_fit_params,
                            )

                            # Get threshold scorer based on configured objective
                            from respredai.core.metrics import get_threshold_scorer

                            threshold_scorer = get_threshold_scorer(
                                config_handler.threshold_objective,
                                config_handler.vme_cost,
                                config_handler.me_cost,
                            )

                            # Calculate ROC curve to get candidate thresholds
                            _, _, thresholds = roc_curve(y_train, y_pred_proba_oof[:, 1])

                            # Find optimal threshold by evaluating scorer at each threshold
                            best_score = float("-inf")
                            best_threshold = 0.5
                            for thresh in thresholds:
                                y_pred_thresh = (y_pred_proba_oof[:, 1] >= thresh).astype(int)
                                score = threshold_scorer(y_train.values, y_pred_thresh)
                                if score > best_score:
                                    best_score = score
                                    best_threshold = thresh

                            # The best_estimator is already trained, use it as the final classifier
                            best_classifier = best_estimator

                        else:  # threshold_method == "cv"
                            # Method 2: TunedThresholdClassifierCV approach
                            # Create scorer based on threshold objective
                            from respredai.core.metrics import get_threshold_scorer

                            threshold_scorer_fn = get_threshold_scorer(
                                config_handler.threshold_objective,
                                config_handler.vme_cost,
                                config_handler.me_cost,
                            )
                            objective_scorer = make_scorer(threshold_scorer_fn)

                            # Set best hyperparameters on the unfitted estimator
                            grid.estimator.set_params(**best_params)

                            # Create CV splitter for threshold calibration
                            inner_tuner_cv = StratifiedKFold(
                                n_splits=config_handler.inner_folds,
                                shuffle=True,
                                random_state=config_handler.seed,
                            )

                            # Wrap the unfitted estimator in TunedThresholdClassifierCV
                            tuned_model = TunedThresholdClassifierCV(
                                estimator=grid.estimator,
                                cv=inner_tuner_cv,
                                scoring=objective_scorer,
                                n_jobs=1,
                            )

                            # Fit to calibrate threshold with CV
                            tuned_model.fit(X_train_scaled, y_train)
                            best_classifier = tuned_model
                            best_threshold = tuned_model.best_threshold_
                    else:
                        # No threshold calibration - use GridSearchCV's best estimator with default threshold
                        best_classifier = best_estimator
                        best_threshold = 0.5

                    # Step 4: Predict on test set using calibrated threshold
                    if config_handler.calibrate_threshold and threshold_method == "cv":
                        # For TunedThresholdClassifierCV, use predict() which applies threshold automatically
                        y_pred = best_classifier.predict(X_test_scaled)
                        y_prob = best_classifier.predict_proba(X_test_scaled)
                    elif config_handler.calibrate_threshold and threshold_method == "oof":
                        # For OOF method, manually apply threshold
                        y_prob = best_classifier.predict_proba(X_test_scaled)
                        y_pred = (y_prob[:, 1] >= best_threshold).astype(int)
                    else:
                        # For no calibration, use direct methods
                        y_prob = best_classifier.predict_proba(X_test_scaled)
                        y_pred = best_classifier.predict(X_test_scaled)

                    # Calculate comprehensive metrics
                    fold_metrics = metric_dict(y_true=y_test.values, y_pred=y_pred, y_prob=y_prob)
                    all_metrics[target].append(fold_metrics)

                    # Store sample-level predictions for bootstrap CI
                    all_y_true[target].extend(y_test.values)
                    all_y_pred[target].extend(y_pred)
                    all_y_prob[target].extend(y_prob)

                    # Store individual metrics for backwards compatibility
                    f1scores[target].append(fold_metrics["F1 (weighted)"])
                    mccs[target].append(fold_metrics["MCC"])
                    aurocs[target].append(fold_metrics["AUROC"])
                    cms[target].append(
                        confusion_matrix(
                            y_true=y_test, y_pred=y_pred, normalize="true", labels=[0, 1]
                        )
                    )

                    # Store the best model, transformer, threshold, and hyperparameters for this fold
                    fold_models.append(best_classifier)
                    fold_transformers.append(transformer)
                    fold_thresholds.append(best_threshold)
                    fold_hyperparams.append(best_params)
                    # Store test data for SHAP computation (use transformed feature names)
                    fold_test_data.append(
                        (X_test_scaled, list(transformer.get_feature_names_out()))
                    )

                    # Update progress for successful fold
                    if progress_callback:
                        progress_callback.complete_fold(i + 1, fold_metrics)

                except Exception as e:
                    if config_handler.verbosity:
                        config_handler.logger.error(
                            f"Error in iteration {i + 1} for target {target}: {str(e)}"
                        )
                    # Append NaN for failed iterations
                    nan_metrics = {
                        "Precision (0)": np.nan,
                        "Precision (1)": np.nan,
                        "Recall (0)": np.nan,
                        "Recall (1)": np.nan,
                        "F1 (0)": np.nan,
                        "F1 (1)": np.nan,
                        "F1 (weighted)": np.nan,
                        "MCC": np.nan,
                        "Balanced Acc": np.nan,
                        "AUROC": np.nan,
                        "VME": np.nan,
                        "ME": np.nan,
                    }
                    all_metrics[target].append(nan_metrics)
                    f1scores[target].append(np.nan)
                    mccs[target].append(np.nan)
                    aurocs[target].append(np.nan)
                    cms[target].append(np.full((2, 2), np.nan))
                    fold_models.append(None)
                    fold_transformers.append(None)
                    fold_thresholds.append(None)
                    fold_hyperparams.append(None)
                    fold_test_data.append(None)

                    if progress_callback:
                        progress_callback.complete_fold(i + 1, nan_metrics)

                # Save models after each fold if enabled
                if config_handler.save_models_enable:
                    target_metrics = {
                        "all_metrics": all_metrics[target],
                        "f1scores": f1scores[target],
                        "mccs": mccs[target],
                        "cms": cms[target],
                        "aurocs": aurocs[target],
                        # Sample-level predictions for bootstrap CI
                        "all_y_true": all_y_true[target],
                        "all_y_pred": all_y_pred[target],
                        "all_y_prob": all_y_prob[target],
                    }

                    save_models(
                        fold_models=fold_models,
                        fold_transformers=fold_transformers,
                        fold_thresholds=fold_thresholds,
                        fold_hyperparams=fold_hyperparams,
                        metrics=target_metrics,
                        completed_folds=i + 1,
                        model_path=model_path,
                        compression=config_handler.model_compression,
                        fold_test_data=fold_test_data,
                    )

                    if config_handler.verbosity == 2:
                        config_handler.logger.info(
                            f"Saved models after fold {i + 1} for {model} - {target}"
                        )

            if config_handler.verbosity:
                config_handler.logger.info(
                    f"Completed training for target {target} with model {model}."
                )

            # Calculate summary metrics for progress callback
            if progress_callback:
                summary_metrics = {
                    "F1 (weighted)": np.nanmean(f1scores[target]),
                    "F1_std": np.nanstd(f1scores[target]),
                    "MCC": np.nanmean(mccs[target]),
                    "MCC_std": np.nanstd(mccs[target]),
                    "AUROC": np.nanmean(aurocs[target]),
                    "AUROC_std": np.nanstd(aurocs[target]),
                }
                progress_callback.complete_target(target, summary_metrics)

        # Calculate average confusion matrices
        average_cms = {
            target: pd.DataFrame(
                data=np.nanmean(cms[target], axis=0),
                index=["Susceptible", "Resistant"],
                columns=["Susceptible", "Resistant"],
            )
            for target in Y.columns
        }

        # Save confusion matrix visualizations
        save_cm(
            f1scores=f1scores,
            mccs=mccs,
            cms=average_cms,
            aurocs=aurocs,
            out_dir=config_handler.out_folder,
            model=model.replace(" ", "_"),
        )

        # Save comprehensive metrics for each target
        model_safe_name = model.replace(" ", "_")
        for target in Y.columns:
            target_safe_name = target.replace(" ", "_")
            metrics_output_path = (
                Path(config_handler.out_folder)
                / "metrics"
                / target_safe_name
                / f"{model_safe_name}_metrics_detailed.csv"
            )

            save_metrics_summary(
                metrics_dict=all_metrics[target],
                output_path=metrics_output_path,
                confidence=0.95,
                n_bootstrap=1_000,
                random_state=config_handler.seed,
                y_true_all=np.array(all_y_true[target]),
                y_pred_all=np.array(all_y_pred[target]),
                y_prob_all=np.array(all_y_prob[target]),
            )

            if config_handler.verbosity:
                config_handler.logger.info(
                    f"Saved detailed metrics for {model} - {target} to {metrics_output_path}"
                )

        if config_handler.verbosity:
            config_handler.logger.info(f"Completed model {model}.")

        # Complete model progress
        if progress_callback:
            progress_callback.complete_model(model)

    # Stop progress tracking
    if progress_callback:
        progress_callback.stop()

    # Generate summary reports
    generate_summary_report(
        output_folder=config_handler.out_folder,
        models=config_handler.models,
        targets=list(Y.columns),
    )

    # Generate HTML report
    if config_handler.verbosity:
        config_handler.logger.info("Generating HTML report...")
    try:
        report_path = generate_html_report(
            output_folder=config_handler.out_folder,
            models=config_handler.models,
            targets=list(Y.columns),
            config_handler=config_handler,
        )
        if config_handler.verbosity:
            config_handler.logger.info(f"HTML report generated: {report_path}")
    except Exception as e:
        if config_handler.verbosity:
            config_handler.logger.warning(f"Failed to generate HTML report: {e}")

    # Generate reproducibility manifest
    from respredai.io.reproducibility import (
        create_reproducibility_manifest,
        save_reproducibility_manifest,
    )

    manifest = create_reproducibility_manifest(config_handler, datasetter)
    save_reproducibility_manifest(manifest, Path(config_handler.out_folder))
    if config_handler.verbosity:
        config_handler.logger.info("Reproducibility manifest saved.")

    if config_handler.verbosity:
        config_handler.logger.info("Analysis completed.")


def perform_training(
    datasetter: DataSetter,
    models: List[str],
    config_handler: ConfigHandler,
    progress_callback: Optional[Any] = None,
) -> None:
    """
    Train models on entire dataset using GridSearchCV for hyperparameter tuning.

    Trains each model-target combination on the full dataset and saves the best
    model to disk for later use with perform_evaluation().

    Parameters
    ----------
    datasetter : DataSetter
        Data container with features (X), targets (Y), and optional groups.
    models : List[str]
        Model names to train (e.g., ['LR', 'RF', 'XGB']).
    config_handler : ConfigHandler
        Configuration handler with pipeline settings.
    progress_callback : optional
        Callback for progress updates (SimpleTrainingProgressCallback).
    """
    X = datasetter.X
    Y = datasetter.Y

    # List of categorical columns (non-continuous) - same approach as perform_pipeline
    categorical_cols = [col for col in X.columns if col not in datasetter.continuous_features]

    # One-hot encoding of categorical features using ColumnTransformer
    ohe_transformer = ColumnTransformer(
        transformers=[
            (
                "ohe",
                OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"),
                categorical_cols,
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")

    # Apply one-hot encoding
    X = ohe_transformer.fit_transform(X)

    # Clean feature names for XGBoost compatibility (remove <, > characters)
    X.columns = X.columns.str.replace("<", "_lt_", regex=False)
    X.columns = X.columns.str.replace(">", "_gt_", regex=False)

    # Create output directories
    trained_models_dir = Path(config_handler.out_folder) / "trained_models"
    trained_models_dir.mkdir(parents=True, exist_ok=True)

    # Store metadata for evaluation
    metadata = {
        "features": list(datasetter.X.columns),
        "continuous_features": datasetter.continuous_features,
        "categorical_features": categorical_cols,
        "targets": list(Y.columns),
        "feature_names_transformed": list(X.columns),
        "feature_dtypes": {col: str(dtype) for col, dtype in datasetter.X.dtypes.items()},
        "training_data_path": str(config_handler.data_path),
        "training_timestamp": datetime.now().isoformat(),
        "config": {
            "inner_folds": config_handler.inner_folds,
            "calibrate_threshold": config_handler.calibrate_threshold,
            "threshold_method": config_handler.threshold_method
            if config_handler.calibrate_threshold
            else None,
            "seed": config_handler.seed,
        },
    }

    if progress_callback:
        progress_callback.start(
            total_models=len(models), total_targets=len(Y.columns), total_folds=1
        )

    for model in models:
        if progress_callback:
            progress_callback.start_model(model)

        for target in Y.columns:
            if progress_callback:
                progress_callback.start_target(target)

            y = Y[target]

            # Get pipeline components
            transformer, grid = get_pipeline(
                model_name=model,
                continuous_cols=datasetter.continuous_features,
                n_jobs=config_handler.n_jobs,
                rnd_state=config_handler.seed,
                inner_folds=config_handler.inner_folds,
                use_groups=datasetter.groups is not None,
                imputation_method=config_handler.imputation_method,
                imputation_strategy=config_handler.imputation_strategy,
                imputation_n_neighbors=config_handler.imputation_n_neighbors,
                imputation_estimator=config_handler.imputation_estimator,
            )

            # Scale features
            X_scaled = transformer.fit_transform(X)

            # Fit GridSearchCV on entire dataset
            fit_params = {}
            if datasetter.groups is not None:
                fit_params["groups"] = datasetter.groups

            grid.fit(X=X_scaled, y=y, **fit_params)

            best_estimator = grid.best_estimator_
            best_params = grid.best_params_

            # Threshold calibration using OOF predictions
            best_threshold = 0.5
            if config_handler.calibrate_threshold:
                threshold_method = config_handler.threshold_method
                if threshold_method == "auto":
                    threshold_method = "oof" if len(y) < 1000 else "cv"

                if datasetter.groups is not None:
                    inner_cv = StratifiedGroupKFold(
                        n_splits=config_handler.inner_folds,
                        shuffle=True,
                        random_state=config_handler.seed,
                    )
                    cv_fit_params = {"groups": datasetter.groups}
                else:
                    inner_cv = StratifiedKFold(
                        n_splits=config_handler.inner_folds,
                        shuffle=True,
                        random_state=config_handler.seed,
                    )
                    cv_fit_params = {}

                # Get OOF predictions
                y_pred_proba_oof = cross_val_predict(
                    best_estimator,
                    X_scaled,
                    y,
                    cv=inner_cv,
                    method="predict_proba",
                    **cv_fit_params,
                )[:, 1]

                # Find optimal threshold using configured objective
                from respredai.core.metrics import get_threshold_scorer

                threshold_scorer = get_threshold_scorer(
                    config_handler.threshold_objective,
                    config_handler.vme_cost,
                    config_handler.me_cost,
                )

                _, _, thresholds = roc_curve(y, y_pred_proba_oof)

                best_score = float("-inf")
                best_threshold = 0.5
                for thresh in thresholds:
                    y_pred_thresh = (y_pred_proba_oof >= thresh).astype(int)
                    score = threshold_scorer(y.values, y_pred_thresh)
                    if score > best_score:
                        best_score = score
                        best_threshold = thresh

            # Save model bundle
            model_bundle = {
                "model": best_estimator,
                "transformer": transformer,
                "threshold": best_threshold,
                "hyperparams": best_params,
                "feature_names": list(datasetter.X.columns),
                "feature_names_transformed": list(X.columns),
                "target_name": target,
                "model_name": model,
                "training_timestamp": datetime.now().isoformat(),
            }

            model_safe = model.replace(" ", "_")
            target_safe = target.replace(" ", "_")
            model_path = trained_models_dir / f"{model_safe}_{target_safe}.joblib"
            joblib.dump(model_bundle, model_path, compress=3)

            if config_handler.verbosity:
                config_handler.logger.info(f"Saved trained model: {model_path}")

            if progress_callback:
                progress_callback.complete_target(target, {"threshold": best_threshold})

        if progress_callback:
            progress_callback.complete_model(model)

    # Save metadata
    metadata_path = trained_models_dir / "training_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if progress_callback:
        progress_callback.stop()

    # Generate reproducibility manifest
    from respredai.io.reproducibility import (
        create_reproducibility_manifest,
        save_reproducibility_manifest,
    )

    manifest = create_reproducibility_manifest(config_handler, datasetter)
    save_reproducibility_manifest(manifest, Path(config_handler.out_folder))
    if config_handler.verbosity:
        config_handler.logger.info("Reproducibility manifest saved.")

    if config_handler.verbosity:
        config_handler.logger.info("Training completed.")


def perform_evaluation(
    models_dir: Path, data_path: Path, output_dir: Path, verbose: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Evaluate trained models on new data with ground truth.

    Applies models trained with perform_training() to new data and computes
    performance metrics against known labels.

    Parameters
    ----------
    models_dir : Path
        Directory containing trained model files and training_metadata.json.
    data_path : Path
        Path to new data CSV file (must include target columns for ground truth).
    output_dir : Path
        Directory to save evaluation results.
    verbose : bool
        Print progress messages (default: True).

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Evaluation results keyed by 'model_target' with metrics dictionary.
    """
    # Load training metadata
    metadata_path = models_dir / "training_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Training metadata not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load new data
    new_data = pd.read_csv(data_path)

    # Validate columns
    required_features = metadata["features"]
    required_targets = metadata["targets"]

    missing_features = set(required_features) - set(new_data.columns)
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    missing_targets = set(required_targets) - set(new_data.columns)
    if missing_targets:
        raise ValueError(f"Missing target columns (ground truth required): {missing_targets}")

    # Extract features and targets
    X_new = new_data[required_features].copy()
    Y_new = new_data[required_targets].copy()

    # One-hot encode categorical features (same as training)
    categorical_features = metadata["categorical_features"]
    if categorical_features:
        # Use drop_first=False to match OneHotEncoder(drop=None) used in training
        X_new = pd.get_dummies(X_new, columns=categorical_features, drop_first=False)

    # Clean feature names for XGBoost compatibility (same as training)
    X_new.columns = X_new.columns.str.replace("<", "_lt_", regex=False)
    X_new.columns = X_new.columns.str.replace(">", "_gt_", regex=False)

    # Align columns with training (add missing dummy columns as 0, remove extra)
    expected_features = metadata["feature_names_transformed"]
    for col in expected_features:
        if col not in X_new.columns:
            X_new[col] = 0
    X_new = X_new[expected_features]

    # Create output directories
    output_dir = Path(output_dir)
    metrics_dir = output_dir / "metrics"
    predictions_dir = output_dir / "predictions"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Find model files
    model_files = list(models_dir.glob("*.joblib"))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {models_dir}")

    results = {}
    all_summaries = []

    for model_file in model_files:
        bundle = joblib.load(model_file)

        model_name = bundle["model_name"]
        target_name = bundle["target_name"]
        model = bundle["model"]
        transformer = bundle["transformer"]
        threshold = bundle["threshold"]

        if target_name not in Y_new.columns:
            continue

        y_true = Y_new[target_name].values

        # Scale features
        X_scaled = transformer.transform(X_new)

        # Predict
        y_prob = model.predict_proba(X_scaled)
        y_pred = (y_prob[:, 1] >= threshold).astype(int)

        # Calculate metrics
        metrics = metric_dict(y_true=y_true, y_pred=y_pred, y_prob=y_prob)
        results[f"{model_name}_{target_name}"] = metrics

        # Save predictions with uncertainty
        model_safe = model_name.replace(" ", "_")
        target_safe = target_name.replace(" ", "_")

        # Calculate uncertainty scores
        from respredai.core.metrics import calculate_uncertainty

        uncertainty_scores, is_uncertain = calculate_uncertainty(
            y_prob[:, 1], threshold, margin=0.1
        )

        pred_df = pd.DataFrame(
            {
                "sample_id": range(len(y_true)),
                "y_true": y_true,
                "y_pred": y_pred,
                "y_prob": y_prob[:, 1],
                "uncertainty": uncertainty_scores,
                "is_uncertain": is_uncertain,
            }
        )
        pred_path = predictions_dir / f"{model_safe}_{target_safe}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)

        # Save metrics
        target_metrics_dir = metrics_dir / target_safe
        target_metrics_dir.mkdir(parents=True, exist_ok=True)

        metrics_df = pd.DataFrame([{"Metric": k, "Value": v} for k, v in metrics.items()])
        metrics_path = target_metrics_dir / f"{model_safe}_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)

        # Collect for summary
        row = {"Model": model_name, "Target": target_name}
        row.update(metrics)
        all_summaries.append(row)

        if verbose:
            print(
                f"Evaluated {model_name} on {target_name}: F1={metrics['F1 (weighted)']:.3f}, MCC={metrics['MCC']:.3f}"
            )

    # Save evaluation summary
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        summary_path = output_dir / "evaluation_summary.csv"
        summary_df.to_csv(summary_path, index=False)

    return results
