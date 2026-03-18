"""Main ML workflow execution for ResPredAI."""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, make_scorer, roc_curve
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    TunedThresholdClassifierCV,
    cross_val_predict,
)
from sklearn.preprocessing import OneHotEncoder

from respredai.core.cv_utils import get_outer_cv, get_temporal_split
from respredai.core.metrics import metric_dict, save_metrics_summary
from respredai.core.model_builder import get_pipeline
from respredai.core.models import generate_summary_report, get_model_path, load_models, save_models
from respredai.io.config import ConfigHandler, DataSetter
from respredai.visualization.confusion_matrix import save_cm
from respredai.visualization.html_report import generate_html_report


def _deduplicate_repeated_cv_predictions(
    indices: list,
    y_true: list,
    y_pred: list,
    y_prob: list,
    threshold: float = 0.5,
) -> tuple:
    """Deduplicate sample-level predictions from repeated CV by averaging per sample.

    Parameters
    ----------
    indices : list
        Sample indices from all folds (may contain duplicates across repeats).
    y_true : list
        True labels corresponding to each entry in ``indices``.
    y_pred : list
        Predicted labels corresponding to each entry in ``indices``.
    y_prob : list
        Predicted probabilities corresponding to each entry in ``indices``.
    threshold : float
        Decision threshold to apply to averaged probabilities.
        Should match the threshold used during fold-level evaluation.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray, np.ndarray)
        (dedup_true, dedup_pred, dedup_prob) arrays with one entry per unique sample.
    """
    indices = np.array(indices)
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # O(n log n) via sort-based grouping instead of O(n*k) equality scans
    order = np.argsort(indices)
    sorted_idx = indices[order]
    sorted_true = y_true[order]
    sorted_prob = y_prob[order]

    unique_idx, start_pos = np.unique(sorted_idx, return_index=True)
    groups = np.split(np.arange(len(sorted_idx)), start_pos[1:])

    n_unique = len(unique_idx)
    dedup_true = np.empty(n_unique, dtype=y_true.dtype)
    dedup_prob = np.empty((n_unique, y_prob.shape[1]), dtype=y_prob.dtype)

    for j, g in enumerate(groups):
        dedup_true[j] = sorted_true[g[0]]
        dedup_prob[j] = sorted_prob[g].mean(axis=0)

    dedup_pred = (dedup_prob[:, 1] >= threshold).astype(int)
    return dedup_true, dedup_pred, dedup_prob


def _build_ohe_transformer(categorical_cols, ohe_min_frequency=None):
    """Build an unfitted OHE ColumnTransformer.

    Parameters
    ----------
    categorical_cols : list[str]
        Names of categorical columns to one-hot encode.
    ohe_min_frequency : float or int, optional
        Minimum frequency for a category to get its own column.
        Float in (0, 1) is treated as a proportion; int >= 1 as an absolute count.

    Returns
    -------
    ColumnTransformer
        Unfitted transformer ready to be cloned per CV fold.
    """
    ohe_kwargs = {
        "drop": "if_binary",
        "sparse_output": False,
        "handle_unknown": "infrequent_if_exist" if ohe_min_frequency is not None else "ignore",
    }
    if ohe_min_frequency is not None:
        ohe_kwargs["min_frequency"] = ohe_min_frequency
    return ColumnTransformer(
        transformers=[("ohe", OneHotEncoder(**ohe_kwargs), categorical_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def _apply_ohe_and_clean(ohe_transformer, X_train, X_test=None):
    """Fit OHE on train, transform both, clean feature names, align columns.

    Parameters
    ----------
    ohe_transformer : ColumnTransformer
        Unfitted OHE transformer (will be fit on *X_train*).
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame, optional
        Test features. If provided, transformed and column-aligned to *X_train*.

    Returns
    -------
    pd.DataFrame or tuple of (pd.DataFrame, pd.DataFrame)
        Transformed training data, or (train, test) if *X_test* is given.
    """
    X_train_ohe = ohe_transformer.fit_transform(X_train)
    X_train_ohe.columns = X_train_ohe.columns.str.replace("<", "_lt_", regex=False).str.replace(
        ">", "_gt_", regex=False
    )

    if X_test is not None:
        X_test_ohe = ohe_transformer.transform(X_test)
        X_test_ohe.columns = X_test_ohe.columns.str.replace("<", "_lt_", regex=False).str.replace(
            ">", "_gt_", regex=False
        )
        # Align test columns to match train (add missing as 0, reorder)
        for col in X_train_ohe.columns:
            if col not in X_test_ohe.columns:
                X_test_ohe[col] = 0
        X_test_ohe = X_test_ohe[X_train_ohe.columns]
        return X_train_ohe, X_test_ohe

    return X_train_ohe


def _generate_final_reports(
    config_handler: ConfigHandler,
    Y: pd.DataFrame,
    datasetter: DataSetter,
) -> None:
    """Generate summary reports, HTML report, and reproducibility manifest."""
    generate_summary_report(
        output_folder=config_handler.out_folder,
        models=config_handler.models,
        targets=list(Y.columns),
    )

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

    from respredai.io.reproducibility import (
        create_reproducibility_manifest,
        save_reproducibility_manifest,
    )

    manifest = create_reproducibility_manifest(config_handler, datasetter)
    save_reproducibility_manifest(manifest, Path(config_handler.out_folder))
    if config_handler.verbosity:
        config_handler.logger.info("Reproducibility manifest saved.")


def _aggregate_confusion_matrices(
    cms: dict,
    Y: pd.DataFrame,
    config_handler: ConfigHandler,
) -> dict:
    """Compute average confusion matrices across folds/repeats."""
    if config_handler.outer_cv_repeats > 1:
        n_folds = config_handler.outer_folds
        n_repeats = config_handler.outer_cv_repeats
        average_cms = {}
        for target in Y.columns:
            repeat_cms = [
                np.nanmean(cms[target][r * n_folds : (r + 1) * n_folds], axis=0)
                for r in range(n_repeats)
            ]
            average_cms[target] = pd.DataFrame(
                data=np.nanmean(repeat_cms, axis=0),
                index=["Susceptible", "Resistant"],
                columns=["Susceptible", "Resistant"],
            )
    else:
        average_cms = {
            target: pd.DataFrame(
                data=np.nanmean(cms[target], axis=0),
                index=["Susceptible", "Resistant"],
                columns=["Susceptible", "Resistant"],
            )
            for target in Y.columns
        }
    return average_cms


def _compute_ci_metrics_for_target(
    target: str,
    config_handler: ConfigHandler,
    all_test_indices: dict,
    all_y_true: dict,
    all_y_pred: dict,
    all_y_prob: dict,
    all_metrics: dict,
    metrics_output_path: Path,
) -> None:
    """Deduplicate samples for repeated CV and save metrics with bootstrap CI."""
    if config_handler.outer_cv_repeats > 1 and all_test_indices[target]:
        y_true_ci, _, y_prob_ci = _deduplicate_repeated_cv_predictions(
            all_test_indices[target],
            all_y_true[target],
            all_y_pred[target],
            all_y_prob[target],
        )

        if config_handler.calibrate_threshold:
            from respredai.core.metrics import get_threshold_scorer

            threshold_scorer = get_threshold_scorer(
                config_handler.threshold_objective,
                config_handler.vme_cost,
                config_handler.me_cost,
            )
            _, _, thresholds = roc_curve(y_true_ci, y_prob_ci[:, 1])

            best_score = float("-inf")
            dedup_threshold = 0.5
            for thresh in thresholds:
                y_pred_thresh = (y_prob_ci[:, 1] >= thresh).astype(int)
                score = threshold_scorer(y_true_ci, y_pred_thresh)
                if score > best_score:
                    best_score = score
                    dedup_threshold = thresh
        else:
            dedup_threshold = 0.5

        y_pred_ci = (y_prob_ci[:, 1] >= dedup_threshold).astype(int)
    else:
        y_true_ci = np.array(all_y_true[target])
        y_pred_ci = np.array(all_y_pred[target])
        y_prob_ci = np.array(all_y_prob[target])

    save_metrics_summary(
        metrics_dict=all_metrics[target],
        output_path=metrics_output_path,
        confidence=0.95,
        n_bootstrap=1_000,
        random_state=config_handler.seed,
        y_true_all=y_true_ci,
        y_pred_all=y_pred_ci,
        y_prob_all=y_prob_ci,
        n_folds=config_handler.outer_folds,
        n_repeats=config_handler.outer_cv_repeats,
    )


def _apply_probability_calibration(
    best_estimator,
    config_handler: ConfigHandler,
    datasetter: DataSetter,
    X_train_scaled,
    y_train,
    train_set,
):
    """Wrap estimator with CalibratedClassifierCV and return calibrated estimator + splits."""
    if datasetter.groups is not None:
        prob_calib_cv = StratifiedGroupKFold(
            n_splits=config_handler.probability_calibration_cv,
            shuffle=True,
            random_state=config_handler.seed,
        )
        prob_calib_splits = list(
            prob_calib_cv.split(X_train_scaled, y_train, datasetter.groups[train_set])
        )
    else:
        prob_calib_splits = config_handler.probability_calibration_cv

    calibrated_classifier = CalibratedClassifierCV(
        estimator=best_estimator,
        method=config_handler.probability_calibration_method,
        cv=prob_calib_splits,
        n_jobs=1,
    )
    calibrated_classifier.fit(X_train_scaled, y_train)
    return calibrated_classifier, prob_calib_splits


def _optimize_threshold(
    best_estimator,
    config_handler: ConfigHandler,
    datasetter: DataSetter,
    grid,
    X_train_scaled,
    y_train,
    train_set,
    best_params: dict,
    prob_calib_splits=None,
):
    """Find the optimal decision threshold. Returns (best_classifier, best_threshold, method)."""
    threshold_method = config_handler.threshold_method
    if threshold_method == "auto":
        threshold_method = "oof" if len(y_train) < 1000 else "cv"

    from respredai.core.metrics import get_threshold_scorer

    if threshold_method == "oof":
        # OOF predictions approach
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

        oof_estimator = clone(best_estimator)
        if (
            config_handler.calibrate_probabilities
            and hasattr(oof_estimator, "cv")
            and isinstance(getattr(oof_estimator, "cv", None), list)
        ):
            oof_estimator.cv = config_handler.probability_calibration_cv

        y_pred_proba_oof = cross_val_predict(
            oof_estimator,
            X_train_scaled,
            y_train,
            cv=inner_cv,
            method="predict_proba",
            **cv_fit_params,
        )

        threshold_scorer = get_threshold_scorer(
            config_handler.threshold_objective,
            config_handler.vme_cost,
            config_handler.me_cost,
        )

        _, _, thresholds = roc_curve(y_train, y_pred_proba_oof[:, 1])

        best_score = float("-inf")
        best_threshold = 0.5
        for thresh in thresholds:
            y_pred_thresh = (y_pred_proba_oof[:, 1] >= thresh).astype(int)
            score = threshold_scorer(y_train.values, y_pred_thresh)
            if score > best_score:
                best_score = score
                best_threshold = thresh

        return best_estimator, best_threshold, threshold_method

    else:  # cv
        threshold_scorer_fn = get_threshold_scorer(
            config_handler.threshold_objective,
            config_handler.vme_cost,
            config_handler.me_cost,
        )
        objective_scorer = make_scorer(threshold_scorer_fn)

        if config_handler.calibrate_probabilities:
            base_est = clone(grid.estimator)
            base_est.set_params(**best_params)

            if datasetter.groups is not None:
                calib_cv = StratifiedGroupKFold(
                    n_splits=config_handler.probability_calibration_cv,
                    shuffle=True,
                    random_state=config_handler.seed,
                )
            else:
                calib_cv = prob_calib_splits  # int - safe for any subset
            estimator_for_threshold = CalibratedClassifierCV(
                estimator=base_est,
                method=config_handler.probability_calibration_method,
                cv=calib_cv,
                n_jobs=1,
            )
        else:
            estimator_for_threshold = clone(grid.estimator)
            estimator_for_threshold.set_params(**best_params)

        if datasetter.groups is not None:
            inner_tuner_cv = StratifiedGroupKFold(
                n_splits=config_handler.inner_folds,
                shuffle=True,
                random_state=config_handler.seed,
            )
        else:
            inner_tuner_cv = StratifiedKFold(
                n_splits=config_handler.inner_folds,
                shuffle=True,
                random_state=config_handler.seed,
            )

        tuned_model = TunedThresholdClassifierCV(
            estimator=estimator_for_threshold,
            cv=inner_tuner_cv,
            scoring=objective_scorer,
            n_jobs=1,
        )

        fit_kwargs = {}
        if datasetter.groups is not None:
            fit_kwargs["groups"] = datasetter.groups[train_set]
        tuned_model.fit(X_train_scaled, y_train, **fit_kwargs)
        return tuned_model, tuned_model.best_threshold_, threshold_method


def _make_nan_metrics() -> dict:
    """Return a dictionary of NaN values for all metrics (used on fold failure)."""
    return {
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
        "Brier Score": np.nan,
        "ECE": np.nan,
        "MCE": np.nan,
    }


def perform_pipeline(
    datasetter: DataSetter, models: list[str], config_handler: ConfigHandler, progress_callback=None
):
    """Execute the machine learning pipeline with nested cross-validation.

    Parameters
    ----------
    datasetter : DataSetter
        Object containing the dataset and feature information.
    models : list[str]
        List of model names to train.
    config_handler : ConfigHandler
        Configuration handler with pipeline parameters.
    progress_callback : TrainingProgressCallback, optional
        Callback object for progress updates.
    """
    X, Y = datasetter.X, datasetter.Y
    if config_handler.verbosity:
        config_handler.logger.info(f"Data dimension: {X.shape}")

    # List of categorical columns (non-continuous)
    categorical_cols = [col for col in X.columns if col not in datasetter.continuous_features]

    # Build unfitted OHE template (will be cloned and fit inside each CV fold)
    ohe_template = _build_ohe_transformer(categorical_cols, config_handler.ohe_min_frequency)

    if config_handler.verbosity:
        config_handler.logger.info(
            f"Data dimension (pre-OHE): {X.shape}. Training on {len(models)} models: {models}."
        )

    # Calculate total iterations (folds * repeats)
    total_outer_iterations = config_handler.outer_folds * config_handler.outer_cv_repeats

    if progress_callback:
        total_work = len(models) * len(Y.columns) * total_outer_iterations
        progress_callback.start(total_work=total_work)

    for model in models:
        if config_handler.verbosity:
            config_handler.logger.info(f"Starting model: {model}")

        if progress_callback:
            total_work_for_model = len(Y.columns) * total_outer_iterations
            progress_callback.start_model(model, total_work=total_work_for_model)

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
                calibrate_probabilities=config_handler.calibrate_probabilities,
            )
        except Exception as e:
            if config_handler.verbosity:
                config_handler.logger.error(f"Failed to initialize model {model}: {str(e)}")
            warnings.warn(f"Skipping model {model} due to initialization error: {str(e)}")
            if progress_callback:
                total_work_skipped = len(Y.columns) * total_outer_iterations
                progress_callback.skip_model(model, total_work_skipped, "initialization error")
            continue

        # Get outer CV splitter (supports repeated CV and groups)
        outer_cv = get_outer_cv(
            n_splits=config_handler.outer_folds,
            n_repeats=config_handler.outer_cv_repeats,
            use_groups=(datasetter.groups is not None),
            random_state=config_handler.seed,
        )

        f1scores, mccs, cms, aurocs = {}, {}, {}, {}
        all_metrics = {}  # Store comprehensive metrics
        # Sample-level predictions for bootstrap CI
        all_y_true = {}
        all_y_pred = {}
        all_y_prob = {}
        all_test_indices = {}
        # Per-fold data for reliability curves (lists of arrays, one per fold)
        fold_y_true_calib = {}
        fold_y_prob_calib = {}

        for target in Y.columns:
            # Check for existing saved models
            model_path = get_model_path(config_handler.out_folder, model, target)

            model_data = None
            start_fold = 0
            fold_models = []
            fold_transformers = []
            fold_ohe_transformers = []
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
                        all_test_indices[target] = model_data["metrics"].get("all_test_indices", [])

                        # Initialize empty calibration data (per-fold data not saved)
                        fold_y_true_calib[target] = []
                        fold_y_prob_calib[target] = []

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
                        fold_ohe_transformers = model_data.get("fold_ohe_transformers", [])
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
                        all_test_indices[target] = model_data["metrics"].get("all_test_indices", [])

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
                all_test_indices[target] = []
                # Per-fold data for reliability curves
                fold_y_true_calib[target] = []
                fold_y_prob_calib[target] = []

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

                X_train_raw, X_test_raw = X.iloc[train_set], X.iloc[test_set]
                y_train, y_test = y.iloc[train_set], y.iloc[test_set]

                # OHE: fit on training data only to prevent data leakage
                fold_ohe = clone(ohe_template)
                X_train_ohe, X_test_ohe = _apply_ohe_and_clean(fold_ohe, X_train_raw, X_test_raw)

                # Apply scaling (clone transformer to avoid state leakage)
                fold_transformer = clone(transformer)
                X_train_scaled = fold_transformer.fit_transform(X_train_ohe)
                X_test_scaled = fold_transformer.transform(X_test_ohe)

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

                    # Step 2.5: Post-hoc probability calibration (if enabled)
                    prob_calib_splits = None
                    if config_handler.calibrate_probabilities:
                        best_estimator, prob_calib_splits = _apply_probability_calibration(
                            best_estimator,
                            config_handler,
                            datasetter,
                            X_train_scaled,
                            y_train,
                            train_set,
                        )
                        if config_handler.verbosity == 2:
                            config_handler.logger.info(
                                f"Probability calibration applied "
                                f"(method={config_handler.probability_calibration_method})."
                            )

                    # Step 3: Threshold optimization (if enabled)
                    if config_handler.calibrate_threshold:
                        best_classifier, best_threshold, threshold_method = _optimize_threshold(
                            best_estimator,
                            config_handler,
                            datasetter,
                            grid,
                            X_train_scaled,
                            y_train,
                            train_set,
                            best_params,
                            prob_calib_splits,
                        )
                    else:
                        best_classifier = best_estimator
                        best_threshold = 0.5

                    # Step 4: Predict on test set using calibrated threshold
                    y_prob = best_classifier.predict_proba(X_test_scaled)
                    if config_handler.calibrate_threshold and threshold_method == "cv":
                        # TunedThresholdClassifierCV applies threshold internally
                        y_pred = best_classifier.predict(X_test_scaled)
                    elif config_handler.calibrate_threshold and threshold_method == "oof":
                        # OOF method: manually apply threshold
                        y_pred = (y_prob[:, 1] >= best_threshold).astype(int)
                    else:
                        y_pred = best_classifier.predict(X_test_scaled)

                    # Calculate comprehensive metrics
                    fold_metrics = metric_dict(y_true=y_test.values, y_pred=y_pred, y_prob=y_prob)
                    all_metrics[target].append(fold_metrics)

                    # Store sample-level predictions for bootstrap CI
                    all_y_true[target].extend(y_test.values)
                    all_y_pred[target].extend(y_pred)
                    all_y_prob[target].extend(y_prob)
                    all_test_indices[target].extend(test_set)

                    # Store per-fold data for reliability curves (as separate arrays)
                    fold_y_true_calib[target].append(y_test.values)
                    fold_y_prob_calib[target].append(y_prob[:, 1])

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
                    fold_transformers.append(fold_transformer)
                    fold_ohe_transformers.append(fold_ohe)
                    fold_thresholds.append(best_threshold)
                    fold_hyperparams.append(best_params)
                    # Store test data for SHAP computation (use transformed feature names)
                    fold_test_data.append(
                        (X_test_scaled, list(fold_transformer.get_feature_names_out()))
                    )

                    # Update progress for successful fold
                    if progress_callback:
                        progress_callback.complete_fold(i + 1, fold_metrics)

                except Exception as e:
                    if config_handler.verbosity:
                        config_handler.logger.error(
                            f"Error in iteration {i + 1} for target {target}: {str(e)}"
                        )
                    nan_metrics = _make_nan_metrics()
                    all_metrics[target].append(nan_metrics)
                    f1scores[target].append(np.nan)
                    mccs[target].append(np.nan)
                    aurocs[target].append(np.nan)
                    cms[target].append(np.full((2, 2), np.nan))
                    fold_models.append(None)
                    fold_transformers.append(None)
                    fold_ohe_transformers.append(None)
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
                        "all_test_indices": all_test_indices[target],
                    }

                    save_models(
                        fold_models=fold_models,
                        fold_transformers=fold_transformers,
                        fold_ohe_transformers=fold_ohe_transformers,
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
                if config_handler.outer_cv_repeats > 1:
                    n_folds = config_handler.outer_folds
                    n_repeats = config_handler.outer_cv_repeats
                    repeat_f1 = [
                        np.nanmean(f1scores[target][r * n_folds : (r + 1) * n_folds])
                        for r in range(n_repeats)
                    ]
                    repeat_mcc = [
                        np.nanmean(mccs[target][r * n_folds : (r + 1) * n_folds])
                        for r in range(n_repeats)
                    ]
                    repeat_auroc = [
                        np.nanmean(aurocs[target][r * n_folds : (r + 1) * n_folds])
                        for r in range(n_repeats)
                    ]
                    summary_metrics = {
                        "F1 (weighted)": np.nanmean(repeat_f1),
                        "F1_std": np.nanstd(repeat_f1, ddof=1),
                        "MCC": np.nanmean(repeat_mcc),
                        "MCC_std": np.nanstd(repeat_mcc, ddof=1),
                        "AUROC": np.nanmean(repeat_auroc),
                        "AUROC_std": np.nanstd(repeat_auroc, ddof=1),
                    }
                else:
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
        average_cms = _aggregate_confusion_matrices(cms, Y, config_handler)

        import re

        model_safe_name = re.sub(r"[^\w.-]", "_", model)

        # Save confusion matrix visualizations
        save_cm(
            f1scores=f1scores,
            mccs=mccs,
            cms=average_cms,
            aurocs=aurocs,
            out_dir=config_handler.out_folder,
            model=model_safe_name,
        )

        # Save comprehensive metrics for each target
        for target in Y.columns:
            target_safe_name = re.sub(r"[^\w.-]", "_", target)
            metrics_output_path = (
                Path(config_handler.out_folder)
                / "metrics"
                / target_safe_name
                / f"{model_safe_name}_metrics_detailed.csv"
            )

            _compute_ci_metrics_for_target(
                target,
                config_handler,
                all_test_indices,
                all_y_true,
                all_y_pred,
                all_y_prob,
                all_metrics,
                metrics_output_path,
            )

            # Generate reliability curves for this model-target combination
            # Skip if no per-fold data available (e.g., loaded from saved models)
            if fold_y_true_calib[target] and fold_y_prob_calib[target]:
                from respredai.visualization.reliability_curves import save_reliability_curves

                calibration_dir = Path(config_handler.out_folder) / "calibration"
                n_total_folds = len(fold_y_true_calib[target])
                save_reliability_curves(
                    y_true_list=fold_y_true_calib[target],
                    y_prob_list=fold_y_prob_calib[target],
                    fold_labels=(
                        [
                            f"R{r + 1}-F{f + 1}"
                            for r in range(config_handler.outer_cv_repeats)
                            for f in range(config_handler.outer_folds)
                        ]
                        if config_handler.outer_cv_repeats > 1
                        else [f"Fold {i + 1}" for i in range(n_total_folds)]
                    ),
                    out_dir=calibration_dir,
                    model=model_safe_name,
                    target=target_safe_name,
                )
                if config_handler.verbosity:
                    config_handler.logger.info(
                        f"Generated reliability curves for {model} - {target}"
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

    _generate_final_reports(config_handler, Y, datasetter)
    if config_handler.verbosity:
        config_handler.logger.info("Analysis completed.")


def perform_temporal_validation(
    datasetter: DataSetter,
    models: list,
    config_handler: ConfigHandler,
    progress_callback=None,
) -> None:
    """
    Evaluate models using a temporal (prospective-style) train/test split.

    Splits data chronologically using the configured temporal column and cutoff,
    trains each model on the historical portion, and evaluates on the prospective
    portion. This simulates real-world deployment where models are trained on
    past data and tested on future data.

    Parameters
    ----------
    datasetter : DataSetter
        Data container with features (X), targets (Y), optional groups,
        and temporal_column_values.
    models : list
        List of model names to train.
    config_handler : ConfigHandler
        Configuration handler with pipeline and temporal split parameters.
    progress_callback : optional
        Callback object for progress updates.
    """
    X, Y = datasetter.X, datasetter.Y

    # Get temporal split indices
    train_idx, test_idx = get_temporal_split(
        temporal_values=datasetter.temporal_column_values,
        split_date=config_handler.temporal_split_date,
        split_ratio=config_handler.temporal_split_ratio,
        groups=datasetter.groups,
    )

    if config_handler.verbosity:
        config_handler.logger.info(
            f"Temporal split: {len(train_idx)} train / {len(test_idx)} test samples"
        )
        # Log class distribution per target
        for target in Y.columns:
            y = Y[target]
            train_pos = y.iloc[train_idx].sum()
            test_pos = y.iloc[test_idx].sum()
            config_handler.logger.info(
                f"  {target}: train positive={train_pos}/{len(train_idx)}, "
                f"test positive={test_pos}/{len(test_idx)}"
            )

    # Build categorical column list and OHE transformer
    categorical_cols = [col for col in X.columns if col not in datasetter.continuous_features]
    ohe = _build_ohe_transformer(categorical_cols, config_handler.ohe_min_frequency)

    # Split data
    X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]

    # OHE: fit on training data only
    X_train_ohe, X_test_ohe = _apply_ohe_and_clean(ohe, X_train_raw, X_test_raw)

    if config_handler.verbosity:
        config_handler.logger.info(
            f"Temporal validation: data dimension (post-OHE): {X_train_ohe.shape}. "
            f"Training on {len(models)} models: {models}."
        )

    for model_name in models:
        if config_handler.verbosity:
            config_handler.logger.info(f"[Temporal] Starting model: {model_name}")

        try:
            transformer, grid = get_pipeline(
                model_name=model_name,
                continuous_cols=datasetter.continuous_features,
                inner_folds=config_handler.inner_folds,
                n_jobs=config_handler.n_jobs,
                rnd_state=config_handler.seed,
                use_groups=(datasetter.groups is not None),
                imputation_method=config_handler.imputation_method,
                imputation_strategy=config_handler.imputation_strategy,
                imputation_n_neighbors=config_handler.imputation_n_neighbors,
                imputation_estimator=config_handler.imputation_estimator,
                calibrate_probabilities=config_handler.calibrate_probabilities,
            )
        except Exception as e:
            if config_handler.verbosity:
                config_handler.logger.error(
                    f"[Temporal] Failed to initialize model {model_name}: {e}"
                )
            warnings.warn(
                f"[Temporal] Skipping model {model_name} due to initialization error: {e}"
            )
            continue

        # Scale features
        X_train_scaled = transformer.fit_transform(X_train_ohe)
        X_test_scaled = transformer.transform(X_test_ohe)

        temporal_cms = {}
        temporal_f1s: dict[str, list] = {}
        temporal_mccs: dict[str, list] = {}
        temporal_aurocs: dict[str, list] = {}

        for target in Y.columns:
            y_train = Y[target].iloc[train_idx]
            y_test = Y[target].iloc[test_idx]

            try:
                # Hyperparameter tuning with GridSearchCV
                fit_params = {}
                if datasetter.groups is not None:
                    fit_params["groups"] = datasetter.groups[train_idx]

                grid.fit(X=X_train_scaled, y=y_train, **fit_params)

                best_estimator = grid.best_estimator_
                best_params = grid.best_params_

                # Post-hoc probability calibration (if enabled)
                if config_handler.calibrate_probabilities:
                    if datasetter.groups is not None:
                        prob_calib_cv = StratifiedGroupKFold(
                            n_splits=config_handler.probability_calibration_cv,
                            shuffle=True,
                            random_state=config_handler.seed,
                        )
                        prob_calib_splits = list(
                            prob_calib_cv.split(
                                X_train_scaled, y_train, datasetter.groups[train_idx]
                            )
                        )
                    else:
                        prob_calib_splits = config_handler.probability_calibration_cv

                    calibrated_classifier = CalibratedClassifierCV(
                        estimator=best_estimator,
                        method=config_handler.probability_calibration_method,
                        cv=prob_calib_splits,
                        n_jobs=1,
                    )
                    calibrated_classifier.fit(X_train_scaled, y_train)
                    best_estimator = calibrated_classifier

                # Threshold optimization (if enabled)
                if config_handler.calibrate_threshold:
                    threshold_method = config_handler.threshold_method
                    if threshold_method == "auto":
                        threshold_method = "oof" if len(y_train) < 1000 else "cv"

                    if threshold_method == "oof":
                        if datasetter.groups is not None:
                            inner_cv = StratifiedGroupKFold(
                                n_splits=config_handler.inner_folds,
                                shuffle=True,
                                random_state=config_handler.seed,
                            )
                            cv_fit_params = {"groups": datasetter.groups[train_idx]}
                        else:
                            inner_cv = StratifiedKFold(
                                n_splits=config_handler.inner_folds,
                                shuffle=True,
                                random_state=config_handler.seed,
                            )
                            cv_fit_params = {}

                        oof_estimator = clone(best_estimator)
                        if (
                            config_handler.calibrate_probabilities
                            and hasattr(oof_estimator, "cv")
                            and isinstance(getattr(oof_estimator, "cv", None), list)
                        ):
                            oof_estimator.cv = config_handler.probability_calibration_cv

                        y_pred_proba_oof = cross_val_predict(
                            oof_estimator,
                            X_train_scaled,
                            y_train,
                            cv=inner_cv,
                            method="predict_proba",
                            **cv_fit_params,
                        )

                        from respredai.core.metrics import get_threshold_scorer

                        threshold_scorer = get_threshold_scorer(
                            config_handler.threshold_objective,
                            config_handler.vme_cost,
                            config_handler.me_cost,
                        )

                        _, _, thresholds = roc_curve(y_train, y_pred_proba_oof[:, 1])
                        best_score = float("-inf")
                        best_threshold = 0.5
                        for thresh in thresholds:
                            y_pred_thresh = (y_pred_proba_oof[:, 1] >= thresh).astype(int)
                            score = threshold_scorer(y_train.values, y_pred_thresh)
                            if score > best_score:
                                best_score = score
                                best_threshold = thresh

                        best_classifier = best_estimator

                    else:  # threshold_method == "cv"
                        from respredai.core.metrics import get_threshold_scorer

                        threshold_scorer_fn = get_threshold_scorer(
                            config_handler.threshold_objective,
                            config_handler.vme_cost,
                            config_handler.me_cost,
                        )
                        objective_scorer = make_scorer(threshold_scorer_fn)

                        if config_handler.calibrate_probabilities:
                            base_est = clone(grid.estimator)
                            base_est.set_params(**best_params)

                            if datasetter.groups is not None:
                                calib_cv = StratifiedGroupKFold(
                                    n_splits=config_handler.probability_calibration_cv,
                                    shuffle=True,
                                    random_state=config_handler.seed,
                                )
                            else:
                                calib_cv = prob_calib_splits
                            estimator_for_threshold = CalibratedClassifierCV(
                                estimator=base_est,
                                method=config_handler.probability_calibration_method,
                                cv=calib_cv,
                                n_jobs=1,
                            )
                        else:
                            estimator_for_threshold = clone(grid.estimator)
                            estimator_for_threshold.set_params(**best_params)

                        if datasetter.groups is not None:
                            inner_tuner_cv = StratifiedGroupKFold(
                                n_splits=config_handler.inner_folds,
                                shuffle=True,
                                random_state=config_handler.seed,
                            )
                        else:
                            inner_tuner_cv = StratifiedKFold(
                                n_splits=config_handler.inner_folds,
                                shuffle=True,
                                random_state=config_handler.seed,
                            )

                        tuned_model = TunedThresholdClassifierCV(
                            estimator=estimator_for_threshold,
                            cv=inner_tuner_cv,
                            scoring=objective_scorer,
                            n_jobs=1,
                        )

                        fit_kwargs = {}
                        if datasetter.groups is not None:
                            fit_kwargs["groups"] = datasetter.groups[train_idx]
                        tuned_model.fit(X_train_scaled, y_train, **fit_kwargs)
                        best_classifier = tuned_model
                        best_threshold = tuned_model.best_threshold_
                else:
                    best_classifier = best_estimator
                    best_threshold = 0.5

                # Predict on test set
                if config_handler.calibrate_threshold and threshold_method == "cv":
                    y_pred = best_classifier.predict(X_test_scaled)
                    y_prob = best_classifier.predict_proba(X_test_scaled)
                elif config_handler.calibrate_threshold and threshold_method == "oof":
                    y_prob = best_classifier.predict_proba(X_test_scaled)
                    y_pred = (y_prob[:, 1] >= best_threshold).astype(int)
                else:
                    y_prob = best_classifier.predict_proba(X_test_scaled)
                    y_pred = best_classifier.predict(X_test_scaled)

                # Calculate metrics
                temporal_metrics = metric_dict(y_true=y_test.values, y_pred=y_pred, y_prob=y_prob)

                # Save metrics
                model_safe = model_name.replace(" ", "_")
                target_safe = target.replace(" ", "_")
                metrics_path = (
                    Path(config_handler.out_folder)
                    / "metrics"
                    / target_safe
                    / f"{model_safe}_temporal_metrics.csv"
                )

                save_metrics_summary(
                    metrics_dict=[temporal_metrics],
                    output_path=metrics_path,
                    confidence=0.95,
                    n_bootstrap=1_000,
                    random_state=config_handler.seed,
                    y_true_all=y_test.values,
                    y_pred_all=y_pred,
                    y_prob_all=y_prob,
                )

                # Store confusion matrix and metrics for visualization
                cm = confusion_matrix(y_true=y_test, y_pred=y_pred, normalize="true", labels=[0, 1])
                temporal_cms[target] = pd.DataFrame(
                    data=cm,
                    index=["Susceptible", "Resistant"],
                    columns=["Susceptible", "Resistant"],
                )
                temporal_f1s[target] = [temporal_metrics["F1 (weighted)"]]
                temporal_mccs[target] = [temporal_metrics["MCC"]]
                temporal_aurocs[target] = [temporal_metrics["AUROC"]]

                # Generate reliability curve for temporal split
                from respredai.visualization.reliability_curves import save_reliability_curves

                calibration_dir = Path(config_handler.out_folder) / "calibration"
                save_reliability_curves(
                    y_true_list=[y_test.values],
                    y_prob_list=[y_prob[:, 1]],
                    fold_labels=["Temporal"],
                    out_dir=calibration_dir,
                    model=f"{model_safe}_temporal",
                    target=target_safe,
                )

                if config_handler.verbosity:
                    config_handler.logger.info(
                        f"[Temporal] {model_name} - {target}: "
                        f"AUROC={temporal_metrics['AUROC']:.3f}, "
                        f"F1={temporal_metrics['F1 (weighted)']:.3f}, "
                        f"MCC={temporal_metrics['MCC']:.3f}"
                    )

            except Exception as e:
                if config_handler.verbosity:
                    config_handler.logger.error(
                        f"[Temporal] Error for {model_name} - {target}: {e}"
                    )
                warnings.warn(f"[Temporal] {model_name} - {target} failed: {e}")
                continue

        # Save temporal confusion matrices
        if temporal_cms:
            save_cm(
                f1scores=temporal_f1s,
                mccs=temporal_mccs,
                cms=temporal_cms,
                aurocs=temporal_aurocs,
                out_dir=config_handler.out_folder,
                model=f"{model_name.replace(' ', '_')}_temporal",
            )

    if config_handler.verbosity:
        config_handler.logger.info("[Temporal] Temporal validation completed.")


def perform_training(
    datasetter: DataSetter,
    models: list[str],
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

    # OHE: fit on full dataset (no train/test split in perform_training)
    ohe_transformer = _build_ohe_transformer(categorical_cols, config_handler.ohe_min_frequency)
    X = _apply_ohe_and_clean(ohe_transformer, X)

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
            "uncertainty_margin": config_handler.uncertainty_margin,
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
                calibrate_probabilities=config_handler.calibrate_probabilities,
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

            # Post-hoc probability calibration (if enabled)
            if config_handler.calibrate_probabilities:
                if datasetter.groups is not None:
                    prob_calib_cv = StratifiedGroupKFold(
                        n_splits=config_handler.probability_calibration_cv,
                        shuffle=True,
                        random_state=config_handler.seed,
                    )
                    prob_calib_splits = list(prob_calib_cv.split(X_scaled, y, datasetter.groups))
                else:
                    prob_calib_splits = config_handler.probability_calibration_cv

                calibrated_classifier = CalibratedClassifierCV(
                    estimator=best_estimator,
                    method=config_handler.probability_calibration_method,
                    cv=prob_calib_splits,
                    n_jobs=1,
                )
                calibrated_classifier.fit(X_scaled, y)
                best_estimator = calibrated_classifier

                if config_handler.verbosity == 2:
                    config_handler.logger.info(
                        f"Probability calibration applied "
                        f"(method={config_handler.probability_calibration_method})."
                    )

            # Threshold optimization
            best_threshold = 0.5
            if config_handler.calibrate_threshold:
                threshold_method = config_handler.threshold_method
                if threshold_method == "auto":
                    # Heuristic: OOF is more efficient for smaller datasets;
                    # CV wraps the full estimator and is more robust for larger ones.
                    # With <1000 samples, TunedThresholdClassifierCV's internal CV
                    # reduces effective training size significantly.
                    threshold_method = "oof" if len(y) < 1000 else "cv"

                if threshold_method == "oof":
                    # Method 1: Out-of-Fold (OOF) predictions approach
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

                    # Get OOF predictions using the (possibly calibrated)
                    # estimator so threshold is optimized in the same
                    # probability space it will be applied in.
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

                else:  # threshold_method == "cv"
                    # Method 2: TunedThresholdClassifierCV approach
                    from respredai.core.metrics import get_threshold_scorer

                    threshold_scorer_fn = get_threshold_scorer(
                        config_handler.threshold_objective,
                        config_handler.vme_cost,
                        config_handler.me_cost,
                    )
                    objective_scorer = make_scorer(threshold_scorer_fn)

                    # Set best hyperparameters on the unfitted estimator
                    grid.estimator.set_params(**best_params)

                    # Use group-aware CV when groups are specified
                    if datasetter.groups is not None:
                        inner_tuner_cv = StratifiedGroupKFold(
                            n_splits=config_handler.inner_folds,
                            shuffle=True,
                            random_state=config_handler.seed,
                        )
                    else:
                        inner_tuner_cv = StratifiedKFold(
                            n_splits=config_handler.inner_folds,
                            shuffle=True,
                            random_state=config_handler.seed,
                        )

                    tuned_model = TunedThresholdClassifierCV(
                        estimator=grid.estimator,
                        cv=inner_tuner_cv,
                        scoring=objective_scorer,
                        n_jobs=1,
                    )
                    fit_kwargs = {}
                    if datasetter.groups is not None:
                        fit_kwargs["groups"] = datasetter.groups
                    tuned_model.fit(X_scaled, y, **fit_kwargs)
                    best_estimator = tuned_model
                    best_threshold = tuned_model.best_threshold_

            # Save model bundle
            model_bundle = {
                "model": best_estimator,
                "transformer": transformer,
                "ohe_transformer": ohe_transformer,
                "threshold": best_threshold,
                "hyperparams": best_params,
                "feature_names": list(datasetter.X.columns),
                "feature_names_transformed": list(X.columns),
                "target_name": target,
                "model_name": model,
                "training_timestamp": datetime.now().isoformat(),
                "uncertainty_margin": config_handler.uncertainty_margin,
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
) -> dict[str, dict[str, Any]]:
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
        bundle_ohe = bundle["ohe_transformer"]
        threshold = bundle["threshold"]
        uncertainty_margin = bundle.get("uncertainty_margin", 0.1)

        if target_name not in Y_new.columns:
            continue

        y_true = Y_new[target_name].values

        # Apply OHE using the saved training transformer (ensures identical encoding)
        X_encoded = bundle_ohe.transform(X_new)
        X_encoded.columns = X_encoded.columns.str.replace("<", "_lt_", regex=False).str.replace(
            ">", "_gt_", regex=False
        )

        # Scale features
        X_scaled = transformer.transform(X_encoded)

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
            y_prob[:, 1], threshold, margin=uncertainty_margin
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
