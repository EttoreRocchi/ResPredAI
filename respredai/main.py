"""Main pipeline execution for ResPredAI."""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict, TunedThresholdClassifierCV
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    roc_curve,
    make_scorer,
)

from .utils import ConfigHandler, DataSetter
from .pipe import get_pipeline
from .cm import save_cm


def youden_j_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Youden's J statistic.

    J = Sensitivity + Specificity - 1 = TPR - FPR

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred : np.ndarray
        Predicted binary labels (0 or 1)

    Returns
    -------
    float
        Youden's J statistic, ranging from 0 (random) to 1 (perfect)
    
    Notes
    -----
    Maximizing the Youden's J statistic is equivalent to maximizing the balanced accuracy.
    """
    # TPR (sensitivity) = recall for positive class (label=1)
    tpr = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    # TNR (specificity) = recall for negative class (label=0)
    tnr = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return tpr + tnr - 1


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Calculate comprehensive classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    y_prob : np.ndarray
        Predicted probabilities (2D array)

    Returns
    -------
    dict
        Dictionary with all metrics
    """
    return {
        "Precision (0)": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Precision (1)": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall (0)": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Recall (1)": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1 (0)": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "F1 (1)": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "F1 (weighted)": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Balanced Acc": balanced_accuracy_score(y_true, y_pred),
        "AUROC": roc_auc_score(y_true, y_prob[:, 1]) if len(np.unique(y_true)) > 1 else np.nan,
    }


def ci_halfwidth(std: float, n: int, confidence: float = 0.95) -> float:
    """
    Calculate confidence interval half-width.

    Parameters
    ----------
    std : float
        Standard deviation
    n : int
        Sample size (number of folds)
    confidence : float
        Confidence level (default: 0.95)

    Returns
    -------
    float
        Half-width of the confidence interval
    """
    # Use t-distribution for small sample sizes
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    return t_value * (std / np.sqrt(n))


def save_metrics_summary(
    metrics_dict: dict,
    n_folds: int,
    output_path: Path,
    confidence: float = 0.95
):
    """
    Save metrics summary with mean, std, and confidence intervals.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary with metric names as keys and lists of values as items
    n_folds : int
        Number of cross-validation folds
    output_path : Path
        Path to save the CSV file
    confidence : float
        Confidence level for CI (default: 0.95)
    """
    df_metrics = pd.DataFrame(metrics_dict)
    mean = df_metrics.mean()
    std = df_metrics.std()
    ci = df_metrics.apply(lambda x: ci_halfwidth(x.std(), n_folds, confidence))

    summary_df = pd.DataFrame({
        'Metric': df_metrics.columns,
        'Mean': mean.values,
        'Std': std.values,
        f'CI_{int(confidence*100)}': ci.values,
        'Mean±CI': [f"{m:.3f} ± {c:.3f}" for m, c in zip(mean, ci)]
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    return summary_df


def get_model_path(
    output_folder: str,
    model: str,
    target: str
) -> Path:
    """
    Get the model file path for a model-target combination.

    Parameters
    ----------
    output_folder : str
        Output folder path
    model : str
        Model name
    target : str
        Target name

    Returns
    -------
    Path
        Path to the model file
    """
    model_safe = model.replace(" ", "_")
    target_safe = target.replace(" ", "_")
    models_dir = Path(output_folder) / "models"
    return models_dir / f"{model_safe}_{target_safe}_models.joblib"


def save_models(
    fold_models: list,
    fold_transformers: list,
    fold_thresholds: list,
    fold_hyperparams: list,
    metrics: dict,
    completed_folds: int,
    model_path: Path,
    compression: int = 3
):
    """
    Save trained models with all fold models, thresholds, hyperparameters, and metrics.

    Parameters
    ----------
    fold_models : list
        List of trained models (one per completed fold)
    fold_transformers : list
        List of fitted transformers (one per completed fold)
    fold_thresholds : list
        List of calibrated thresholds (one per completed fold)
    fold_hyperparams : list
        List of best hyperparameters (one per completed fold)
    metrics : dict
        Dictionary containing all metrics for this model-target
    completed_folds : int
        Number of completed folds
    model_path : Path
        Path to save the model file
    compression : int
        Compression level (1-9)
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        'fold_models': fold_models,
        'fold_transformers': fold_transformers,
        'fold_thresholds': fold_thresholds,
        'fold_hyperparams': fold_hyperparams,
        'metrics': metrics,
        'completed_folds': completed_folds,
        'timestamp': pd.Timestamp.now().isoformat()
    }

    joblib.dump(model_data, model_path, compress=compression)


def load_models(model_path: Path) -> dict:
    """
    Load trained models from file.

    Parameters
    ----------
    model_path : Path
        Path to the model file

    Returns
    -------
    dict
        Dictionary with model data, or None if file doesn't exist
    """
    if not model_path.exists():
        return None

    try:
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        warnings.warn(f"Failed to load model from {model_path}: {str(e)}")
        return None


def perform_pipeline(
    datasetter: DataSetter,
    models: list[str],
    config_handler: ConfigHandler,
    progress_callback=None
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
        config_handler.logger.info(
            f"Data dimension: {X.shape}"
        )

    # List of categorical columns (non-continuous)
    categorical_cols = [
        col for col in X.columns
        if col not in datasetter.continuous_features
    ]

    # One-hot encoding of categorical features
    ohe_transformer = ColumnTransformer(
        transformers=[
            (
                "ohe",
                OneHotEncoder(
                    drop=None,
                    sparse_output=False,
                    handle_unknown="ignore"
                ),
                categorical_cols
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    # Apply one-hot encoding
    X = ohe_transformer.fit_transform(X)

    # Clean feature names for XGBoost compatibility (remove <, > characters)
    X.columns = X.columns.str.replace('<', '_lt_', regex=False)
    X.columns = X.columns.str.replace('>', '_gt_', regex=False)

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
                use_groups=(datasetter.groups is not None)
            )
        except Exception as e:
            if config_handler.verbosity:
                config_handler.logger.error(
                    f"Failed to initialize model {model}: {str(e)}"
                )
            warnings.warn(f"Skipping model {model} due to initialization error: {str(e)}")
            if progress_callback:
                total_work_skipped = len(Y.columns) * config_handler.outer_folds
                progress_callback.skip_model(model, total_work_skipped, "initialization error")
            continue

        # Use StratifiedGroupKFold if groups are specified, otherwise StratifiedKFold
        if datasetter.groups is not None:
            outer_cv = StratifiedGroupKFold(
                n_splits=config_handler.outer_folds,
                shuffle=True,
                random_state=config_handler.seed
            )
        else:
            outer_cv = StratifiedKFold(
                n_splits=config_handler.outer_folds,
                shuffle=True,
                random_state=config_handler.seed
            )

        f1scores, mccs, cms, aurocs = {}, {}, {}, {}
        all_metrics = {}  # Store comprehensive metrics

        for target in Y.columns:
            # Check for existing saved models
            model_path = get_model_path(
                config_handler.out_folder,
                model,
                target
            )

            # Try to load saved models
            model_data = None
            start_fold = 0
            fold_models = []
            fold_transformers = []
            fold_thresholds = []
            fold_hyperparams = []

            if config_handler.save_models_enable and model_path.exists():
                model_data = load_models(model_path)
                if model_data is not None:
                    completed_folds = model_data.get('completed_folds', 0)

                    # Check if all folds are completed
                    if completed_folds >= config_handler.outer_folds:
                        if config_handler.verbosity:
                            config_handler.logger.info(
                                f"All folds completed for {model} - {target}. Loading from saved models."
                            )

                        # Restore metrics from saved models
                        all_metrics[target] = model_data['metrics'].get('all_metrics', [])
                        f1scores[target] = model_data['metrics'].get('f1scores', [])
                        mccs[target] = model_data['metrics'].get('mccs', [])
                        cms[target] = model_data['metrics'].get('cms', [])
                        aurocs[target] = model_data['metrics'].get('aurocs', [])

                        if progress_callback:
                            progress_callback.skip_target(target, config_handler.outer_folds, "saved models")

                        continue
                    else:
                        # Resume from last completed fold
                        start_fold = completed_folds
                        fold_models = model_data.get('fold_models', [])
                        fold_transformers = model_data.get('fold_transformers', [])
                        fold_thresholds = model_data.get('fold_thresholds', [])
                        fold_hyperparams = model_data.get('fold_hyperparams', [])

                        # Restore partial metrics
                        all_metrics[target] = model_data['metrics'].get('all_metrics', [])
                        f1scores[target] = model_data['metrics'].get('f1scores', [])
                        mccs[target] = model_data['metrics'].get('mccs', [])
                        cms[target] = model_data['metrics'].get('cms', [])
                        aurocs[target] = model_data['metrics'].get('aurocs', [])

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

            y = Y[target]
            if config_handler.verbosity:
                config_handler.logger.info(
                    f"Starting training for target: {target} (from fold {start_fold + 1})."
                )

            # Start target progress
            if progress_callback:
                progress_callback.start_target(
                    target,
                    total_folds=config_handler.outer_folds,
                    resumed_from=start_fold
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
                    config_handler.logger.info(
                        f"Starting iteration: {i+1}."
                    )

                X_train, X_test = X.iloc[train_set], X.iloc[test_set]
                y_train, y_test = y.iloc[train_set], y.iloc[test_set]

                # Apply scaling
                X_train_scaled = transformer.fit_transform(X_train)
                X_test_scaled = transformer.transform(X_test)

                try:
                    # Pass groups to GridSearchCV if available
                    fit_params = {}
                    if datasetter.groups is not None:
                        fit_params['groups'] = datasetter.groups[train_set]

                    # Step 1: Hyperparameter tuning with GridSearchCV (optimizes ROC-AUC)
                    grid.fit(X=X_train_scaled, y=y_train, **fit_params)

                    if config_handler.verbosity == 2:
                        config_handler.logger.info(
                            f"Model {model} trained for iteration: {i+1}."
                        )

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
                                    random_state=config_handler.seed
                                )
                                cv_fit_params = {'groups': datasetter.groups[train_set]}
                            else:
                                inner_cv = StratifiedKFold(
                                    n_splits=config_handler.inner_folds,
                                    shuffle=True,
                                    random_state=config_handler.seed
                                )
                                cv_fit_params = {}

                            # Get OOF probability predictions on training data
                            y_pred_proba_oof = cross_val_predict(
                                best_estimator,
                                X_train_scaled,
                                y_train,
                                cv=inner_cv,
                                method='predict_proba',
                                **cv_fit_params
                            )

                            # Calculate ROC curve and find optimal threshold using Youden's J
                            fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba_oof[:, 1])
                            youden_j = tpr - fpr
                            best_threshold = thresholds[np.argmax(youden_j)]

                            # The best_estimator is already trained, use it as the final classifier
                            best_classifier = best_estimator

                        else:  # threshold_method == "cv"
                            # Method 2: TunedThresholdClassifierCV approach
                            # Create Youden's J scorer
                            youden_scorer = make_scorer(youden_j_score)

                            # Set best hyperparameters on the unfitted estimator
                            grid.estimator.set_params(**best_params)

                            # Create CV splitter for threshold calibration
                            inner_tuner_cv = StratifiedKFold(
                                n_splits=config_handler.inner_folds,
                                shuffle=True,
                                random_state=config_handler.seed
                            )

                            # Wrap the unfitted estimator in TunedThresholdClassifierCV
                            tuned_model = TunedThresholdClassifierCV(
                                estimator=grid.estimator,
                                cv=inner_tuner_cv,
                                scoring=youden_scorer,
                                n_jobs=1
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
                    fold_metrics = metric_dict(
                        y_true=y_test.values,
                        y_pred=y_pred,
                        y_prob=y_prob
                    )
                    all_metrics[target].append(fold_metrics)

                    # Store individual metrics for backwards compatibility
                    f1scores[target].append(fold_metrics["F1 (weighted)"])
                    mccs[target].append(fold_metrics["MCC"])
                    aurocs[target].append(fold_metrics["AUROC"])
                    cms[target].append(
                        confusion_matrix(
                            y_true=y_test,
                            y_pred=y_pred,
                            normalize="true",
                            labels=[0, 1]
                        )
                    )

                    # Store the best model, transformer, threshold, and hyperparameters for this fold
                    fold_models.append(best_classifier)
                    fold_transformers.append(transformer)
                    fold_thresholds.append(best_threshold)
                    fold_hyperparams.append(best_params)

                    # Update progress for successful fold
                    if progress_callback:
                        progress_callback.complete_fold(i + 1, fold_metrics)

                except Exception as e:
                    if config_handler.verbosity:
                        config_handler.logger.error(
                            f"Error in iteration {i+1} for target {target}: {str(e)}"
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
                        "AUROC": np.nan
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

                    if progress_callback:
                        progress_callback.complete_fold(i + 1, nan_metrics)

                # Save models after each fold if enabled
                if config_handler.save_models_enable:
                    target_metrics = {
                        'all_metrics': all_metrics[target],
                        'f1scores': f1scores[target],
                        'mccs': mccs[target],
                        'cms': cms[target],
                        'aurocs': aurocs[target]
                    }

                    save_models(
                        fold_models=fold_models,
                        fold_transformers=fold_transformers,
                        fold_thresholds=fold_thresholds,
                        fold_hyperparams=fold_hyperparams,
                        metrics=target_metrics,
                        completed_folds=i + 1,
                        model_path=model_path,
                        compression=config_handler.model_compression
                    )

                    if config_handler.verbosity == 2:
                        config_handler.logger.info(
                            f"Saved models after fold {i+1} for {model} - {target}"
                        )

            if config_handler.verbosity:
                config_handler.logger.info(
                    f"Completed training for target {target} with model {model}."
                )

            # Calculate summary metrics for progress callback
            if progress_callback:
                summary_metrics = {
                    'F1 (weighted)': np.nanmean(f1scores[target]),
                    'F1_std': np.nanstd(f1scores[target]),
                    'MCC': np.nanmean(mccs[target]),
                    'MCC_std': np.nanstd(mccs[target]),
                    'AUROC': np.nanmean(aurocs[target]),
                    'AUROC_std': np.nanstd(aurocs[target])
                }
                progress_callback.complete_target(target, summary_metrics)

        # Calculate average confusion matrices
        average_cms = {
            target: pd.DataFrame(
                data=np.nanmean(cms[target], axis=0),
                index=["Susceptible", "Resistant"],
                columns=["Susceptible", "Resistant"]
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
            model=model.replace(" ", "_")
        )

        # Save comprehensive metrics for each target
        model_safe_name = model.replace(" ", "_")
        for target in Y.columns:
            target_safe_name = target.replace(" ", "_")
            metrics_output_path = (
                Path(config_handler.out_folder) / "metrics" / target_safe_name /
                f"{model_safe_name}_metrics_detailed.csv"
            )

            save_metrics_summary(
                metrics_dict=all_metrics[target],
                n_folds=config_handler.outer_folds,
                output_path=metrics_output_path,
                confidence=0.95
            )

            if config_handler.verbosity:
                config_handler.logger.info(
                    f"Saved detailed metrics for {model} - {target} to {metrics_output_path}"
                )

        if config_handler.verbosity:
            config_handler.logger.info(
                f"Completed model {model}."
            )

        # Complete model progress
        if progress_callback:
            progress_callback.complete_model(model)

    # Stop progress tracking
    if progress_callback:
        progress_callback.stop()

    if config_handler.verbosity:
        config_handler.logger.info(
            "Analysis completed."
        )