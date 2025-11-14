"""Main pipeline execution for ResPredAI."""

__author__ = "Ettore Rocchi"
__email__ = "ettore.rocchi3@unibo.it"

import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import joblib

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    matthews_corrcoef,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)

from .utils import ConfigHandler, DataSetter
from .pipe import get_pipeline
from .cm import save_cm


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


def get_checkpoint_path(
    output_folder: str,
    model: str,
    target: str
) -> Path:
    """
    Get the checkpoint file path for a model-target combination.
    
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
        Path to the checkpoint file
    """
    model_safe = model.replace(" ", "_")
    target_safe = target.replace(" ", "_")
    checkpoint_dir = Path(output_folder) / "checkpoints"
    return checkpoint_dir / f"{model_safe}_{target_safe}_checkpoint.joblib"


def save_checkpoint(
    model,
    metrics: dict,
    checkpoint_path: Path,
    compression: int = 3
):
    """
    Save model checkpoint with metrics.
    
    Parameters
    ----------
    model : estimator
        Trained model to save
    metrics : dict
        Dictionary containing all metrics for this model-target
    checkpoint_path : Path
        Path to save the checkpoint
    compression : int
        Compression level (1-9)
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint_data = {
        'model': model,
        'metrics': metrics,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    joblib.dump(checkpoint_data, checkpoint_path, compress=compression)


def load_checkpoint(checkpoint_path: Path) -> dict:
    """
    Load model checkpoint.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to the checkpoint file
        
    Returns
    -------
    dict
        Dictionary with 'model' and 'metrics' keys, or None if file doesn't exist
    """
    if not checkpoint_path.exists():
        return None
    
    try:
        checkpoint_data = joblib.load(checkpoint_path)
        return checkpoint_data
    except Exception as e:
        warnings.warn(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
        return None


def perform_pipeline(
    datasetter: DataSetter,
    models: list[str],
    config_handler: ConfigHandler,
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
    """

    X, Y = datasetter.X, datasetter.Y
    if config_handler.verbosity:
        config_handler.logger.info(
            f"Data dimension: {X.shape}"
        )

    # One-hot encode categorical features
    ohe = ColumnTransformer(
        transformers=[
            (
                "ohe",
                OneHotEncoder(drop="if_binary", sparse_output=False),
                X.columns.difference(datasetter.continuous_features, sort=False)
            )
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")
    
    X = ohe.fit_transform(X)
    if config_handler.verbosity:
        config_handler.logger.info(
            f"Data dimension after encoding of categorical variables: {X.shape}"
        )

    # Check TabPFN constraints
    for model in models:
        if model == "TabPFN":
            if X.shape[0] > 1000:
                warnings.warn(
                    f"TabPFN has a limit of 1000 samples. Current dataset has {X.shape[0]} samples. "
                    "TabPFN will use a subset or may fail.",
                    UserWarning
                )
            if X.shape[1] > 100:
                warnings.warn(
                    f"TabPFN has a limit of 100 features. Current dataset has {X.shape[1]} features. "
                    "TabPFN may fail or perform suboptimally.",
                    UserWarning
                )

    for model in models:
        if config_handler.verbosity:
            config_handler.logger.info(
                f"Starting analyses with model: {model}."
            )

        try:
            transformer, grid = get_pipeline(
                model_name=model,
                continuous_cols=datasetter.continuous_features,
                inner_folds=config_handler.inner_folds,
                n_jobs=config_handler.n_jobs,
                rnd_state=config_handler.seed
            )
        except Exception as e:
            if config_handler.verbosity:
                config_handler.logger.error(
                    f"Failed to initialize model {model}: {str(e)}"
                )
            warnings.warn(f"Skipping model {model} due to initialization error: {str(e)}")
            continue

        outer_cv = StratifiedKFold(
            n_splits=config_handler.outer_folds,
            shuffle=True,
            random_state=config_handler.seed
        )

        f1scores, mccs, cms, aurocs = {}, {}, {}, {}
        all_metrics = {}  # Store comprehensive metrics
        
        for target in Y.columns:
            # Check for existing checkpoint
            checkpoint_path = get_checkpoint_path(
                config_handler.out_folder,
                model,
                target
            )
            
            if config_handler.checkpoint_enable and checkpoint_path.exists():
                checkpoint_data = load_checkpoint(checkpoint_path)
                if checkpoint_data is not None:
                    if config_handler.verbosity:
                        config_handler.logger.info(
                            f"Loading checkpoint for {model} - {target} from {checkpoint_path}"
                        )
                    
                    # Restore metrics from checkpoint
                    all_metrics[target] = checkpoint_data['metrics'].get('all_metrics', [])
                    f1scores[target] = checkpoint_data['metrics'].get('f1scores', [])
                    mccs[target] = checkpoint_data['metrics'].get('mccs', [])
                    cms[target] = checkpoint_data['metrics'].get('cms', [])
                    aurocs[target] = checkpoint_data['metrics'].get('aurocs', [])
                    
                    if config_handler.verbosity:
                        config_handler.logger.info(
                            f"Checkpoint loaded for {model} - {target}. Skipping training."
                        )
                    continue
            
            # Initialize metrics storage
            f1scores[target] = []
            mccs[target] = []
            cms[target] = []
            aurocs[target] = []
            all_metrics[target] = []  # List of metric dictionaries per fold

            y = Y[target]
            if config_handler.verbosity:
                config_handler.logger.info(
                    f"Starting training for target: {target}."
                )
            
            for i, (train_set, test_set) in enumerate(outer_cv.split(X, y)):
                if config_handler.verbosity == 2:
                    config_handler.logger.info(
                        f"Starting iteration: {i+1}."
                    )
                
                X_train, X_test = X.iloc[train_set], X.iloc[test_set]
                y_train, y_test = y.iloc[train_set], y.iloc[test_set]

                # Apply scaling
                X_train_scaled = transformer.fit_transform(X_train)
                X_test_scaled = transformer.transform(X_test)

                # Handle TabPFN sample limit
                if model == "TabPFN" and X_train_scaled.shape[0] > 1000:
                    # Randomly sample 1000 training samples
                    sample_idx = np.random.RandomState(config_handler.seed + i).choice(
                        X_train_scaled.shape[0], 1000, replace=False
                    )
                    X_train_scaled = X_train_scaled.iloc[sample_idx]
                    y_train = y_train.iloc[sample_idx]

                try:
                    grid.fit(X=X_train_scaled, y=y_train)
                    
                    if config_handler.verbosity == 2:
                        config_handler.logger.info(
                            f"Model {model} trained for iteration: {i+1}."
                        )

                    best_classifier = grid.best_estimator_
                    y_pred = best_classifier.predict(X_test_scaled)
                    
                    # Handle probability prediction
                    if hasattr(best_classifier, 'predict_proba'):
                        y_prob = best_classifier.predict_proba(X_test_scaled)
                    else:
                        # Fallback for models without predict_proba
                        y_score = best_classifier.decision_function(X_test_scaled)
                        # Convert to 2D probability array
                        y_prob = np.column_stack([1 - y_score, y_score])
                    
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
            
            if config_handler.verbosity:
                config_handler.logger.info(
                    f"Completed training for target {target} with model {model}."
                )
            
            # Save checkpoint if enabled
            if config_handler.checkpoint_enable:
                checkpoint_path = get_checkpoint_path(
                    config_handler.out_folder,
                    model,
                    target
                )
                
                # Save metrics for this target
                target_metrics = {
                    'all_metrics': all_metrics[target],
                    'f1scores': f1scores[target],
                    'mccs': mccs[target],
                    'cms': cms[target],
                    'aurocs': aurocs[target]
                }
                
                save_checkpoint(
                    model=grid.best_estimator_ if hasattr(grid, 'best_estimator_') else grid,
                    metrics=target_metrics,
                    checkpoint_path=checkpoint_path,
                    compression=config_handler.checkpoint_compression
                )
                
                if config_handler.verbosity:
                    config_handler.logger.info(
                        f"Saved checkpoint for {model} - {target} to {checkpoint_path}"
                    )

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
            metrics_output_path = Path(config_handler.out_folder) / "metrics" / target_safe_name / f"{model_safe_name}_metrics_detailed.csv"
            
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
    
    if config_handler.verbosity:
        config_handler.logger.info(
            "Analysis completed."
        )