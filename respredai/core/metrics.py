"""Metrics calculation utilities for ResPredAI."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
)


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


def bootstrap_ci(
    values: np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    random_state: int = 42
) -> tuple:
    """
    Calculate bootstrap confidence interval.

    Parameters
    ----------
    values : np.ndarray
        Array of metric values from cross-validation folds
    confidence : float
        Confidence level (default: 0.95)
    n_bootstrap : int
        Number of bootstrap resamples (default: 10,000)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the confidence interval
    """
    rng = np.random.default_rng(random_state)
    n = len(values)

    # Bootstrap resampling with replacement
    bootstrap_means = np.array([
        np.mean(rng.choice(values, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    # Calculate percentiles for CI
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)

    return lower, upper


def save_metrics_summary(
    metrics_dict: dict,
    output_path: Path,
    confidence: float = 0.95,
    n_bootstrap: int = 10_000,
    random_state: int = 42
):
    """
    Save metrics summary with mean, std, and bootstrap confidence intervals.

    Parameters
    ----------
    metrics_dict : dict
        Dictionary with metric names as keys and lists of values as items
    output_path : Path
        Path to save the CSV file
    confidence : float
        Confidence level for CI (default: 0.95)
    n_bootstrap : int
        Number of bootstrap resamples (default: 10,000)
    random_state : int
        Random seed for reproducibility
    """
    df_metrics = pd.DataFrame(metrics_dict)
    mean = df_metrics.mean()
    std = df_metrics.std()

    # Calculate bootstrap CI for each metric
    ci_lower = []
    ci_upper = []
    for col in df_metrics.columns:
        lower, upper = bootstrap_ci(
            df_metrics[col].values,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            random_state=random_state
        )
        ci_lower.append(lower)
        ci_upper.append(upper)

    ci_pct = int(confidence * 100)
    summary_df = pd.DataFrame({
        'Metric': df_metrics.columns,
        'Mean': mean.values,
        'Std': std.values,
        f'CI{ci_pct}_lower': ci_lower,
        f'CI{ci_pct}_upper': ci_upper
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    return summary_df
