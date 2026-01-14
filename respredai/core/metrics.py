"""Metrics calculation utilities for ResPredAI."""

from pathlib import Path
from typing import Callable, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
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


# Metric wrapper functions for sample-level bootstrapping
def _precision_0_metric(y_true, y_pred, y_prob):
    return precision_score(y_true, y_pred, pos_label=0, zero_division=0)


def _precision_1_metric(y_true, y_pred, y_prob):
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)


def _recall_0_metric(y_true, y_pred, y_prob):
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


def _recall_1_metric(y_true, y_pred, y_prob):
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def _f1_0_metric(y_true, y_pred, y_prob):
    return f1_score(y_true, y_pred, pos_label=0, zero_division=0)


def _f1_1_metric(y_true, y_pred, y_prob):
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)


def _f1_weighted_metric(y_true, y_pred, y_prob):
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def _mcc_metric(y_true, y_pred, y_prob):
    return matthews_corrcoef(y_true, y_pred)


def _balanced_acc_metric(y_true, y_pred, y_prob):
    return balanced_accuracy_score(y_true, y_pred)


def _auroc_metric(y_true, y_pred, y_prob):
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob[:, 1])


# Mapping from metric names to wrapper functions
METRIC_FUNCTIONS = {
    "Precision (0)": _precision_0_metric,
    "Precision (1)": _precision_1_metric,
    "Recall (0)": _recall_0_metric,
    "Recall (1)": _recall_1_metric,
    "F1 (0)": _f1_0_metric,
    "F1 (1)": _f1_1_metric,
    "F1 (weighted)": _f1_weighted_metric,
    "MCC": _mcc_metric,
    "Balanced Acc": _balanced_acc_metric,
    "AUROC": _auroc_metric,
}


def bootstrap_ci_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable,
    confidence: float = 0.95,
    n_bootstrap: int = 1_000,
    random_state: int = 42,
) -> tuple:
    """
    Calculate bootstrap confidence interval at the sample level.

    This method provides more reliable CIs by bootstrapping on individual
    sample predictions rather than fold-level aggregated metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (concatenated from all folds)
    y_pred : np.ndarray
        Predicted labels (concatenated from all folds)
    y_prob : np.ndarray
        Predicted probabilities (concatenated from all folds, shape [n_samples, 2])
    metric_fn : Callable
        Function that takes (y_true, y_pred, y_prob) and returns a float
    confidence : float
        Confidence level (default: 0.95)
    n_bootstrap : int
        Number of bootstrap resamples (default: 1,000)
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the confidence interval
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    # Pre-generate all bootstrap indices at once for efficiency
    all_indices = rng.choice(n, size=(n_bootstrap, n), replace=True)

    bootstrap_metrics = []
    for indices in all_indices:
        y_true_boot = y_true[indices]

        # Skip if bootstrap sample has only one class
        if len(np.unique(y_true_boot)) < 2:
            continue

        y_pred_boot = y_pred[indices]
        y_prob_boot = y_prob[indices]

        # Calculate metric on bootstrap sample
        try:
            metric_value = metric_fn(y_true_boot, y_pred_boot, y_prob_boot)
            if not np.isnan(metric_value):
                bootstrap_metrics.append(metric_value)
        except Exception:
            continue

    if len(bootstrap_metrics) == 0:
        return np.nan, np.nan

    bootstrap_metrics = np.array(bootstrap_metrics)

    # Calculate percentiles for CI
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_metrics, alpha * 100)
    upper = np.percentile(bootstrap_metrics, (1 - alpha) * 100)

    return lower, upper


def save_metrics_summary(
    metrics_dict: List[dict],
    output_path: Path,
    confidence: float = 0.95,
    n_bootstrap: int = 1_000,
    random_state: int = 42,
    *,
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    y_prob_all: np.ndarray,
):
    """
    Save metrics summary with mean, std, and bootstrap confidence intervals.

    Uses sample-level bootstrapping for reliable confidence intervals.

    Parameters
    ----------
    metrics_dict : List[dict]
        List of dictionaries with metric values for each fold
    output_path : Path
        Path to save the CSV file
    confidence : float
        Confidence level for CI (default: 0.95)
    n_bootstrap : int
        Number of bootstrap resamples (default: 1,000)
    random_state : int
        Random seed for reproducibility
    y_true_all : np.ndarray
        Concatenated true labels from all folds
    y_pred_all : np.ndarray
        Concatenated predicted labels from all folds
    y_prob_all : np.ndarray
        Concatenated predicted probabilities from all folds (shape: [n_samples, 2])
    """
    df_metrics = pd.DataFrame(metrics_dict)
    mean = df_metrics.mean()
    std = df_metrics.std()

    # Calculate bootstrap CI for each metric
    ci_lower = []
    ci_upper = []
    for col in df_metrics.columns:
        if col in METRIC_FUNCTIONS:
            lower, upper = bootstrap_ci_samples(
                y_true=y_true_all,
                y_pred=y_pred_all,
                y_prob=y_prob_all,
                metric_fn=METRIC_FUNCTIONS[col],
                confidence=confidence,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
        else:
            lower, upper = np.nan, np.nan
        ci_lower.append(lower)
        ci_upper.append(upper)

    ci_pct = int(confidence * 100)
    summary_df = pd.DataFrame(
        {
            "Metric": df_metrics.columns,
            "Mean": mean.values,
            "Std": std.values,
            f"CI{ci_pct}_lower": ci_lower,
            f"CI{ci_pct}_upper": ci_upper,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    return summary_df
