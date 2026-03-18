"""Metrics calculation utilities for ResPredAI."""

import warnings
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from respredai.core.calibration import (
    CALIBRATION_METRIC_FUNCTIONS,
    calibration_metrics_dict,
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
        Youden's J statistic, ranging from -1 (perfectly wrong) to 1 (perfect)

    Notes
    -----
    Maximizing the Youden's J statistic is equivalent to maximizing the balanced accuracy.
    """
    # TPR (sensitivity) = recall for positive class (label=1)
    tpr = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    # TNR (specificity) = recall for negative class (label=0)
    tnr = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return tpr + tnr - 1


def f1_threshold_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F1 score for threshold optimization.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred : np.ndarray
        Predicted binary labels (0 or 1)

    Returns
    -------
    float
        F1 score for the positive class (Resistant)
    """
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)


def f2_threshold_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate F2 score (recall-weighted) for threshold optimization.

    F2 weights recall higher than precision, reducing false negatives.
    Useful in AMR where missing resistance (VME) is more costly.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1)
    y_pred : np.ndarray
        Predicted binary labels (0 or 1)

    Returns
    -------
    float
        F2 score for the positive class (Resistant)
    """
    return fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)


def cost_sensitive_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    vme_cost: float = 1.0,
    me_cost: float = 1.0,
) -> float:
    """
    Calculate negative weighted error cost for threshold optimization.

    Minimizes: vme_cost * VME + me_cost * ME
    Returns negative cost so that maximizing the score minimizes cost.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 = Susceptible, 1 = Resistant)
    y_pred : np.ndarray
        Predicted binary labels
    vme_cost : float
        Cost weight for Very Major Errors (false negatives for resistance)
    me_cost : float
        Cost weight for Major Errors (false positives for resistance)

    Returns
    -------
    float
        Negative weighted error cost (higher is better)
    """
    recall_1 = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall_0 = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    vme = 1 - recall_1  # False negative rate for resistance
    me = 1 - recall_0  # False positive rate for resistance
    cost = vme_cost * vme + me_cost * me
    return -cost  # Negative so we can maximize


def get_threshold_scorer(
    objective: str,
    vme_cost: float = 1.0,
    me_cost: float = 1.0,
):
    """
    Get the appropriate scorer function for threshold optimization.

    Parameters
    ----------
    objective : str
        One of 'youden', 'f1', 'f2', 'cost_sensitive'
    vme_cost : float
        Cost weight for VME (only used if objective='cost_sensitive')
    me_cost : float
        Cost weight for ME (only used if objective='cost_sensitive')

    Returns
    -------
    Callable
        Scorer function that takes (y_true, y_pred) and returns a score
    """
    if objective == "youden":
        return youden_j_score
    elif objective == "f1":
        return f1_threshold_score
    elif objective == "f2":
        return f2_threshold_score
    elif objective == "cost_sensitive":

        def cost_scorer(y_true, y_pred):
            return cost_sensitive_score(y_true, y_pred, vme_cost, me_cost)

        return cost_scorer
    else:
        raise ValueError(f"Unknown threshold objective: {objective}")


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """
    Calculate comprehensive classification metrics including calibration diagnostics.

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
        Dictionary with all metrics including Brier Score, ECE, and MCE
    """
    # Base classification metrics
    metrics = {
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
        "VME": 1 - recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "ME": 1 - recall_score(y_true, y_pred, pos_label=0, zero_division=0),
    }

    # Add calibration diagnostics (always computed, independent of calibration settings)
    calibration_metrics = calibration_metrics_dict(y_true, y_prob[:, 1])
    metrics.update(calibration_metrics)

    return metrics


# Metric wrapper functions for sample-level bootstrapping
def _precision_0_metric(y_true, y_pred, y_prob):
    """Precision for class 0 (Susceptible)."""
    return precision_score(y_true, y_pred, pos_label=0, zero_division=0)


def _precision_1_metric(y_true, y_pred, y_prob):
    """Precision for class 1 (Resistant)."""
    return precision_score(y_true, y_pred, pos_label=1, zero_division=0)


def _recall_0_metric(y_true, y_pred, y_prob):
    """Recall for class 0 (Susceptible)."""
    return recall_score(y_true, y_pred, pos_label=0, zero_division=0)


def _recall_1_metric(y_true, y_pred, y_prob):
    """Recall for class 1 (Resistant)."""
    return recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def _f1_0_metric(y_true, y_pred, y_prob):
    """F1 score for class 0 (Susceptible)."""
    return f1_score(y_true, y_pred, pos_label=0, zero_division=0)


def _f1_1_metric(y_true, y_pred, y_prob):
    """F1 score for class 1 (Resistant)."""
    return f1_score(y_true, y_pred, pos_label=1, zero_division=0)


def _f1_weighted_metric(y_true, y_pred, y_prob):
    """Weighted-average F1 score across both classes."""
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def _mcc_metric(y_true, y_pred, y_prob):
    """Matthews Correlation Coefficient."""
    return matthews_corrcoef(y_true, y_pred)


def _balanced_acc_metric(y_true, y_pred, y_prob):
    """Balanced accuracy (average of per-class recall)."""
    return balanced_accuracy_score(y_true, y_pred)


def _auroc_metric(y_true, y_pred, y_prob):
    """Area Under the ROC Curve (requires both classes present)."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_prob[:, 1])


def _vme_metric(y_true, y_pred, y_prob):
    """VME rate: 1 - Recall(1) = FN / (FN + TP)."""
    return 1 - recall_score(y_true, y_pred, pos_label=1, zero_division=0)


def _me_metric(y_true, y_pred, y_prob):
    """ME rate: 1 - Recall(0) = FP / (FP + TN)."""
    return 1 - recall_score(y_true, y_pred, pos_label=0, zero_division=0)


# Metrics that require both classes: map to uninformative baseline values
# when a bootstrap sample contains only one class. Using baseline values
# instead of skipping avoids optimistic bias in CIs for imbalanced data.
# AUROC=0.5 (random classifier), MCC=0.0 (no correlation).
_SINGLE_CLASS_DEFAULTS = {_auroc_metric: 0.5, _mcc_metric: 0.0}

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
    "VME": _vme_metric,
    "ME": _me_metric,
}

METRIC_FUNCTIONS.update(CALIBRATION_METRIC_FUNCTIONS)


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

    Notes
    -----
    Uses percentile bootstrap which may have under-coverage for small samples
    (n < 100) or highly skewed metric distributions. For improved coverage,
    consider BCa (bias-corrected and accelerated) bootstrap.
    Ref: Efron, B. "Better Bootstrap Confidence Intervals", JASA, 1987.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)

    bootstrap_metrics = []
    n_failures = 0
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        y_true_boot = y_true[indices]

        # For metrics requiring both classes (AUROC, MCC), assign an
        # uninformative baseline value when only one class is present.
        # This avoids optimistic bias from discarding single-class samples,
        # which disproportionately affects imbalanced datasets.
        if len(np.unique(y_true_boot)) < 2 and metric_fn in _SINGLE_CLASS_DEFAULTS:
            bootstrap_metrics.append(_SINGLE_CLASS_DEFAULTS[metric_fn])
            continue

        y_pred_boot = y_pred[indices]
        y_prob_boot = y_prob[indices]

        # Calculate metric on bootstrap sample
        try:
            metric_value = metric_fn(y_true_boot, y_pred_boot, y_prob_boot)
            if not np.isnan(metric_value):
                bootstrap_metrics.append(metric_value)
        except Exception:
            n_failures += 1
            continue

    if n_failures > 0.05 * n_bootstrap:
        warnings.warn(
            f"Bootstrap CI: {n_failures}/{n_bootstrap} iterations failed (>{5}%)",
            stacklevel=2,
        )

    if len(bootstrap_metrics) == 0:
        return np.nan, np.nan

    bootstrap_metrics = np.array(bootstrap_metrics)

    # Calculate percentiles for CI
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_metrics, alpha * 100)
    upper = np.percentile(bootstrap_metrics, (1 - alpha) * 100)

    return lower, upper


def save_metrics_summary(
    metrics_dict: list[dict],
    output_path: Path,
    confidence: float = 0.95,
    n_bootstrap: int = 1_000,
    random_state: int = 42,
    *,
    y_true_all: np.ndarray,
    y_pred_all: np.ndarray,
    y_prob_all: np.ndarray,
    n_folds: int = 0,
    n_repeats: int = 1,
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
    n_folds : int
        Number of outer CV folds (used for repeat-level std with repeated CV)
    n_repeats : int
        Number of CV repeats. When > 1, std is computed across repeat-level
        means rather than across all individual folds, following Bouckaert &
        Frank (2004) for repeated CV variance estimation.
    """
    df_metrics = pd.DataFrame(metrics_dict)
    mean = df_metrics.mean()

    # For repeated CV, compute std across repeat-level means to avoid
    # inflating variance from within-repeat fold correlation.
    if n_repeats > 1 and n_folds > 0:
        expected_rows = n_repeats * n_folds
        if len(df_metrics) != expected_rows:
            warnings.warn(
                f"Expected {expected_rows} fold results ({n_repeats} repeats x "
                f"{n_folds} folds), got {len(df_metrics)}",
                stacklevel=2,
            )
        repeat_means = []
        for r in range(n_repeats):
            fold_slice = df_metrics.iloc[r * n_folds : (r + 1) * n_folds]
            repeat_means.append(fold_slice.mean())
        std = pd.DataFrame(repeat_means).std(ddof=1)
    else:
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


def calculate_uncertainty(
    y_prob: np.ndarray,
    threshold: float,
    margin: float = 0.1,
) -> tuple:
    """
    Calculate uncertainty scores for predictions.

    A prediction is flagged as uncertain if the probability is within
    `margin` of the decision threshold.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities for class 1 (Resistant)
    threshold : float
        Decision threshold
    margin : float
        Margin around threshold for flagging uncertainty (default 0.1)

    Returns
    -------
    tuple
        (uncertainty_scores, is_uncertain)
        - uncertainty_scores: 0 = most certain, 1 = most uncertain (at threshold)
        - is_uncertain: boolean array, True if within margin of threshold
    """
    # Distance from threshold (0 at threshold, up to 0.5 at extremes)
    distance_from_threshold = np.abs(y_prob - threshold)

    # Normalize to 0-1 scale where 0 = most certain, 1 = most uncertain
    # Max possible distance is max(threshold, 1-threshold)
    max_distance = max(threshold, 1 - threshold)
    certainty_score = distance_from_threshold / max_distance

    # Uncertainty is inverse of certainty
    uncertainty_scores = 1 - certainty_score

    # Flag as uncertain if within margin
    is_uncertain = distance_from_threshold < margin

    return uncertainty_scores, is_uncertain
