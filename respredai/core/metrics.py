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
    expected_calibration_error,
    maximum_calibration_error,
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


def metric_dict(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
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
    n_bins : int, default=10
        Number of bins for the ECE/MCE point estimates.

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
        "FOR": 1 - precision_score(y_true, y_pred, pos_label=0, zero_division=0),
    }

    # Add calibration diagnostics (always computed, independent of calibration settings)
    calibration_metrics = calibration_metrics_dict(y_true, y_prob[:, 1], n_bins=n_bins)
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


def _for_metric(y_true, y_pred, y_prob):
    """False Omission Rate (FOR) = 1 - Precision(0) = FN / (FN + TN)."""
    return 1 - precision_score(y_true, y_pred, pos_label=0, zero_division=0)


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
    "FOR": _for_metric,
}

METRIC_FUNCTIONS.update(CALIBRATION_METRIC_FUNCTIONS)


def _ci_metric_fn(col: str, n_bins: int):
    """Return the bootstrap metric function for a metric column.

    For the bin-dependent calibration metrics (ECE, MCE) this binds ``n_bins`` so
    the confidence interval uses the same bin count as the reported point
    estimate. All other metrics use the standard fixed functions.
    """
    if col == "ECE":
        return lambda y_true, y_pred, y_prob: expected_calibration_error(
            y_true, y_prob[:, 1], n_bins=n_bins
        )
    if col == "MCE":
        return lambda y_true, y_pred, y_prob: maximum_calibration_error(
            y_true, y_prob[:, 1], n_bins=n_bins
        )
    return METRIC_FUNCTIONS[col]


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
    Calculate BCa bootstrap confidence interval at the sample level.

    Uses the bias-corrected and accelerated (BCa) bootstrap method via
    ``scipy.stats.bootstrap`` for improved coverage compared to the
    percentile method, especially for small samples (n < 100) and
    bounded/skewed metric distributions (AUROC near 0 or 1, Brier near 0).

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
    Ref: Efron, B. "Better Bootstrap Confidence Intervals", JASA, 1987.

    For metrics requiring both classes (AUROC, MCC), an uninformative
    baseline value is returned when only one class is present in a bootstrap
    sample, avoiding optimistic bias from discarding single-class samples.
    """
    from scipy.stats import bootstrap as scipy_bootstrap

    # Flatten y_prob to 1D (class-1 probabilities) so all arrays have the
    # same shape, avoiding scipy's broadcast warning. The 2D array is
    # reconstructed inside the statistic function.
    y_prob_1d = y_prob[:, 1] if y_prob.ndim == 2 else y_prob

    def _statistic(y_true_b, y_pred_b, y_prob_1d_b, axis=None):
        # Handle single-class bootstrap samples with uninformative defaults
        if len(np.unique(y_true_b)) < 2 and metric_fn in _SINGLE_CLASS_DEFAULTS:
            return _SINGLE_CLASS_DEFAULTS[metric_fn]

        y_prob_b = np.column_stack([1 - y_prob_1d_b, y_prob_1d_b])
        try:
            value = metric_fn(y_true_b, y_pred_b, y_prob_b)
            return value if not np.isnan(value) else np.nan
        except Exception:
            return np.nan

    try:
        result = scipy_bootstrap(
            (y_true, y_pred, y_prob_1d),
            statistic=_statistic,
            n_resamples=n_bootstrap,
            confidence_level=confidence,
            method="BCa",
            random_state=random_state,
            paired=True,
            vectorized=False,
        )
        lower = float(result.confidence_interval.low)
        upper = float(result.confidence_interval.high)

        if np.isnan(lower) or np.isnan(upper):
            warnings.warn(
                "Bootstrap CI: BCa method returned NaN - returning NaN bounds",
                stacklevel=2,
            )
            return np.nan, np.nan

        return lower, upper

    except Exception as exc:
        warnings.warn(
            f"Bootstrap CI: BCa method failed ({exc}) - returning NaN bounds",
            stacklevel=2,
        )
        return np.nan, np.nan


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
    n_bins: int = 10,
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
    n_bins : int, default=10
        Number of bins for the ECE/MCE confidence intervals; matches the bin
        count used for the corresponding point estimates.
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
                metric_fn=_ci_metric_fn(col, n_bins),
                confidence=confidence,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
        else:
            lower, upper = np.nan, np.nan
        ci_lower.append(lower)
        ci_upper.append(upper)

    # Nadeau-Bengio corrected standard error with Bouckaert & Frank
    # variance estimation for repeated CV.
    if n_folds > 0:
        n_test_frac = 1.0 / n_folds
        n_train_frac = 1.0 - n_test_frac
        correction = n_test_frac + (n_test_frac / n_train_frac)
        n_effective = n_repeats if n_repeats > 1 else n_folds
        # Convert ddof=1 variance to ddof=0 (biased) variance
        se = np.sqrt(correction * (std**2) * (n_effective - 1) / n_effective)
    else:
        se = std  # fallback: no fold info available

    ci_pct = int(confidence * 100)
    summary_df = pd.DataFrame(
        {
            "Metric": df_metrics.columns,
            "Mean": mean.values,
            "Std": std.values,
            "SE": se.values,
            f"CI{ci_pct}_lower": ci_lower,
            f"CI{ci_pct}_upper": ci_upper,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    return summary_df


def compute_conformal_qhat(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    alpha: float = 0.1,
) -> dict[int, float]:
    """Compute Mondrian (class-conditional) conformal quantiles from calibration data.

    Uses nonconformity score ``s(x, y) = 1 - p̂(y | x)`` and computes
    separate ``q_hat`` per class for class-conditional coverage.

    The quantile is computed as the exact ``k``-th order statistic where
    ``k = ceil((n + 1) * (1 - alpha))``.  Using an exact order statistic
    (rather than interpolation) is required for the finite-sample coverage
    guarantee.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary 0/1).
    y_prob : np.ndarray
        Predicted probabilities, shape ``(n, 2)``.
    alpha : float
        Miscoverage rate (default 0.1 for 90% coverage).

    Returns
    -------
    dict[int, float]
        ``{0: q_hat_0, 1: q_hat_1}`` per-class quantile thresholds.
    """
    q_hat: dict[int, float] = {}
    for cls in (0, 1):
        mask = y_true == cls
        if mask.sum() == 0:
            q_hat[cls] = 1.0  # conservative: include everything
            continue
        scores = 1 - y_prob[mask, cls]
        n = int(mask.sum())
        # k-th order statistic for the conformal guarantee
        k = int(np.ceil((n + 1) * (1 - alpha)))
        if k > n:
            # Not enough calibration data; include all classes conservatively
            q_hat[cls] = 1.0
        else:
            sorted_scores = np.sort(scores)
            q_hat[cls] = float(sorted_scores[k - 1])  # k-th smallest (1-indexed)
    return q_hat


def conformal_prediction_sets(
    y_prob: np.ndarray,
    q_hat: dict[int, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute conformal prediction sets for each sample.

    A class is included in the prediction set if
    ``1 - p̂(class | x) <= q_hat[class]``.

    Parameters
    ----------
    y_prob : np.ndarray
        Predicted probabilities, shape ``(n, 2)``.
    q_hat : dict[int, float]
        Per-class quantile thresholds from :func:`compute_conformal_qhat`.

    Returns
    -------
    tuple of (set_sizes, is_uncertain)
        ``set_sizes``: int array - number of classes in prediction set per sample
        (0 = empty, 1 = certain, 2 = uncertain).
        ``is_uncertain``: bool array - True if ``set_size != 1``.
    """
    include_0 = (1 - y_prob[:, 0]) <= q_hat[0]
    include_1 = (1 - y_prob[:, 1]) <= q_hat[1]
    set_sizes = include_0.astype(int) + include_1.astype(int)
    is_uncertain = set_sizes != 1
    return set_sizes, is_uncertain


def conformal_coverage_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    q_hat: dict[int, float],
    alpha: float = 0.1,
    is_cv_plus: bool = False,
) -> dict[str, float]:
    """Compute conformal prediction diagnostics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (binary 0/1).
    y_prob : np.ndarray
        Predicted probabilities, shape ``(n, 2)``.
    q_hat : dict[int, float]
        Per-class quantile thresholds.
    alpha : float
        Miscoverage rate (for reference in report).
    is_cv_plus : bool
        If True, the q_hat was computed from CV+ (cross-validated)
        predictions.  The formal CV+ guarantee is ``1 - 2*alpha``,
        not ``1 - alpha``.

    Returns
    -------
    dict[str, float]
        Keys: ``empirical_coverage_class_0``, ``empirical_coverage_class_1``,
        ``empirical_coverage_overall``, ``fraction_uncertain``,
        ``fraction_empty``, ``avg_set_size``, ``guaranteed_coverage``.
    """
    set_sizes, _ = conformal_prediction_sets(y_prob, q_hat)

    # Coverage: true label is in the prediction set
    include_0 = (1 - y_prob[:, 0]) <= q_hat[0]
    include_1 = (1 - y_prob[:, 1]) <= q_hat[1]
    in_set = np.where(y_true == 0, include_0, include_1)

    # CV+: formal guarantee is 1-2α, not 1-α.
    # Split conformal: formal guarantee is 1-α.
    guaranteed = 1 - 2 * alpha if is_cv_plus else 1 - alpha

    report: dict[str, float] = {
        "empirical_coverage_overall": float(in_set.mean()),
        "guaranteed_coverage": guaranteed,
        "avg_set_size": float(set_sizes.mean()),
        "fraction_uncertain": float((set_sizes == 2).mean()),
        "fraction_empty": float((set_sizes == 0).mean()),
    }
    for cls in (0, 1):
        mask = y_true == cls
        if mask.sum() > 0:
            report[f"empirical_coverage_class_{cls}"] = float(in_set[mask].mean())
        else:
            report[f"empirical_coverage_class_{cls}"] = float("nan")
    return report
