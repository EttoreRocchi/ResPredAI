"""Calibration metrics and diagnostics for ResPredAI."""

from typing import Dict, Tuple

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate the Brier score (lower is better).

    The Brier score measures the mean squared difference between the
    predicted probability and the actual outcome.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class (class 1).

    Returns
    -------
    float
        Brier score, ranging from 0 (perfect) to 1 (worst).

    Notes
    -----
    Brier score = (1/n) * sum((p_i - y_i)^2)
    where p_i is the predicted probability and y_i is the true label.
    """
    return brier_score_loss(y_true, y_prob)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    ECE measures the average absolute difference between predicted
    confidence and actual accuracy, weighted by the number of samples
    in each bin.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int, default=10
        Number of bins to use for calibration.
    strategy : str, default="uniform"
        Strategy for defining bin edges: "uniform" or "quantile".

    Returns
    -------
    float
        Expected Calibration Error, ranging from 0 (perfectly calibrated)
        to 1 (maximally miscalibrated).

    Notes
    -----
    ECE = sum_b (|B_b| / n) * |acc(B_b) - conf(B_b)|
    where B_b is bin b, acc is accuracy in the bin, and conf is mean
    predicted probability in the bin.
    """
    if len(y_true) == 0:
        return 0.0

    # Handle edge case where all predictions are the same
    if np.std(y_prob) == 0:
        # All predictions are the same - compute single bin ECE
        accuracy = np.mean(y_true)
        confidence = np.mean(y_prob)
        return abs(accuracy - confidence)

    # Define bin edges based on strategy
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else:  # quantile
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))

    # Assign samples to bins (digitize returns 1-indexed, subtract 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        bin_mask = bin_indices == i
        bin_count = bin_mask.sum()

        if bin_count > 0:
            bin_accuracy = y_true[bin_mask].mean()
            bin_confidence = y_prob[bin_mask].mean()
            ece += (bin_count / n_samples) * abs(bin_accuracy - bin_confidence)

    return ece


def maximum_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """
    Calculate Maximum Calibration Error (MCE).

    MCE is the maximum absolute difference between predicted confidence
    and actual accuracy across all bins.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int, default=10
        Number of bins to use for calibration.
    strategy : str, default="uniform"
        Strategy for defining bin edges: "uniform" or "quantile".

    Returns
    -------
    float
        Maximum Calibration Error, ranging from 0 (perfectly calibrated)
        to 1 (maximally miscalibrated in at least one bin).

    Notes
    -----
    MCE = max_b |acc(B_b) - conf(B_b)|
    """
    if len(y_true) == 0:
        return 0.0

    # Handle edge case where all predictions are the same
    if np.std(y_prob) == 0:
        accuracy = np.mean(y_true)
        confidence = np.mean(y_prob)
        return abs(accuracy - confidence)

    # Define bin edges based on strategy
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else:  # quantile
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))

    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    mce = 0.0

    for i in range(n_bins):
        bin_mask = bin_indices == i
        bin_count = bin_mask.sum()

        if bin_count > 0:
            bin_accuracy = y_true[bin_mask].mean()
            bin_confidence = y_prob[bin_mask].mean()
            mce = max(mce, abs(bin_accuracy - bin_confidence))

    return mce


def compute_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute reliability curve (calibration curve) data.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int, default=10
        Number of bins.
    strategy : str, default="uniform"
        Strategy for defining bins: "uniform" or "quantile".

    Returns
    -------
    prob_true : np.ndarray
        True probability (fraction of positives) in each bin.
    prob_pred : np.ndarray
        Mean predicted probability in each bin.
    bin_counts : np.ndarray
        Number of samples in each bin.

    Notes
    -----
    Uses sklearn's calibration_curve for the core computation.
    """
    if len(y_true) == 0:
        return np.array([]), np.array([]), np.array([])

    # Handle edge case where all predictions are the same
    if np.std(y_prob) == 0:
        prob_true = np.array([np.mean(y_true)])
        prob_pred = np.array([np.mean(y_prob)])
        bin_counts = np.array([len(y_true)])
        return prob_true, prob_pred, bin_counts

    # Use sklearn's calibration_curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)

    # Compute bin counts
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else:
        bin_edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))

    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    bin_counts = np.array([(bin_indices == i).sum() for i in range(n_bins)])

    # Filter to only non-empty bins (to match calibration_curve output)
    non_empty_mask = bin_counts > 0
    bin_counts = bin_counts[non_empty_mask]

    return prob_true, prob_pred, bin_counts


def calibration_metrics_dict(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Calculate all calibration metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int, default=10
        Number of bins for ECE and MCE calculation.

    Returns
    -------
    dict
        Dictionary with keys "Brier Score", "ECE", and "MCE".
    """
    return {
        "Brier Score": brier_score(y_true, y_prob),
        "ECE": expected_calibration_error(y_true, y_prob, n_bins=n_bins),
        "MCE": maximum_calibration_error(y_true, y_prob, n_bins=n_bins),
    }


# Bootstrap wrapper functions for use with METRIC_FUNCTIONS in metrics.py
# These have the signature (y_true, y_pred, y_prob) to match other metrics


def _brier_metric(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score wrapper for bootstrap CI calculation."""
    return brier_score(y_true, y_prob[:, 1])


def _ece_metric(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
    """ECE wrapper for bootstrap CI calculation."""
    return expected_calibration_error(y_true, y_prob[:, 1])


def _mce_metric(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> float:
    """MCE wrapper for bootstrap CI calculation."""
    return maximum_calibration_error(y_true, y_prob[:, 1])


CALIBRATION_METRIC_FUNCTIONS = {
    "Brier Score": _brier_metric,
    "ECE": _ece_metric,
    "MCE": _mce_metric,
}
