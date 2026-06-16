"""Calibration metrics and diagnostics for ResPredAI."""

import numpy as np
from sklearn.metrics import brier_score_loss

# Threshold for detecting near-constant predictions.
# When std(y_prob) < this value, all predictions are treated as identical
# and a single-bin fallback is used to avoid empty/degenerate binning.
_CONSTANT_PREDICTION_THRESHOLD = 1e-10


def _compute_bin_stats(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
    strategy: str,
) -> tuple[list[float], list[float], list[int]]:
    """Compute per-bin accuracy, confidence, and counts.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob : np.ndarray
        Predicted probabilities.
    n_bins : int
        Number of bins.
    strategy : str
        "uniform" or "quantile".

    Returns
    -------
    accuracies : list[float]
        Mean true label per non-empty bin.
    confidences : list[float]
        Mean predicted probability per non-empty bin.
    counts : list[int]
        Sample count per non-empty bin.
    """
    if strategy == "uniform":
        bin_edges = np.linspace(0, 1, n_bins + 1)
    else:
        # Quantile edges can contain duplicates when probabilities are tied;
        # dedupe so tied bins do not silently collapse into empty/degenerate bins.
        bin_edges = np.unique(np.percentile(y_prob, np.linspace(0, 100, n_bins + 1)))

    # Effective number of bins after any deduplication of edges.
    n_effective_bins = max(1, len(bin_edges) - 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])

    accuracies: list[float] = []
    confidences: list[float] = []
    counts: list[int] = []
    for i in range(n_effective_bins):
        mask = bin_indices == i
        count = mask.sum()
        if count > 0:
            accuracies.append(float(y_true[mask].mean()))
            confidences.append(float(y_prob[mask].mean()))
            counts.append(int(count))

    return accuracies, confidences, counts


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
    if np.std(y_prob) < _CONSTANT_PREDICTION_THRESHOLD:
        # All predictions are (nearly) the same - compute single bin ECE
        accuracy = np.mean(y_true)
        confidence = np.mean(y_prob)
        return abs(accuracy - confidence)

    accuracies, confidences, counts = _compute_bin_stats(y_true, y_prob, n_bins, strategy)
    n_samples = len(y_true)
    ece = sum((c / n_samples) * abs(a - p) for a, p, c in zip(accuracies, confidences, counts))
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
    if np.std(y_prob) < _CONSTANT_PREDICTION_THRESHOLD:
        accuracy = np.mean(y_true)
        confidence = np.mean(y_prob)
        return abs(accuracy - confidence)

    accuracies, confidences, counts = _compute_bin_stats(y_true, y_prob, n_bins, strategy)
    if not accuracies:
        return 0.0
    return max(abs(a - p) for a, p in zip(accuracies, confidences))


def compute_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    strategy : str, default="quantile"
        Strategy for defining bins: "uniform" or "quantile".

    Returns
    -------
    prob_true : np.ndarray
        True probability (fraction of positives) in each bin.
    prob_pred : np.ndarray
        Mean predicted probability in each bin.
    bin_counts : np.ndarray
        Number of samples in each bin.

    """
    if len(y_true) == 0:
        return np.array([]), np.array([]), np.array([])

    # Handle edge case where all predictions are the same
    if np.std(y_prob) < _CONSTANT_PREDICTION_THRESHOLD:
        prob_true = np.array([np.mean(y_true)])
        prob_pred = np.array([np.mean(y_prob)])
        bin_counts = np.array([len(y_true)])
        return prob_true, prob_pred, bin_counts

    accuracies, confidences, counts = _compute_bin_stats(y_true, y_prob, n_bins, strategy)
    return np.array(accuracies), np.array(confidences), np.array(counts)


def calibration_metrics_dict(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
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
