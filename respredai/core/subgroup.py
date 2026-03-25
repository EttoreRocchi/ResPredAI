"""Subgroup performance evaluation for ResPredAI."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from respredai.core.constants import SUBGROUP_MIN_SAMPLES
from respredai.core.metrics import metric_dict


def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    subgroup_values: np.ndarray,
    subgroup_column_name: str,
) -> pd.DataFrame:
    """Compute the full metric set for each unique value of a subgroup column.

    Parameters
    ----------
    y_true : np.ndarray
        True labels (1-D, binary 0/1).
    y_pred : np.ndarray
        Predicted labels (1-D).
    y_prob : np.ndarray
        Predicted probabilities (2-D, shape ``(n, 2)``).
    subgroup_values : np.ndarray
        Subgroup label for every sample (same length as *y_true*).
    subgroup_column_name : str
        Human-readable name of the subgroup column (used in warnings).

    Returns
    -------
    pd.DataFrame
        One row per unique subgroup value with columns:
        ``Subgroup``, ``N``, ``Prevalence``, plus every metric returned by
        :func:`~respredai.core.metrics.metric_dict`.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    subgroup_values = np.asarray(subgroup_values, dtype=object)

    # Replace NaN / None with "Unknown"
    mask_missing = pd.isna(subgroup_values)
    if mask_missing.any():
        subgroup_values = subgroup_values.copy()
        subgroup_values[mask_missing] = "Unknown"

    unique_groups = np.unique(subgroup_values)
    rows: list[dict] = []

    for group in unique_groups:
        idx = subgroup_values == group
        n = int(idx.sum())

        if n < SUBGROUP_MIN_SAMPLES:
            warnings.warn(
                f"Subgroup '{group}' in column '{subgroup_column_name}' has only "
                f"{n} samples (threshold={SUBGROUP_MIN_SAMPLES}). "
                f"Metrics may be unreliable.",
                stacklevel=2,
            )

        yt = y_true[idx]
        yp = y_pred[idx]
        ypr = y_prob[idx]
        prevalence = float(yt.mean()) if len(yt) > 0 else np.nan

        try:
            metrics = metric_dict(yt, yp, ypr)
        except Exception:
            # If metric computation fails (e.g. single-class subgroup), fill NaN
            metrics = {}

        row: dict = {
            "Subgroup": group,
            "N": n,
            "Prevalence": prevalence,
        }
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def save_subgroup_metrics(
    df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a subgroup metrics DataFrame to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`compute_subgroup_metrics`.
    output_path : Path
        Destination CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
