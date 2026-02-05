"""Reliability curve (calibration plot) generation for ResPredAI."""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from respredai.core.calibration import compute_reliability_curve


def plot_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Curve",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plot a reliability curve (calibration plot).

    A reliability curve shows the relationship between predicted probabilities
    and actual outcomes. A perfectly calibrated model would follow the diagonal.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 or 1).
    y_prob : np.ndarray
        Predicted probabilities for the positive class.
    n_bins : int, default=10
        Number of bins for the calibration curve.
    title : str, default="Reliability Curve"
        Title for the plot.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. If None, creates new figure.

    Returns
    -------
    plt.Axes
        The axes with the reliability curve plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # Compute reliability curve data
    prob_true, prob_pred, bin_counts = compute_reliability_curve(y_true, y_prob, n_bins=n_bins)

    # Perfect calibration line (diagonal)
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.7)

    # Model calibration curve
    if len(prob_pred) > 0:
        ax.plot(prob_pred, prob_true, "s-", label="Model", markersize=8, linewidth=2)

    ax.set_xlabel("Mean predicted probability", fontsize=10)
    ax.set_ylabel("Fraction of positives", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    return ax


def save_reliability_curves(
    y_true_list: List[np.ndarray],
    y_prob_list: List[np.ndarray],
    fold_labels: List[str],
    out_dir: Path,
    model: str,
    target: str,
    n_bins: int = 10,
) -> Path:
    """
    Save reliability curves for all folds and an aggregate.

    Creates a multi-panel figure with individual fold curves and an
    aggregate curve combining all folds.

    Parameters
    ----------
    y_true_list : List[np.ndarray]
        List of true labels arrays, one per fold.
    y_prob_list : List[np.ndarray]
        List of predicted probabilities arrays, one per fold.
    fold_labels : List[str]
        Labels for each fold (e.g., ["Fold 1", "Fold 2", ...]).
    out_dir : Path
        Directory to save the output image.
    model : str
        Model name (used in filename).
    target : str
        Target name (used in filename).
    n_bins : int, default=10
        Number of bins for calibration curves.

    Returns
    -------
    Path
        Path to the saved image file.
    """
    n_folds = len(y_true_list)

    if n_folds == 0:
        # No data to plot
        return None

    # Calculate grid dimensions (include +1 for aggregate)
    n_plots = n_folds + 1
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    # Plot individual folds
    for i, (y_true, y_prob, label) in enumerate(zip(y_true_list, y_prob_list, fold_labels)):
        plot_reliability_curve(y_true, y_prob, n_bins=n_bins, title=label, ax=axes[i])

    # Plot aggregate (all folds combined)
    y_true_all = np.concatenate(y_true_list)
    y_prob_all = np.concatenate(y_prob_list)
    plot_reliability_curve(
        y_true_all, y_prob_all, n_bins=n_bins, title="Aggregate", ax=axes[n_folds]
    )

    # Hide unused axes
    for j in range(n_folds + 1, len(axes)):
        axes[j].set_visible(False)

    # Add overall title
    fig.suptitle(
        f"Reliability Curves - {model} - {target}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    # Save the figure
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"reliability_curve_{model}_{target}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return out_path
