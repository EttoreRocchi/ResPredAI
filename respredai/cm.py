"""Confusion matrix visualization and saving."""

import math
import os
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def save_cm(
    f1scores: Dict[str, list],
    mccs: Dict[str, list],
    cms: Dict[str, pd.DataFrame],
    aurocs: Dict[str, list],
    out_dir: str,
    model: str
) -> None:
    """
    Save confusion matrices with performance metrics as a figure.

    Parameters
    ----------
    f1scores : Dict[str, list]
        F1 scores for each target
    mccs : Dict[str, list]
        Matthews Correlation Coefficients for each target
    cms : Dict[str, pd.DataFrame]
        Confusion matrices for each target
    aurocs : Dict[str, list]
        AUROC scores for each target
    out_dir : str
        Output directory path
    model : str
        Model name for the output filename
    """

    targets = list(cms.keys())
    n_targets = len(targets)

    # Calculate grid dimensions
    rows = int(math.ceil(math.sqrt(n_targets)))
    cols = int(math.ceil(n_targets / rows))

    # Create figure with subplots
    fig, axs = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=(6 * cols, 6 * rows),
        dpi=300
    )

    # Handle different subplot configurations
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    # Plot confusion matrix for each target
    for i, target in enumerate(targets):
        ax = axs[i]

        # Calculate mean and std of metrics
        f1_mean, f1_std = np.nanmean(f1scores[target]), np.nanstd(f1scores[target])
        mcc_mean, mcc_std = np.nanmean(mccs[target]), np.nanstd(mccs[target])
        auroc_mean, auroc_std = np.nanmean(aurocs[target]), np.nanstd(aurocs[target])

        # Create title with metrics
        title_str = (
            f"\n{target}\n\n"
            f"F1 score = {f1_mean:.3f} ± {f1_std:.3f}\n"
            f"Matthews Correlation Coefficient = {mcc_mean:.3f} ± {mcc_std:.3f}\n"
            f"AUROC = {auroc_mean:.3f} ± {auroc_std:.3f}\n"
        )

        ax.set_title(title_str, color="firebrick", fontsize=14)

        # Create heatmap
        hm = sns.heatmap(
            cms[target],
            annot=True,
            annot_kws={"size": 14},
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            fmt=".3f",
            xticklabels=cms[target].columns if hasattr(cms[target], 'columns') else None,
            yticklabels=cms[target].index if hasattr(cms[target], 'index') else None,
            ax=ax
        )

        # Set labels
        ax.set_xlabel("Predicted class", fontsize=12)
        ax.set_ylabel("True class", fontsize=12)
        ax.tick_params(axis='both', labelsize=10)

        # Adjust colorbar
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)

    # Remove unused subplots
    for j in range(i + 1, rows * cols):
        fig.delaxes(axs[j])

    # Save figure in confusion_matrices folder
    plt.tight_layout()
    confusion_matrices_dir = os.path.join(out_dir, "confusion_matrices")
    os.makedirs(confusion_matrices_dir, exist_ok=True)
    output_path = os.path.join(confusion_matrices_dir, f"Confusion_matrices_{model}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)