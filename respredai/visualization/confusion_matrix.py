"""Confusion matrix visualization and saving."""

from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def save_cm(
    f1scores: Dict[str, list],
    mccs: Dict[str, list],
    cms: Dict[str, pd.DataFrame],
    aurocs: Dict[str, list],
    out_dir: str,
    model: str,
) -> List[Path]:
    """
    Save individual confusion matrix PNGs for each target.

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

    Returns
    -------
    List[Path]
        List of paths to saved PNG files
    """
    confusion_matrices_dir = Path(out_dir) / "confusion_matrices"
    confusion_matrices_dir.mkdir(parents=True, exist_ok=True)

    model_safe = model.replace(" ", "_")
    saved_paths = []

    for target in cms.keys():
        target_safe = target.replace(" ", "_")

        # Create single figure
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        # Calculate mean and std of metrics
        f1_mean, f1_std = np.nanmean(f1scores[target]), np.nanstd(f1scores[target])
        mcc_mean, mcc_std = np.nanmean(mccs[target]), np.nanstd(mccs[target])
        auroc_mean, auroc_std = np.nanmean(aurocs[target]), np.nanstd(aurocs[target])

        # Create title with metrics
        title_str = (
            f"{target}\n\n"
            f"F1 = {f1_mean:.3f} ± {f1_std:.3f}  |  "
            f"MCC = {mcc_mean:.3f} ± {mcc_std:.3f}  |  "
            f"AUROC = {auroc_mean:.3f} ± {auroc_std:.3f}\n"
        )

        ax.set_title(title_str, color="firebrick", fontsize=11)

        # Create heatmap
        hm = sns.heatmap(
            cms[target],
            annot=True,
            annot_kws={"size": 14},
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            fmt=".3f",
            xticklabels=cms[target].columns if hasattr(cms[target], "columns") else None,
            yticklabels=cms[target].index if hasattr(cms[target], "index") else None,
            ax=ax,
        )

        # Set labels
        ax.set_xlabel("Predicted class", fontsize=12)
        ax.set_ylabel("True class", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)

        # Adjust colorbar
        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)

        # Save figure
        plt.tight_layout()
        output_path = confusion_matrices_dir / f"Confusion_matrix_{model_safe}_{target_safe}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(output_path)

    return saved_paths
