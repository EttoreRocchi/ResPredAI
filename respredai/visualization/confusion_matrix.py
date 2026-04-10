"""Confusion matrix visualization and saving."""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from respredai.core.constants import DIR_CONFUSION_MATRICES, sanitize_name


def save_cm(
    f1scores: dict[str, list],
    mccs: dict[str, list],
    cms: dict[str, pd.DataFrame],
    aurocs: dict[str, list],
    out_dir: str,
    model: str,
) -> list[Path]:
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
    confusion_matrices_dir = Path(out_dir) / DIR_CONFUSION_MATRICES
    confusion_matrices_dir.mkdir(parents=True, exist_ok=True)

    model_safe = sanitize_name(model)
    saved_paths = []

    for target in cms.keys():
        target_safe = sanitize_name(target)

        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            f1_mean, f1_std = np.nanmean(f1scores[target]), np.nanstd(f1scores[target], ddof=1)
            mcc_mean, mcc_std = np.nanmean(mccs[target]), np.nanstd(mccs[target], ddof=1)
            auroc_mean, auroc_std = np.nanmean(aurocs[target]), np.nanstd(aurocs[target], ddof=1)

        def _fmt(name: str, mean: float, std: float) -> str:
            if np.isnan(std):
                return f"{name} = {mean:.3f}"
            return f"{name} = {mean:.3f} ± {std:.3f}"

        title_str = (
            f"{target}\n\n"
            f"{_fmt('F1', f1_mean, f1_std)}  |  "
            f"{_fmt('MCC', mcc_mean, mcc_std)}  |  "
            f"{_fmt('AUROC', auroc_mean, auroc_std)}\n"
        )

        ax.set_title(title_str, color="firebrick", fontsize=11)

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

        ax.set_xlabel("Predicted class", fontsize=12)
        ax.set_ylabel("True class", fontsize=12)
        ax.tick_params(axis="both", labelsize=10)

        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)

        plt.tight_layout()
        output_path = confusion_matrices_dir / f"Confusion_matrix_{model_safe}_{target_safe}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        saved_paths.append(output_path)

    return saved_paths
