__author__ = "Ettore Rocchi"
__email__ = "ettore.rocchi3@unibo.it"

import math
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def save_cm( 
    f1scores: dict[str, float],
    mccs: dict[str, float],
    cms: dict[str, np.ndarray],
    aurocs: dict[str, np.ndarray],
    out_dir: str,
    model: str
):

    targets = list(cms.keys())
    n_targets = len(targets)

    rows = int(math.ceil(math.sqrt(n_targets)))
    cols = int(math.ceil(n_targets / rows))

    fig, axs = plt.subplots(nrows=rows, ncols=cols, 
                            figsize=(6 * cols, 6 * rows), 
                            dpi=300)
    
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, target in enumerate(targets):
        ax = axs[i]
        f1_mean, f1_std = np.nanmean(f1scores[target]), np.nanstd(f1scores[target])
        mcc_mean, mcc_std = np.nanmean(mccs[target]), np.nanstd(mccs[target])
        auroc_mean, auroc_std = np.nanmean(aurocs[target]), np.nanstd(aurocs[target])

        title_str = (
            f"\n{target}\n\n"
            f"F1 score = {f1_mean:.3f} ± {f1_std:.3f}\n"
            f"Matthews Correlation Coefficient = {mcc_mean:.3f} ± {mcc_std:.3f}\n"
            f"AUROC = {auroc_mean:.3f} ± {auroc_std:.3f}\n"
        )

        ax.set_title(title_str, color="firebrick", fontsize=14)

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
        ax.set_xlabel("Predicted class", fontsize=12)
        ax.set_ylabel("True class", fontsize=12)
        ax.tick_params(axis='both', labelsize=10)

        cbar = hm.collections[0].colorbar
        cbar.ax.tick_params(labelsize=10)

    for j in range(i + 1, rows * cols):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"Confusion_matrices_{model}.png"), dpi=300)
    plt.close(fig)
