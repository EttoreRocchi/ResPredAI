"""Model I/O utilities for ResPredAI."""

import warnings
from pathlib import Path

import joblib
import pandas as pd


def generate_summary_report(output_folder: str, models: list, targets: list) -> None:
    """
    Generate aggregated summary CSVs: one per target plus a global summary_all.csv.

    Parameters
    ----------
    output_folder : str
        Output folder path.
    models : list
        Model names.
    targets : list
        Target names.
    """
    metrics_dir = Path(output_folder) / "metrics"
    all_summaries = []

    for target in targets:
        target_safe = target.replace(" ", "_")
        target_dir = metrics_dir / target_safe
        target_rows = []

        for model in models:
            model_safe = model.replace(" ", "_")
            metrics_file = target_dir / f"{model_safe}_metrics_detailed.csv"

            if not metrics_file.exists():
                continue

            df = pd.read_csv(metrics_file)
            row = {"Model": model, "Target": target}

            for _, metric_row in df.iterrows():
                metric_name = (
                    metric_row["Metric"].replace(" ", "_").replace("(", "").replace(")", "")
                )
                mean_val = metric_row["Mean"]
                std_val = metric_row["Std"]
                row[metric_name] = f"{mean_val:.3f}±{std_val:.3f}"

            target_rows.append(row)
            all_summaries.append(row)

        # Save per-target summary
        if target_rows:
            target_summary_df = pd.DataFrame(target_rows)
            target_summary_df = target_summary_df.drop(columns=["Target"])
            target_summary_path = target_dir / "summary.csv"
            target_summary_df.to_csv(target_summary_path, index=False)

    # Save global summary
    if all_summaries:
        all_summary_df = pd.DataFrame(all_summaries)
        cols = ["Model", "Target"] + [
            c for c in all_summary_df.columns if c not in ["Model", "Target"]
        ]
        all_summary_df = all_summary_df[cols]
        all_summary_path = metrics_dir / "summary_all.csv"
        all_summary_df.to_csv(all_summary_path, index=False)


def get_model_path(output_folder: str, model: str, target: str) -> Path:
    """
    Get the model file path for a model-target combination.

    Parameters
    ----------
    output_folder : str
        Output folder path
    model : str
        Model name
    target : str
        Target name

    Returns
    -------
    Path
        Path to the model file
    """
    model_safe = model.replace(" ", "_")
    target_safe = target.replace(" ", "_")
    models_dir = Path(output_folder) / "models"
    return models_dir / f"{model_safe}_{target_safe}_models.joblib"


def save_models(
    fold_models: list,
    fold_transformers: list,
    fold_thresholds: list,
    fold_hyperparams: list,
    metrics: dict,
    completed_folds: int,
    model_path: Path,
    compression: int = 3,
    fold_test_data: list = None,
):
    """
    Save trained models with all fold data for feature importance (including SHAP).

    Parameters
    ----------
    fold_models : list
        List of trained models (one per completed fold).
    fold_transformers : list
        List of fitted transformers (one per completed fold).
    fold_thresholds : list
        List of calibrated thresholds (one per completed fold).
    fold_hyperparams : list
        List of best hyperparameters (one per completed fold).
    metrics : dict
        Dictionary containing all metrics for this model-target.
    completed_folds : int
        Number of completed folds.
    model_path : Path
        Path to save the model file.
    compression : int
        Compression level (1-9).
    fold_test_data : list, optional
        List of (X_test_scaled, feature_names) tuples for SHAP computation.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model_data = {
        "fold_models": fold_models,
        "fold_transformers": fold_transformers,
        "fold_thresholds": fold_thresholds,
        "fold_hyperparams": fold_hyperparams,
        "metrics": metrics,
        "completed_folds": completed_folds,
        "timestamp": pd.Timestamp.now().isoformat(),
        "fold_test_data": fold_test_data,
    }

    joblib.dump(model_data, model_path, compress=compression)


def load_models(model_path: Path) -> dict:
    """
    Load trained models from file.

    Parameters
    ----------
    model_path : Path
        Path to the model file

    Returns
    -------
    dict
        Dictionary with model data, or None if file doesn't exist
    """
    if not model_path.exists():
        return None

    try:
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        warnings.warn(f"Failed to load model from {model_path}: {str(e)}")
        return None
