"""Feature importance and coefficient extraction for ResPredAI models."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple
import joblib
import warnings


def has_feature_importance(model) -> bool:
    """
    Check if a model has feature importance or coefficients.

    Parameters
    ----------
    model : sklearn estimator
        The trained model

    Returns
    -------
    bool
        True if model has feature_importances_ or coef_ attribute
    """
    return hasattr(model, 'feature_importances_') or hasattr(model, 'coef_')


def get_feature_importance(model, feature_names: List[str]) -> Optional[pd.Series]:
    """
    Extract feature importance or coefficients from a model.

    Parameters
    ----------
    model : sklearn estimator
        The trained model
    feature_names : list
        List of feature names

    Returns
    -------
    pd.Series or None
        Series with feature names as index and importance/coefficient as values.
        For linear models, returns SIGNED coefficients (preserves direction).
        For tree-based models, returns feature importances (always positive).
    """
    if model is None:
        return None

    if hasattr(model, 'feature_importances_'):
        # For tree-based models (RF, XGB, CatBoost)
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # For linear models (LR, Linear_SVC)
        # Keep signed coefficients to show direction of effect
        coef = model.coef_
        if len(coef.shape) > 1:
            # For binary classification, take the first row
            coef = coef[0]
        importances = coef
    else:
        return None

    return pd.Series(importances, index=feature_names)


def extract_feature_importance_from_models(
    model_path: Path,
    top_n: Optional[int] = None
) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """
    Extract feature importance from a saved model file.

    Parameters
    ----------
    model_path : Path
        Path to the saved model file
    top_n : int, optional
        Number of top features to return (default: all features)

    Returns
    -------
    tuple or None
        (DataFrame with importances for each fold, list of feature names)
        Returns None if model file doesn't exist or models don't support importance
    """
    if not model_path.exists():
        warnings.warn(f"Model file not found: {model_path}")
        return None

    try:
        model_data = joblib.load(model_path)
    except Exception as e:
        warnings.warn(f"Failed to load model from {model_path}: {str(e)}")
        return None

    fold_models = model_data.get('fold_models', [])

    if not fold_models:
        warnings.warn(f"No models found in file: {model_path}")
        return None

    # Check if first non-None model has feature importance
    first_model = next((m for m in fold_models if m is not None), None)
    if first_model is None or not has_feature_importance(first_model):
        return None

    # Get feature names from the first model
    if hasattr(first_model, 'feature_names_in_'):
        feature_names = first_model.feature_names_in_.tolist()
    else:
        # Fallback to generic feature names
        n_features = (first_model.coef_.shape[1] if hasattr(first_model, 'coef_')
                     else len(first_model.feature_importances_))
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Extract importance from each fold
    importances_list = []
    for model in fold_models:
        importance = get_feature_importance(model, feature_names)
        if importance is not None:
            importances_list.append(importance)

    if not importances_list:
        return None

    # Create DataFrame with all folds
    importances_df = pd.DataFrame(importances_list)

    # Calculate mean importance
    mean_importance = importances_df.mean(axis=0)

    # Sort by ABSOLUTE value of mean importance (for ranking)
    # but keep signed values in the data
    abs_mean_importance = mean_importance.abs().sort_values(ascending=False)

    # Select top N features if specified (based on absolute value)
    if top_n is not None:
        top_features = abs_mean_importance.head(top_n).index.tolist()
        importances_df = importances_df[top_features]
    else:
        # Reorder columns by absolute mean importance
        importances_df = importances_df[abs_mean_importance.index]

    return importances_df, feature_names


def plot_feature_importance(
    importances_df: pd.DataFrame,
    model_name: str,
    target_name: str,
    output_path: Path,
    top_n: int = 20,
    figsize: Tuple[int, int] = (10, 8)
):
    """
    Create a barplot of feature importance with error bars.

    Features are selected based on absolute value of mean importance,
    but plotted with signed values (for linear models).

    Parameters
    ----------
    importances_df : pd.DataFrame
        DataFrame with feature importances (rows=folds, columns=features)
    model_name : str
        Name of the model
    target_name : str
        Name of the target variable
    output_path : Path
        Path to save the plot
    top_n : int
        Number of top features to plot (default: 20)
    figsize : tuple
        Figure size (width, height)
    """
    mean_importance = importances_df.mean(axis=0)
    std_importance = importances_df.std(axis=0)

    abs_mean = mean_importance.abs()
    top_feature_names = abs_mean.nlargest(top_n).index

    top_features = mean_importance[top_feature_names]
    top_std = std_importance[top_feature_names]

    fig, ax = plt.subplots(figsize=figsize)

    has_negative = (top_features.values < 0).any()

    y_pos = np.arange(len(top_features))
    if has_negative:
        # Linear models: use firebrick for positive, seagreen for negative
        colors = ['firebrick' if val >= 0 else 'seagreen' for val in top_features.values]
    else:
        # Tree-based models: use cornflowerblue for all bars
        colors = ['cornflowerblue'] * len(top_features)

    ax.barh(y_pos, top_features.values, xerr=top_std.values,
            align='center', alpha=0.7, ecolor='black', capsize=5, color=colors)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features.index)
    ax.invert_yaxis()  # Features read top-to-bottom
    ax.set_xlabel('Importance (mean ± std)')
    ax.set_title(f'Top {top_n} Feature Importance\nModel: {model_name} | Target: {target_name}')
    ax.grid(axis='x', alpha=0.3)
    # Add zero line
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)

    plt.tight_layout()

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_feature_importance_csv(
    importances_df: pd.DataFrame,
    output_path: Path
):
    """
    Save feature importance to CSV with mean and std.

    Features are sorted by absolute value of mean importance,
    but signed values are preserved in the output.

    Parameters
    ----------
    importances_df : pd.DataFrame
        DataFrame with feature importances (rows=folds, columns=features)
    output_path : Path
        Path to save the CSV file
    """
    mean_importance = importances_df.mean(axis=0)
    std_importance = importances_df.std(axis=0)
    abs_mean_importance = mean_importance.abs()

    summary_df = pd.DataFrame({
        'Feature': importances_df.columns,
        'Mean_Importance': mean_importance.values,
        'Std_Importance': std_importance.values,
        'Abs_Mean_Importance': abs_mean_importance.values,
        'Mean±Std': [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean_importance, std_importance)]
    })

    # Sort by ABSOLUTE mean importance (for ranking)
    summary_df = summary_df.sort_values('Abs_Mean_Importance', ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)


def process_feature_importance(
    output_folder: str,
    model: str,
    target: str,
    top_n: int = 20,
    save_plot: bool = True,
    save_csv: bool = True
) -> Optional[pd.DataFrame]:
    """
    Process feature importance for a specific model-target combination.

    Parameters
    ----------
    output_folder : str
        Output folder where trained models are stored
    model : str
        Model name
    target : str
        Target name
    top_n : int
        Number of top features to plot (default: 20)
    save_plot : bool
        Whether to save the plot (default: True)
    save_csv : bool
        Whether to save CSV file (default: True)

    Returns
    -------
    pd.DataFrame or None
        DataFrame with feature importances, or None if not available
    """
    # Get model file path
    from .main import get_model_path

    model_path = get_model_path(output_folder, model, target)

    # Extract importances
    result = extract_feature_importance_from_models(model_path, top_n=None)

    if result is None:
        warnings.warn(
            f"Feature importance not available for {model} - {target}. "
            f"Model may not support feature importance or saved model file may not exist."
        )
        return None

    importances_df, feature_names = result

    # Prepare safe names for file paths
    model_safe = model.replace(" ", "_")
    target_safe = target.replace(" ", "_")

    # Save CSV
    if save_csv:
        csv_path = Path(output_folder) / "feature_importance" / target_safe / f"{model_safe}_feature_importance.csv"
        save_feature_importance_csv(importances_df, csv_path)

    # Save plot
    if save_plot:
        plot_path = Path(output_folder) / "feature_importance" / target_safe / f"{model_safe}_feature_importance.png"
        plot_feature_importance(
            importances_df=importances_df,
            model_name=model,
            target_name=target,
            output_path=plot_path,
            top_n=top_n
        )

    return importances_df
