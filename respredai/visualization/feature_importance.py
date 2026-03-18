"""Feature importance and coefficient extraction for ResPredAI models."""

import warnings
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TunedThresholdClassifierCV


def unwrap_calibrated_model(model):
    """
    Recursively extract underlying estimator from calibration wrappers.

    Handles nested wrappers when both calibration types are enabled:
    - TunedThresholdClassifierCV (threshold optimization with cv method)
    - CalibratedClassifierCV (probability calibration)

    Returns the innermost model with coef_/feature_importances_.
    """
    # Unwrap TunedThresholdClassifierCV
    if isinstance(model, TunedThresholdClassifierCV):
        if hasattr(model, "estimator_"):
            return unwrap_calibrated_model(model.estimator_)

    # Unwrap CalibratedClassifierCV
    if isinstance(model, CalibratedClassifierCV):
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            return unwrap_calibrated_model(model.calibrated_classifiers_[0].estimator)

    return model


def has_feature_importance(model) -> bool:
    """
    Check if a model has native feature importance or coefficients.

    Unwraps calibration wrappers (CalibratedClassifierCV, TunedThresholdClassifierCV)
    to check the underlying model.

    Parameters
    ----------
    model : sklearn estimator
        The trained model (possibly wrapped).

    Returns
    -------
    bool
        True if underlying model has `feature_importances_` or `coef_` attribute.
    """
    inner_model = unwrap_calibrated_model(model)
    return hasattr(inner_model, "feature_importances_") or hasattr(inner_model, "coef_")


def get_feature_importance(model, feature_names: list[str]) -> Optional[pd.Series]:
    """
    Extract native feature importance or coefficients from a model.

    Unwraps calibration wrappers (CalibratedClassifierCV, TunedThresholdClassifierCV)
    to access the underlying model's coefficients/importances.

    Parameters
    ----------
    model : sklearn estimator
        The trained model (possibly wrapped).
    feature_names : list
        List of feature names.

    Returns
    -------
    pd.Series or None
        Series with feature names as index and importance/coefficient as values.
        Returns signed coefficients for linear models, positive importances for tree-based.
    """
    if model is None:
        return None

    inner_model = unwrap_calibrated_model(model)

    if hasattr(inner_model, "feature_importances_"):
        importances = inner_model.feature_importances_
    elif hasattr(inner_model, "coef_"):
        coef = inner_model.coef_
        if len(coef.shape) > 1:
            # For multi-class, average absolute coefficients across classes
            coef = np.abs(coef).mean(axis=0)
        importances = coef
    else:
        return None

    return pd.Series(importances, index=feature_names)


def compute_shap_importance(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    background_size: int = 100,
    seed: Optional[int] = None,
) -> Optional[pd.Series]:
    """
    Compute SHAP values for a model on test data.

    Parameters
    ----------
    model : sklearn estimator
        Trained model.
    X_test : np.ndarray
        Test data (scaled).
    feature_names : list
        Feature names.
    background_size : int
        Number of background samples for SHAP (default: 100).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.Series or None
        Mean absolute SHAP values per feature.
    """
    if model is None or X_test is None:
        return None

    try:
        X_df = pd.DataFrame(X_test, columns=feature_names)

        if len(X_df) > background_size:
            background = shap.sample(X_df, background_size, random_state=seed)
        else:
            background = X_df

        def predict_fn(x):
            x_df = pd.DataFrame(x, columns=feature_names)
            return model.predict_proba(x_df)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_df)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        return pd.Series(mean_abs_shap, index=feature_names)

    except Exception as e:
        warnings.warn(f"SHAP computation failed: {str(e)}")
        return None


def _resolve_feature_names(
    model, fold_test_data: list, fold_transformers: list, n_features: int
) -> list[str]:
    """
    Resolve feature names from multiple sources, in priority order.

    Checks: (1) outer model's feature_names_in_, (2) inner unwrapped model's
    feature_names_in_, (3) model-specific attributes (e.g. CatBoost feature_names_),
    (4) stored feature names in fold_test_data, (5) fold_transformers'
    get_feature_names_out(), (6) generic feature_N fallback.
    """
    # 1. Check outer model (CalibratedClassifierCV preserves feature_names_in_)
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # 2. Check inner unwrapped model
    inner_model = unwrap_calibrated_model(model)
    if hasattr(inner_model, "feature_names_in_"):
        return list(inner_model.feature_names_in_)

    # 3. Check model-specific attributes (e.g. CatBoost's feature_names_)
    if hasattr(inner_model, "feature_names_") and inner_model.feature_names_:
        return list(inner_model.feature_names_)

    # 4. Check stored feature names from fold_test_data
    for test_data in fold_test_data:
        if test_data is not None and len(test_data) == 2:
            _, stored_names = test_data
            if stored_names is not None and len(stored_names) > 0:
                return list(stored_names)

    # 5. Check fold_transformers for feature names
    for transformer in fold_transformers:
        if transformer is not None and hasattr(transformer, "get_feature_names_out"):
            try:
                return list(transformer.get_feature_names_out())
            except Exception:
                continue

    # 6. Generic fallback
    return [f"feature_{i}" for i in range(n_features)]


def extract_feature_importance_from_models(
    model_path: Path, top_n: Optional[int] = None, use_shap: bool = True, seed: Optional[int] = None
) -> Optional[tuple[pd.DataFrame, list[str], str]]:
    """
    Extract feature importance from a saved model file.

    Uses native importance if available, falls back to SHAP otherwise.

    Parameters
    ----------
    model_path : Path
        Path to the saved model file.
    top_n : int, optional
        Number of top features to return (default: all).
    use_shap : bool
        Whether to use SHAP as fallback (default: True).
    seed : int, optional
        Random seed for SHAP reproducibility.

    Returns
    -------
    tuple or None
        (DataFrame with importances per fold, feature names, method used).
        Method is "native" or "shap". Returns None if not available.
    """
    if not model_path.exists():
        warnings.warn(f"Model file not found: {model_path}")
        return None

    try:
        # Security note (CWE-502): joblib.load() deserialises pickle data and can
        # execute arbitrary code.  Only load model files produced by this project
        # from trusted sources.
        model_data = joblib.load(model_path)
    except Exception as e:
        warnings.warn(f"Failed to load model from {model_path}: {str(e)}")
        return None

    fold_models = model_data.get("fold_models", [])
    fold_test_data = model_data.get("fold_test_data", [])
    fold_transformers = model_data.get("fold_transformers", [])

    if not fold_models:
        warnings.warn(f"No models found in file: {model_path}")
        return None

    first_model = next((m for m in fold_models if m is not None), None)
    if first_model is None:
        return None

    # Try native feature importance first
    if has_feature_importance(first_model):
        inner_model = unwrap_calibrated_model(first_model)
        n_features = (
            inner_model.coef_.shape[1]
            if hasattr(inner_model, "coef_")
            else len(inner_model.feature_importances_)
        )
        feature_names = _resolve_feature_names(
            first_model, fold_test_data, fold_transformers, n_features
        )

        importances_list = []
        for model in fold_models:
            importance = get_feature_importance(model, feature_names)
            if importance is not None:
                importances_list.append(importance)

        if importances_list:
            importances_df = pd.DataFrame(importances_list)
            # fillna(0): features absent in a fold contribute zero importance
            mean_importance = importances_df.fillna(0).mean(axis=0)
            abs_mean_importance = mean_importance.abs().sort_values(ascending=False)

            if top_n is not None:
                top_features = abs_mean_importance.head(top_n).index.tolist()
                importances_df = importances_df[top_features]
            else:
                importances_df = importances_df[abs_mean_importance.index]

            return importances_df, feature_names, "native"

    # Fall back to SHAP if native not available
    if use_shap and fold_test_data:
        importances_list = []
        feature_names = None

        for model, test_data in zip(fold_models, fold_test_data):
            if model is None or test_data is None:
                continue

            X_test, stored_names = test_data
            feat_names = _resolve_feature_names(
                model, [test_data], fold_transformers, X_test.shape[1]
            )

            if feature_names is None:
                feature_names = feat_names

            shap_importance = compute_shap_importance(model, X_test, feat_names, seed=seed)
            if shap_importance is not None:
                importances_list.append(shap_importance)

        if importances_list and feature_names:
            importances_df = pd.DataFrame(importances_list)
            mean_importance = importances_df.mean(axis=0)
            abs_mean_importance = mean_importance.sort_values(ascending=False)

            if top_n is not None:
                top_features = abs_mean_importance.head(top_n).index.tolist()
                importances_df = importances_df[top_features]
            else:
                importances_df = importances_df[abs_mean_importance.index]

            return importances_df, feature_names, "shap"

    return None


def plot_feature_importance(
    importances_df: pd.DataFrame,
    model_name: str,
    target_name: str,
    output_path: Path,
    top_n: int = 20,
    figsize: tuple[int, int] = (10, 8),
    method: str = "native",
):
    """
    Create a barplot of feature importance with error bars.

    Parameters
    ----------
    importances_df : pd.DataFrame
        DataFrame with feature importances (rows=folds, columns=features).
    model_name : str
        Name of the model.
    target_name : str
        Name of the target variable.
    output_path : Path
        Path to save the plot.
    top_n : int
        Number of top features to plot (default: 20).
    figsize : tuple
        Figure size (width, height).
    method : str
        Method used ("native" or "shap").
    """
    mean_importance = importances_df.mean(axis=0)
    std_importance = importances_df.std(axis=0)

    abs_mean = mean_importance.abs()
    top_feature_names = abs_mean.nlargest(top_n).index

    top_features = mean_importance[top_feature_names]
    top_std = std_importance[top_feature_names]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(top_features))

    if method == "shap":
        colors = ["darkorange"] * len(top_features)
        xlabel = "Mean |SHAP value| (mean ± std)"
        title_suffix = "(SHAP)"
    else:
        has_negative = (top_features.values < 0).any()
        if has_negative:
            colors = ["firebrick" if val >= 0 else "seagreen" for val in top_features.values]
        else:
            colors = ["cornflowerblue"] * len(top_features)
        xlabel = "Importance (mean ± std)"
        title_suffix = ""

    ax.barh(
        y_pos,
        top_features.values,
        xerr=top_std.values,
        align="center",
        alpha=0.7,
        ecolor="black",
        capsize=5,
        color=colors,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features.index)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(
        f"Top {top_n} Feature Importance {title_suffix}\nModel: {model_name} | Target: {target_name}"
    )
    ax.grid(axis="x", alpha=0.3)
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.8)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_feature_importance_csv(
    importances_df: pd.DataFrame, output_path: Path, method: str = "native"
):
    """
    Save feature importance to CSV with mean and std.

    Parameters
    ----------
    importances_df : pd.DataFrame
        DataFrame with feature importances (rows=folds, columns=features).
    output_path : Path
        Path to save the CSV file.
    method : str
        Method used ("native" or "shap").
    """
    mean_importance = importances_df.mean(axis=0)
    std_importance = importances_df.std(axis=0)
    abs_mean_importance = mean_importance.abs()

    if method == "shap":
        summary_df = pd.DataFrame(
            {
                "Feature": importances_df.columns,
                "Mean_Abs_SHAP": mean_importance.values,
                "Std_Abs_SHAP": std_importance.values,
                "Mean±Std": [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean_importance, std_importance)],
            }
        )
        summary_df = summary_df.sort_values("Mean_Abs_SHAP", ascending=False)
    else:
        summary_df = pd.DataFrame(
            {
                "Feature": importances_df.columns,
                "Mean_Importance": mean_importance.values,
                "Std_Importance": std_importance.values,
                "Abs_Mean_Importance": abs_mean_importance.values,
                "Mean±Std": [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean_importance, std_importance)],
            }
        )
        summary_df = summary_df.sort_values("Abs_Mean_Importance", ascending=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)


def process_feature_importance(
    output_folder: str,
    model: str,
    target: str,
    top_n: int = 20,
    save_plot: bool = True,
    save_csv: bool = True,
    use_shap: bool = True,
    seed: Optional[int] = None,
) -> Optional[tuple[pd.DataFrame, str]]:
    """
    Process feature importance for a model-target combination.

    Uses native importance if available, falls back to SHAP otherwise.

    Parameters
    ----------
    output_folder : str
        Output folder where trained models are stored.
    model : str
        Model name.
    target : str
        Target name.
    top_n : int
        Number of top features to plot (default: 20).
    save_plot : bool
        Whether to save the plot (default: True).
    save_csv : bool
        Whether to save CSV file (default: True).
    use_shap : bool
        Whether to use SHAP as fallback (default: True).
    seed : int, optional
        Random seed for SHAP reproducibility.

    Returns
    -------
    tuple or None
        (DataFrame with feature importances, method used) or None if not available.
    """
    from respredai.core.models import get_model_path

    model_path = get_model_path(output_folder, model, target)

    result = extract_feature_importance_from_models(
        model_path, top_n=None, use_shap=use_shap, seed=seed
    )

    if result is None:
        warnings.warn(f"Feature importance not available for {model} - {target}.")
        return None

    importances_df, feature_names, method = result

    model_safe = model.replace(" ", "_")
    target_safe = target.replace(" ", "_")

    suffix = "_shap" if method == "shap" else ""

    if save_csv:
        csv_path = (
            Path(output_folder)
            / "feature_importance"
            / target_safe
            / f"{model_safe}_feature_importance{suffix}.csv"
        )
        save_feature_importance_csv(importances_df, csv_path, method=method)

    if save_plot:
        plot_path = (
            Path(output_folder)
            / "feature_importance"
            / target_safe
            / f"{model_safe}_feature_importance{suffix}.png"
        )
        plot_feature_importance(
            importances_df=importances_df,
            model_name=model,
            target_name=target,
            output_path=plot_path,
            top_n=top_n,
            method=method,
        )

    return importances_df, method
