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

from respredai.core.constants import (
    DIR_FEATURE_IMPORTANCE,
    TREE_BASED_MODELS,
    sanitize_name,
)


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


def _shap_direction_from_correlation(
    shap_values: np.ndarray, X_values: np.ndarray, feature_names: list[str]
) -> pd.Series:
    """Determine direction via Spearman correlation between feature values and SHAP values.

    A positive correlation means higher feature values push toward class 1
    (risk factor); negative means they push toward class 0 (protective).
    This is robust to class imbalance, unlike taking the mean of SHAP values.
    """
    from scipy.stats import spearmanr

    directions = np.zeros(len(feature_names))
    for i in range(len(feature_names)):
        feat_col = X_values[:, i] if X_values.ndim == 2 else X_values
        shap_col = shap_values[:, i]
        # Constant columns have no direction
        if np.std(feat_col) == 0 or np.std(shap_col) == 0:
            directions[i] = 0.0
            continue
        corr, _ = spearmanr(feat_col, shap_col)
        directions[i] = corr if not np.isnan(corr) else 0.0
    return pd.Series(directions, index=feature_names)


def compute_feature_direction(
    model,
    X_test: np.ndarray,
    feature_names: list[str],
    model_name: str,
    background_size: int = 100,
    seed: Optional[int] = None,
) -> Optional[pd.Series]:
    """Compute signed feature direction (positive = risk, negative = protective).

    For linear models, the sign of the coefficient is used directly.
    For all other models, SHAP values are computed and the direction is
    determined via Spearman correlation between feature values and their
    SHAP contributions - this is robust to class imbalance.

    Parameters
    ----------
    model : sklearn estimator
        Trained model (possibly wrapped in calibration).
    X_test : np.ndarray
        Test data (scaled).
    feature_names : list[str]
        Feature names matching the column order of *X_test*.
    model_name : str
        Model identifier (e.g. ``"LR"``, ``"RF"``).
    background_size : int
        Background samples for KernelExplainer fallback.
    seed : int, optional
        Random seed.

    Returns
    -------
    pd.Series or None
        Signed direction per feature (positive = risk, negative = protective).
    """
    if model is None or X_test is None:
        return None

    inner_model = unwrap_calibrated_model(model)

    # Linear models: use sign of coefficients directly (fast, no extra computation)
    if hasattr(inner_model, "coef_"):
        coef = inner_model.coef_
        if len(coef.shape) > 1:
            # Binary classification: use class-1 coefficients
            coef = coef[0] if coef.shape[0] == 1 else coef.mean(axis=0)
        return pd.Series(coef, index=feature_names)

    try:
        X_df = pd.DataFrame(X_test, columns=feature_names)
        X_arr = np.asarray(X_test)

        # Tree-based models: use TreeExplainer (fast, exact)
        if model_name in TREE_BASED_MODELS and hasattr(inner_model, "predict"):
            try:
                explainer = shap.TreeExplainer(inner_model)
                shap_values = explainer.shap_values(X_df)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # class 1 (resistant)
                return _shap_direction_from_correlation(
                    np.asarray(shap_values), X_arr, feature_names
                )
            except Exception:
                pass  # Fall through to KernelExplainer

        # Fallback: KernelExplainer (model-agnostic, slower)
        if len(X_df) > background_size:
            background = shap.sample(X_df, background_size, random_state=seed)
        else:
            background = X_df

        def predict_fn(x):
            x_df = pd.DataFrame(x, columns=feature_names)
            return model.predict_proba(x_df)[:, 1]

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X_df)
        return _shap_direction_from_correlation(np.asarray(shap_values), X_arr, feature_names)

    except Exception as e:
        warnings.warn(f"Feature direction computation failed: {str(e)}")
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


def _compute_directions_from_folds(
    fold_models: list,
    fold_test_data: list,
    fold_transformers: list,
    feature_names: list[str],
    model_name: Optional[str],
    seed: Optional[int],
) -> Optional[pd.Series]:
    """Aggregate signed direction across folds and return direction labels.

    Parameters
    ----------
    fold_models : list
        Trained models per fold (may contain ``None``).
    fold_test_data : list
        Per-fold ``(X_test, feature_names)`` tuples.
    fold_transformers : list
        Per-fold transformers (unused but kept for API symmetry).
    feature_names : list[str]
        Canonical feature names to align results to.
    model_name : str or None
        Model identifier (e.g. ``"LR"``, ``"RF"``).
    seed : int or None
        Random seed for SHAP reproducibility.

    Returns
    -------
    pd.Series or None
        Direction labels (``"Risk (+)"`` / ``"Protective (-)"``),
        or ``None`` if direction could not be computed.
    """
    direction_list: list[pd.Series] = []

    for fold_idx, model in enumerate(fold_models):
        if model is None:
            continue

        X_test = None
        original_names = None
        if fold_idx < len(fold_test_data) and fold_test_data[fold_idx] is not None:
            X_test, original_names = fold_test_data[fold_idx]

        if X_test is None or original_names is None:
            continue

        # Use original_names (matches X_test column order) for SHAP / coef_
        # computation, not the importance-reordered feature_names.
        signed = compute_feature_direction(
            model,
            X_test,
            list(original_names),
            model_name=model_name or "",
            seed=seed,
        )
        if signed is not None:
            direction_list.append(signed)

    if not direction_list:
        return None

    mean_signed = pd.DataFrame(direction_list).fillna(0).mean(axis=0)
    # Align to requested feature_names
    mean_signed = mean_signed.reindex(feature_names, fill_value=0)
    return mean_signed.apply(lambda v: "Risk (+)" if v >= 0 else "Protective (-)")


def extract_feature_importance_from_models(
    model_path: Path,
    top_n: Optional[int] = None,
    use_shap: bool = True,
    seed: Optional[int] = None,
    compute_direction_flag: bool = False,
    model_name: Optional[str] = None,
) -> Optional[tuple[pd.DataFrame, list[str], str, Optional[pd.Series]]]:
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
    compute_direction_flag : bool
        Whether to compute signed feature direction (default: False).
    model_name : str, optional
        Model identifier (e.g. ``"LR"``, ``"RF"``).  Required when
        *compute_direction_flag* is True.

    Returns
    -------
    tuple or None
        ``(importances_df, feature_names, method, directions)``.
        *directions* is a :class:`pd.Series` mapping feature names to
        ``"Risk (+)"`` / ``"Protective (-)"`` strings, or ``None`` when
        *compute_direction_flag* is False.
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
        for fold_idx, model in enumerate(fold_models):
            if model is None:
                continue
            inner = unwrap_calibrated_model(model)
            n_feat = (
                inner.coef_.shape[1] if hasattr(inner, "coef_") else len(inner.feature_importances_)
            )
            fold_td = [fold_test_data[fold_idx]] if fold_idx < len(fold_test_data) else []
            fold_tr = [fold_transformers[fold_idx]] if fold_idx < len(fold_transformers) else []
            fold_names = _resolve_feature_names(model, fold_td, fold_tr, n_feat)
            importance = get_feature_importance(model, fold_names)
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

            feature_names = importances_df.columns.tolist()

            # Compute direction if requested
            directions = None
            if compute_direction_flag:
                directions = _compute_directions_from_folds(
                    fold_models,
                    fold_test_data,
                    fold_transformers,
                    feature_names,
                    model_name,
                    seed,
                )

            return importances_df, feature_names, "native", directions

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

            # For SHAP fallback, direction can be derived from signed SHAP
            directions = None
            if compute_direction_flag:
                directions = _compute_directions_from_folds(
                    fold_models,
                    fold_test_data,
                    fold_transformers,
                    feature_names,
                    model_name,
                    seed,
                )

            return importances_df, feature_names, "shap", directions

    return None


def plot_feature_importance(
    importances_df: pd.DataFrame,
    model_name: str,
    target_name: str,
    output_path: Path,
    top_n: int = 20,
    figsize: tuple[int, int] = (10, 8),
    method: str = "native",
    directions: Optional[pd.Series] = None,
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
    directions : pd.Series, optional
        Direction labels per feature (``"Risk (+)"`` / ``"Protective (-)"``).
    """
    mean_importance = importances_df.mean(axis=0)
    std_importance = importances_df.std(axis=0)

    abs_mean = mean_importance.abs()
    top_feature_names = abs_mean.nlargest(top_n).index

    top_features = mean_importance[top_feature_names]
    top_std = std_importance[top_feature_names]

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(top_features))

    if directions is not None:
        # Color by direction: risk = firebrick, protective = seagreen
        colors = [
            "firebrick" if directions.get(feat, "Risk (+)") == "Risk (+)" else "seagreen"
            for feat in top_features.index
        ]
        xlabel = "Importance (mean ± std)"
        title_suffix = "(with direction)"
    elif method == "shap":
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

    # Add legend if direction coloring is used
    if directions is not None:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="firebrick", alpha=0.7, label="Risk (+)"),
            Patch(facecolor="seagreen", alpha=0.7, label="Protective (-)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_feature_importance_csv(
    importances_df: pd.DataFrame,
    output_path: Path,
    method: str = "native",
    directions: Optional[pd.Series] = None,
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
    directions : pd.Series, optional
        Direction labels per feature (``"Risk (+)"`` / ``"Protective (-)"``).
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

    if directions is not None:
        summary_df["Direction"] = summary_df["Feature"].map(directions).fillna("")

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
    compute_direction: bool = False,
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
    compute_direction : bool
        Whether to compute signed feature direction (default: False).

    Returns
    -------
    tuple or None
        (DataFrame with feature importances, method used) or None if not available.
    """
    from respredai.core.models import get_model_path

    model_path = get_model_path(output_folder, model, target)

    result = extract_feature_importance_from_models(
        model_path,
        top_n=None,
        use_shap=use_shap,
        seed=seed,
        compute_direction_flag=compute_direction,
        model_name=model,
    )

    if result is None:
        warnings.warn(f"Feature importance not available for {model} - {target}.")
        return None

    importances_df, feature_names, method, directions = result

    model_safe = sanitize_name(model)
    target_safe = sanitize_name(target)

    suffix = "_shap" if method == "shap" else ""

    if save_csv:
        csv_path = (
            Path(output_folder)
            / DIR_FEATURE_IMPORTANCE
            / target_safe
            / f"{model_safe}_feature_importance{suffix}.csv"
        )
        save_feature_importance_csv(importances_df, csv_path, method=method, directions=directions)

    if save_plot:
        plot_path = (
            Path(output_folder)
            / DIR_FEATURE_IMPORTANCE
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
            directions=directions,
        )

    return importances_df, method
