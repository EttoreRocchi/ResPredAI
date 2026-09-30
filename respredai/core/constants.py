"""Centralized constants, defaults, and utility functions for ResPredAI."""

import re

THRESHOLD_METHODS = ("auto", "oof", "cv")
THRESHOLD_OBJECTIVES = ("youden", "f1", "f2", "cost_sensitive")
CALIBRATION_METHODS = ("sigmoid", "isotonic")
IMPUTATION_METHODS = ("none", "simple", "knn", "iterative")
IMPUTATION_STRATEGIES = ("mean", "median", "most_frequent", "constant")
IMPUTATION_ESTIMATORS = ("bayesian_ridge", "random_forest")
VALIDATION_STRATEGIES = ("cv", "temporal", "both")
AVAILABLE_MODELS = (
    "LR",
    "XGB",
    "RF",
    "MLP",
    "CatBoost",
    "TabPFN",
    "RBF_SVC",
    "Linear_SVC",
    "KNN",
)
VERBOSITY_LEVELS = (0, 1, 2)

_MODEL_NAME_LOOKUP = {name.lower(): name for name in AVAILABLE_MODELS}


def normalize_model_names(models) -> list[str]:
    """Map user-supplied model names to their canonical spelling.

    Names are matched case-insensitively (``lr`` and ``LR`` both select the
    logistic regression) and returned in canonical form, so every downstream
    lookup (pipeline construction, file names, importance extraction) sees one
    spelling. Empty entries and duplicates are dropped, order is preserved.

    Raises
    ------
    ValueError
        If no model name is given or a name does not match an available model.
    """
    cleaned = [str(m).strip() for m in models if str(m).strip()]
    if not cleaned:
        raise ValueError(f"At least one model is required. Available models: {AVAILABLE_MODELS}")
    unknown = [m for m in cleaned if m.lower() not in _MODEL_NAME_LOOKUP]
    if unknown:
        raise ValueError(f"Unknown model name(s): {unknown}. Available models: {AVAILABLE_MODELS}")
    canonical: list[str] = []
    for m in cleaned:
        name = _MODEL_NAME_LOOKUP[m.lower()]
        if name not in canonical:
            canonical.append(name)
    return canonical


DIR_METRICS = "metrics"
DIR_MODELS = "models"
DIR_TRAINED_MODELS = "trained_models"
DIR_CALIBRATION = "calibration"
DIR_CONFUSION_MATRICES = "confusion_matrices"
DIR_FEATURE_IMPORTANCE = "feature_importance"
DIR_PREDICTIONS = "predictions"
DIR_SUBGROUP = "subgroup_analysis"


FILE_SUMMARY = "summary.csv"
FILE_SUMMARY_ALL = "summary_all.csv"
FILE_TRAINING_METADATA = "training_metadata.json"
FILE_EVALUATION_SUMMARY = "evaluation_summary.csv"
FILE_REPRODUCIBILITY = "reproducibility.json"


DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_N_BOOTSTRAP = 1_000
DEFAULT_THRESHOLD = 0.5
DEFAULT_CONFORMAL_ALPHA = 0.1
DEFAULT_CALIBRATION_BINS = 10
SAMPLE_SIZE_THRESHOLD_DECISION = 1_000
SUBGROUP_MIN_SAMPLES = 10
HIGH_CARDINALITY_MIN_LEVELS = 20
HIGH_CARDINALITY_FRACTION = 0.5

TREE_BASED_MODELS = ("RF", "XGB", "CatBoost")
LINEAR_MODELS = ("LR", "Linear_SVC")
SHAP_FALLBACK_MODELS = ("MLP", "RBF_SVC", "KNN", "TabPFN")
NO_CLASS_WEIGHT_MODELS = ("MLP", "KNN", "TabPFN")


def sanitize_name(name: str) -> str:
    """Sanitize a model or target name for use in file/directory names.

    Replaces any character that is not a word character, dot, or hyphen
    with an underscore.
    """
    return re.sub(r"[^\w.-]", "_", name)


# Characters that XGBoost refuses in feature names, with their replacements.
FEATURE_NAME_REPLACEMENTS = (("<", "_lt_"), (">", "_gt_"), ("[", "_lb_"), ("]", "_rb_"))


def sanitize_feature_name(name) -> str:
    """Replace characters that XGBoost rejects in feature names.

    XGBoost raises on feature names containing ``[``, ``]`` or ``<``; ``>`` is
    replaced as well for symmetry. Applied to the one-hot encoded column names,
    which embed category levels and can carry such characters (``age[0-10]``).
    """
    name = str(name)
    for char, replacement in FEATURE_NAME_REPLACEMENTS:
        name = name.replace(char, replacement)
    return name


def sanitize_feature_names(names) -> list[str]:
    """Apply :func:`sanitize_feature_name` to every name in *names*."""
    return [sanitize_feature_name(n) for n in names]


def sanitize_metric_name(name: str) -> str:
    """Sanitize a metric name for use in CSV columns and dictionary keys.

    First strips parentheses, then replaces any remaining non-word
    characters with underscores.
    """
    return re.sub(r"[^\w]", "_", re.sub(r"[()]", "", name))
