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


DIR_METRICS = "metrics"
DIR_MODELS = "models"
DIR_TRAINED_MODELS = "trained_models"
DIR_CALIBRATION = "calibration"
DIR_CONFUSION_MATRICES = "confusion_matrices"
DIR_FEATURE_IMPORTANCE = "feature_importance"
DIR_PREDICTIONS = "predictions"


FILE_SUMMARY = "summary.csv"
FILE_SUMMARY_ALL = "summary_all.csv"
FILE_TRAINING_METADATA = "training_metadata.json"
FILE_EVALUATION_SUMMARY = "evaluation_summary.csv"
FILE_REPRODUCIBILITY = "reproducibility.json"


DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_N_BOOTSTRAP = 1_000
DEFAULT_THRESHOLD = 0.5
SAMPLE_SIZE_THRESHOLD_DECISION = 1_000


def sanitize_name(name: str) -> str:
    """Sanitize a model or target name for use in file/directory names.

    Replaces any character that is not a word character, dot, or hyphen
    with an underscore.
    """
    return re.sub(r"[^\w.-]", "_", name)


def sanitize_metric_name(name: str) -> str:
    """Sanitize a metric name for use in CSV columns and dictionary keys.

    First strips parentheses, then replaces any remaining non-word
    characters with underscores.
    """
    return re.sub(r"[^\w]", "_", re.sub(r"[()]", "", name))
