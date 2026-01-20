"""
ResPredAI - Antimicrobial Resistance Prediction via AI

A machine learning pipeline for predicting antimicrobial resistance.
"""

__version__ = "1.5.0"
__author__ = "Ettore Rocchi"
__email__ = "ettore.rocchi3@unibo.it"

from respredai.core.metrics import (
    calculate_uncertainty,
    cost_sensitive_score,
    f1_threshold_score,
    f2_threshold_score,
    get_threshold_scorer,
    metric_dict,
    save_metrics_summary,
    youden_j_score,
)
from respredai.core.models import generate_summary_report, get_model_path, load_models, save_models
from respredai.core.pipeline import perform_evaluation, perform_pipeline, perform_training
from respredai.io.config import ConfigHandler, DataSetter
from respredai.visualization.confusion_matrix import save_cm
from respredai.visualization.feature_importance import process_feature_importance
from respredai.visualization.html_report import generate_html_report

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    # Core pipeline
    "perform_pipeline",
    "perform_training",
    "perform_evaluation",
    # Models
    "get_model_path",
    "save_models",
    "load_models",
    "generate_summary_report",
    # Metrics
    "metric_dict",
    "save_metrics_summary",
    "youden_j_score",
    "f1_threshold_score",
    "f2_threshold_score",
    "cost_sensitive_score",
    "get_threshold_scorer",
    "calculate_uncertainty",
    # IO
    "ConfigHandler",
    "DataSetter",
    # Visualization
    "save_cm",
    "process_feature_importance",
    "generate_html_report",
]
