"""
ResPredAI - Antimicrobial Resistance Prediction via AI

A machine learning pipeline for predicting antimicrobial resistance.
"""

__version__ = "1.3.1"
__author__ = "Ettore Rocchi"
__email__ = "ettore.rocchi3@unibo.it"

from respredai.core.pipeline import perform_pipeline, perform_training, perform_evaluation
from respredai.core.models import get_model_path, save_models, load_models, generate_summary_report
from respredai.core.metrics import metric_dict, bootstrap_ci, save_metrics_summary, youden_j_score
from respredai.io.config import ConfigHandler, DataSetter
from respredai.visualization.confusion_matrix import save_cm
from respredai.visualization.feature_importance import process_feature_importance

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
    "bootstrap_ci",
    "save_metrics_summary",
    "youden_j_score",
    # IO
    "ConfigHandler",
    "DataSetter",
    # Visualization
    "save_cm",
    "process_feature_importance",
]
