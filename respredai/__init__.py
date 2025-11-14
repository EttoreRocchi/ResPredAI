"""
ResPredAI - Antimicrobial Resistance Prediction via AI

A machine learning pipeline for predicting antimicrobial resistance.
"""

__version__ = "1.0.0"
__author__ = "Ettore Rocchi"
__email__ = "ettore.rocchi3@unibo.it"

from .main import perform_pipeline
from .utils import ConfigHandler, DataSetter
from .pipe import get_pipeline
from .cm import save_cm

__all__ = [
    "perform_pipeline",
    "ConfigHandler",
    "DataSetter",
    "get_pipeline",
    "save_cm",
]