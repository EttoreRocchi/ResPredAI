"""Utility classes for configuration and data handling."""

import logging
import os
from configparser import ConfigParser
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


class ConfigHandler:
    """Handle configuration file parsing and validation."""

    config_path: str
    data_path: str
    targets: List[str]
    continuous_features: List[str]
    group_column: Optional[str]
    models: List[str]
    outer_folds: int
    inner_folds: int
    calibrate_threshold: bool
    threshold_method: str
    threshold_objective: str
    vme_cost: float
    me_cost: float
    uncertainty_margin: float
    seed: int
    verbosity: int
    log_basename: str
    n_jobs: int
    out_folder: str
    save_models_enable: bool
    model_compression: int
    imputation_method: str
    imputation_strategy: str
    imputation_n_neighbors: int
    imputation_estimator: str
    logger: Optional[logging.Logger]

    def __init__(self, config_path: str) -> None:
        """
        Initialize configuration handler.

        Parameters
        ----------
        config_path : str
            Path to the configuration file (.ini format)
        """
        self.config_path = config_path
        self.logger = None
        self._setup_config()
        if self.verbosity:
            self.logger = self._setup_logger(os.path.join(self.out_folder, self.log_basename))

    def _setup_config(self) -> None:
        """Parse and validate configuration file."""
        config = ConfigParser()
        config.read(self.config_path)

        # Section: Data
        self.data_path = config.get("Data", "data_path")
        self.targets = [t.strip() for t in config.get("Data", "targets").split(",")]
        self.continuous_features = [
            f.strip() for f in config.get("Data", "continuous_features").split(",")
        ]
        self.group_column = config.get("Data", "group_column", fallback=None)

        # Section: Pipeline
        self.models = [m.strip() for m in config.get("Pipeline", "models").split(",")]
        self.outer_folds = config.getint("Pipeline", "outer_folds")
        self.inner_folds = config.getint("Pipeline", "inner_folds")
        self.calibrate_threshold = config.getboolean(
            "Pipeline", "calibrate_threshold", fallback=False
        )
        self.threshold_method = config.get("Pipeline", "threshold_method", fallback="auto").lower()
        self.threshold_objective = config.get(
            "Pipeline", "threshold_objective", fallback="youden"
        ).lower()
        self.vme_cost = config.getfloat("Pipeline", "vme_cost", fallback=1.0)
        self.me_cost = config.getfloat("Pipeline", "me_cost", fallback=1.0)

        # Section: Uncertainty
        self.uncertainty_margin = config.getfloat("Uncertainty", "margin", fallback=0.1)

        # Section: Reproducibility
        self.seed = config.getint("Reproducibility", "seed")

        # Section: Log
        self.verbosity = config.getint("Log", "verbosity")
        self.log_basename = config.get("Log", "log_basename")

        # Section: Resources
        self.n_jobs = config.getint("Resources", "n_jobs")

        # Section: Output
        self.out_folder = config.get("Output", "out_folder")

        # Section: ModelSaving
        self.save_models_enable = config.getboolean("ModelSaving", "enable", fallback=False)
        self.model_compression = config.getint("ModelSaving", "compression", fallback=3)
        # Validate compression level (1-9)
        if not 1 <= self.model_compression <= 9:
            raise ValueError(
                f"Model compression must be between 1 and 9, got {self.model_compression}"
            )

        # Validate threshold method
        if self.threshold_method not in ["auto", "oof", "cv"]:
            raise ValueError(
                f"Threshold method must be 'auto', 'oof', or 'cv', got '{self.threshold_method}'"
            )

        # Validate threshold objective
        valid_objectives = ["youden", "f1", "f2", "cost_sensitive"]
        if self.threshold_objective not in valid_objectives:
            raise ValueError(
                f"Threshold objective must be one of {valid_objectives}, "
                f"got '{self.threshold_objective}'"
            )

        # Validate cost weights
        if self.vme_cost <= 0:
            raise ValueError(f"vme_cost must be positive, got {self.vme_cost}")
        if self.me_cost <= 0:
            raise ValueError(f"me_cost must be positive, got {self.me_cost}")

        # Validate uncertainty margin
        if not 0 < self.uncertainty_margin < 0.5:
            raise ValueError(
                f"Uncertainty margin must be between 0 and 0.5, got {self.uncertainty_margin}"
            )

        # Section: Imputation
        self.imputation_method = config.get("Imputation", "method", fallback="none").lower()
        self.imputation_strategy = config.get("Imputation", "strategy", fallback="mean").lower()
        self.imputation_n_neighbors = config.getint("Imputation", "n_neighbors", fallback=5)
        self.imputation_estimator = config.get(
            "Imputation", "estimator", fallback="bayesian_ridge"
        ).lower()

        # Validate imputation method
        valid_methods = ["none", "simple", "knn", "iterative"]
        if self.imputation_method not in valid_methods:
            raise ValueError(
                f"Imputation method must be one of {valid_methods}, got '{self.imputation_method}'"
            )

        # Validate simple imputation strategy
        valid_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.imputation_strategy not in valid_strategies:
            raise ValueError(
                f"Imputation strategy must be one of {valid_strategies}, "
                f"got '{self.imputation_strategy}'"
            )

        # Validate iterative imputer estimator
        valid_estimators = ["bayesian_ridge", "random_forest"]
        if self.imputation_estimator not in valid_estimators:
            raise ValueError(
                f"Imputation estimator must be one of {valid_estimators}, "
                f"got '{self.imputation_estimator}'"
            )

    @staticmethod
    def _setup_logger(log_file: str) -> logging.Logger:
        """
        Set up the logging system.

        Parameters
        ----------
        log_file : str
            Path to the log file

        Returns
        -------
        logging.Logger
            Configured logger instance
        """
        formatter = logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file, mode="w+")
        handler.setFormatter(formatter)

        logger = logging.getLogger("respredai")
        logger.setLevel("INFO")
        logger.addHandler(handler)
        return logger


class DataSetter:
    """Handle data loading and validation."""

    data: pd.DataFrame
    X: pd.DataFrame
    Y: pd.DataFrame
    targets: List[str]
    continuous_features: List[str]
    groups: Optional[np.ndarray]

    def __init__(self, config_handler: ConfigHandler) -> None:
        """
        Initialize data setter.

        Parameters
        ----------
        config_handler : ConfigHandler
            Configuration handler with data paths and parameters
        """
        self.data = self._read_data(config_handler.data_path)
        self._validate_data(self.data, config_handler.targets, config_handler.imputation_method)

        # Extract groups if group_column is specified
        self.groups = None
        if config_handler.group_column:
            if config_handler.group_column not in self.data.columns:
                raise ValueError(
                    f"Group column '{config_handler.group_column}' not found in data. "
                    f"Available columns: {list(self.data.columns)}"
                )
            self.groups = self.data[config_handler.group_column].values
            # Drop group column and targets from X
            self.X = self.data.drop(config_handler.targets + [config_handler.group_column], axis=1)
        else:
            # Drop only targets from X
            self.X = self.data.drop(config_handler.targets, axis=1)

        self.Y = self.data[config_handler.targets]
        self.targets = config_handler.targets
        self.continuous_features = config_handler.continuous_features

    @staticmethod
    def _read_data(data_path: str) -> pd.DataFrame:
        """
        Read data from CSV file.

        Parameters
        ----------
        data_path : str
            Path to the data file

        Returns
        -------
        pd.DataFrame
            Loaded dataframe
        """
        return pd.read_csv(data_path, sep=",", comment="#")

    @staticmethod
    def _validate_data(
        data: pd.DataFrame, targets: Iterable, imputation_method: str = "none"
    ) -> None:
        """
        Validate the loaded data.

        Parameters
        ----------
        data : pd.DataFrame
            The dataframe to validate
        targets : Iterable
            Target column names
        imputation_method : str
            Imputation method from config (none, simple, knn, iterative)

        Raises
        ------
        AssertionError
            If validation fails
        """
        # Check no missing values (only if imputation is disabled)
        if imputation_method == "none":
            assert not data.isnull().values.any(), (
                "Dataset contains missing values. "
                "Enable imputation in config or remove missing values."
            )

        # Check targets in data
        assert set(targets).issubset(data.columns), "Target columns not found in dataset"
