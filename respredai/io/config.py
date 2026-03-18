"""Utility classes for configuration and data handling."""

import logging
import os
from collections.abc import Iterable
from configparser import ConfigParser
from typing import Optional

import numpy as np
import pandas as pd


class ConfigHandler:
    """Handle configuration file parsing and validation."""

    config_path: str
    data_path: str
    targets: list[str]
    continuous_features: list[str]
    group_column: Optional[str]
    models: list[str]
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
    ohe_min_frequency: Optional[float]
    calibrate_probabilities: bool
    probability_calibration_method: str
    probability_calibration_cv: int
    outer_cv_repeats: int
    validation_strategy: str
    temporal_split_column: Optional[str]
    temporal_split_date: Optional[str]
    temporal_split_ratio: Optional[float]
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

    def initialize_logger(self) -> None:
        """Set up the logger using the current out_folder. Call after any CLI overrides."""
        if self.verbosity and self.logger is None:
            self.logger = self._setup_logger(os.path.join(self.out_folder, self.log_basename))

    def _setup_config(self) -> None:
        """Parse and validate configuration file."""
        if not os.path.isfile(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        config = ConfigParser()
        config.read(self.config_path)

        self._parse_data_section(config)
        self._parse_pipeline_section(config)
        self._parse_misc_sections(config)
        self._parse_imputation_section(config)
        self._parse_validation_section(config)
        self._parse_preprocessing_section(config)
        self._validate_cross_field_constraints()

    def _parse_data_section(self, config: ConfigParser) -> None:
        """Parse [Data] section."""
        self.data_path = config.get("Data", "data_path")
        self.targets = [t.strip() for t in config.get("Data", "targets").split(",")]
        self.continuous_features = [
            f.strip() for f in config.get("Data", "continuous_features").split(",")
        ]
        self.group_column = config.get("Data", "group_column", fallback=None)

    def _parse_pipeline_section(self, config: ConfigParser) -> None:
        """Parse [Pipeline] section including probability calibration."""
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
        self.outer_cv_repeats = config.getint("Pipeline", "outer_cv_repeats", fallback=1)

        # Probability calibration settings
        self.calibrate_probabilities = config.getboolean(
            "Pipeline", "calibrate_probabilities", fallback=False
        )
        self.probability_calibration_method = config.get(
            "Pipeline", "probability_calibration_method", fallback="sigmoid"
        ).lower()
        self.probability_calibration_cv = config.getint(
            "Pipeline", "probability_calibration_cv", fallback=5
        )

    def _parse_misc_sections(self, config: ConfigParser) -> None:
        """Parse [Uncertainty], [Reproducibility], [Log], [Resources], [Output], [ModelSaving]."""
        self.uncertainty_margin = config.getfloat("Uncertainty", "margin", fallback=0.1)
        self.seed = config.getint("Reproducibility", "seed")
        self.verbosity = config.getint("Log", "verbosity")
        self.log_basename = config.get("Log", "log_basename")
        self.n_jobs = config.getint("Resources", "n_jobs")
        self.out_folder = config.get("Output", "out_folder")
        self.save_models_enable = config.getboolean("ModelSaving", "enable", fallback=False)
        self.model_compression = config.getint("ModelSaving", "compression", fallback=3)

    def _parse_imputation_section(self, config: ConfigParser) -> None:
        """Parse [Imputation] section."""
        self.imputation_method = config.get("Imputation", "method", fallback="none").lower()
        self.imputation_strategy = config.get("Imputation", "strategy", fallback="mean").lower()
        self.imputation_n_neighbors = config.getint("Imputation", "n_neighbors", fallback=5)
        self.imputation_estimator = config.get(
            "Imputation", "estimator", fallback="bayesian_ridge"
        ).lower()

    def _parse_validation_section(self, config: ConfigParser) -> None:
        """Parse [Validation] section."""
        self.validation_strategy = config.get(
            "Validation", "validation_strategy", fallback="cv"
        ).lower()
        self.temporal_split_column = config.get(
            "Validation", "temporal_split_column", fallback=None
        )
        self.temporal_split_date = config.get("Validation", "temporal_split_date", fallback=None)
        temporal_split_ratio_str = config.get("Validation", "temporal_split_ratio", fallback=None)
        self.temporal_split_ratio = (
            float(temporal_split_ratio_str) if temporal_split_ratio_str is not None else None
        )

    def _parse_preprocessing_section(self, config: ConfigParser) -> None:
        """Parse [Preprocessing] section."""
        ohe_min_freq_str = config.get("Preprocessing", "ohe_min_frequency", fallback=None)
        if ohe_min_freq_str is not None:
            self.ohe_min_frequency = config.getfloat("Preprocessing", "ohe_min_frequency")
            if self.ohe_min_frequency <= 0:
                raise ValueError(
                    f"ohe_min_frequency must be positive, got {self.ohe_min_frequency}"
                )
            if 0 < self.ohe_min_frequency < 1:
                pass  # Valid: proportion of samples
            elif self.ohe_min_frequency >= 1:
                self.ohe_min_frequency = int(self.ohe_min_frequency)
        else:
            self.ohe_min_frequency = None

    def _validate_cross_field_constraints(self) -> None:
        """Validate cross-field constraints after all sections are parsed."""
        if not 1 <= self.model_compression <= 9:
            raise ValueError(
                f"Model compression must be between 1 and 9, got {self.model_compression}"
            )
        if self.threshold_method not in ["auto", "oof", "cv"]:
            raise ValueError(
                f"Threshold method must be 'auto', 'oof', or 'cv', got '{self.threshold_method}'"
            )

        valid_objectives = ["youden", "f1", "f2", "cost_sensitive"]
        if self.threshold_objective not in valid_objectives:
            raise ValueError(
                f"Threshold objective must be one of {valid_objectives}, "
                f"got '{self.threshold_objective}'"
            )
        if self.vme_cost <= 0:
            raise ValueError(f"vme_cost must be positive, got {self.vme_cost}")
        if self.me_cost <= 0:
            raise ValueError(f"me_cost must be positive, got {self.me_cost}")

        valid_calibration_methods = ["sigmoid", "isotonic"]
        if self.probability_calibration_method not in valid_calibration_methods:
            raise ValueError(
                f"Probability calibration method must be one of {valid_calibration_methods}, "
                f"got '{self.probability_calibration_method}'"
            )
        if self.probability_calibration_cv < 2:
            raise ValueError(
                f"probability_calibration_cv must be >= 2, got {self.probability_calibration_cv}"
            )
        if self.outer_cv_repeats < 1:
            raise ValueError(f"outer_cv_repeats must be >= 1, got {self.outer_cv_repeats}")
        if not 0 < self.uncertainty_margin < 0.5:
            raise ValueError(
                f"Uncertainty margin must be between 0 and 0.5, got {self.uncertainty_margin}"
            )

        # Imputation validation
        valid_methods = ["none", "simple", "knn", "iterative"]
        if self.imputation_method not in valid_methods:
            raise ValueError(
                f"Imputation method must be one of {valid_methods}, got '{self.imputation_method}'"
            )
        valid_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.imputation_strategy not in valid_strategies:
            raise ValueError(
                f"Imputation strategy must be one of {valid_strategies}, "
                f"got '{self.imputation_strategy}'"
            )
        valid_estimators = ["bayesian_ridge", "random_forest"]
        if self.imputation_estimator not in valid_estimators:
            raise ValueError(
                f"Imputation estimator must be one of {valid_estimators}, "
                f"got '{self.imputation_estimator}'"
            )

        # Validation strategy constraints
        valid_val_strategies = ["cv", "temporal", "both"]
        if self.validation_strategy not in valid_val_strategies:
            raise ValueError(
                f"validation_strategy must be one of {valid_val_strategies}, "
                f"got '{self.validation_strategy}'"
            )
        if self.validation_strategy in ("temporal", "both"):
            if not self.temporal_split_column:
                raise ValueError(
                    "temporal_split_column is required when validation_strategy "
                    f"is '{self.validation_strategy}'"
                )
            has_date = self.temporal_split_date is not None
            has_ratio = self.temporal_split_ratio is not None
            if has_date and has_ratio:
                raise ValueError(
                    "Only one of temporal_split_date or temporal_split_ratio "
                    "can be specified, not both"
                )
            if not has_date and not has_ratio:
                raise ValueError(
                    "Either temporal_split_date or temporal_split_ratio "
                    "must be specified for temporal validation"
                )
            if (
                has_ratio
                and self.temporal_split_ratio is not None
                and not (0 < self.temporal_split_ratio < 1)
            ):
                raise ValueError(
                    f"temporal_split_ratio must be between 0 and 1 exclusive, "
                    f"got {self.temporal_split_ratio}"
                )
            if has_date:
                try:
                    pd.to_datetime(self.temporal_split_date)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"temporal_split_date '{self.temporal_split_date}' is not a valid date: {e}"
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
    targets: list[str]
    continuous_features: list[str]
    groups: Optional[np.ndarray]
    temporal_column_values: Optional[pd.Series]

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

        # Columns to drop from X (targets + metadata columns)
        cols_to_drop = list(config_handler.targets)

        # Extract groups if group_column is specified
        self.groups = None
        if config_handler.group_column:
            if config_handler.group_column not in self.data.columns:
                raise ValueError(
                    f"Group column '{config_handler.group_column}' not found in data. "
                    f"Available columns: {list(self.data.columns)}"
                )
            self.groups = self.data[config_handler.group_column].values
            cols_to_drop.append(config_handler.group_column)

        # Extract temporal column if specified
        self.temporal_column_values = None
        if config_handler.temporal_split_column:
            if config_handler.temporal_split_column not in self.data.columns:
                raise ValueError(
                    f"Temporal split column '{config_handler.temporal_split_column}' "
                    f"not found in data. Available columns: {list(self.data.columns)}"
                )
            try:
                self.temporal_column_values = pd.to_datetime(
                    self.data[config_handler.temporal_split_column]
                )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Could not parse temporal split column "
                    f"'{config_handler.temporal_split_column}' as dates: {e}"
                )
            cols_to_drop.append(config_handler.temporal_split_column)

        self.X = self.data.drop(cols_to_drop, axis=1)
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
        ValueError
            If validation fails
        """
        # Check no missing values (only if imputation is disabled)
        if imputation_method == "none" and data.isnull().values.any():
            raise ValueError(
                "Dataset contains missing values. "
                "Enable imputation in config or remove missing values."
            )

        # Check targets in data
        if not set(targets).issubset(data.columns):
            missing = set(targets) - set(data.columns)
            raise ValueError(f"Target columns not found in dataset: {missing}")
