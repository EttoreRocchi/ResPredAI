"""Utility classes for configuration and data handling."""

import os
from configparser import ConfigParser
import logging
from typing import Iterable

import pandas as pd


class ConfigHandler:
    """Handle configuration file parsing and validation."""

    def __init__(self, config_path: str):
        """
        Initialize configuration handler.

        Parameters
        ----------
        config_path : str
            Path to the configuration file (.ini format)
        """
        self.config_path = config_path
        self._setup_config()
        if self.verbosity:
            self.logger = self._setup_logger(
                os.path.join(self.out_folder, self.log_basename)
            )

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
        self.calibrate_threshold = config.getboolean("Pipeline", "calibrate_threshold", fallback=False)
        self.threshold_method = config.get("Pipeline", "threshold_method", fallback="auto").lower()

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
            fmt="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
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

    def __init__(self, config_handler: ConfigHandler):
        """
        Initialize data setter.

        Parameters
        ----------
        config_handler : ConfigHandler
            Configuration handler with data paths and parameters
        """
        self.data = self._read_data(config_handler.data_path)
        self._validate_data(self.data, config_handler.targets)

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
            self.X = self.data.drop(
                config_handler.targets + [config_handler.group_column], axis=1
            )
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
    def _validate_data(data: pd.DataFrame, targets: Iterable) -> None:
        """
        Validate the loaded data.

        Parameters
        ----------
        data : pd.DataFrame
            The dataframe to validate
        targets : Iterable
            Target column names

        Raises
        ------
        AssertionError
            If validation fails
        """
        # Check no missing values
        assert not data.isnull().values.any(), "Dataset contains missing values"

        # Check targets in data
        assert set(targets).issubset(data.columns), "Target columns not found in dataset"