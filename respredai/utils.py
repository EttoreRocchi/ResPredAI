"""Utility classes for configuration and data handling."""

__author__ = "Ettore Rocchi"
__email__ = "ettore.rocchi3@unibo.it"

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
        
        # Section: Pipeline
        self.models = [m.strip() for m in config.get("Pipeline", "models").split(",")]
        self.outer_folds = config.getint("Pipeline", "outer_folds")
        self.inner_folds = config.getint("Pipeline", "inner_folds")
        
        # Section: Reproducibility
        self.seed = config.getint("Reproducibility", "seed")
        
        # Section: Log
        self.verbosity = config.getint("Log", "verbosity")
        self.log_basename = config.get("Log", "log_basename")
        
        # Section: Resources
        self.n_jobs = config.getint("Resources", "n_jobs")
        
        # Section: Output
        self.out_folder = config.get("Output", "out_folder")
    
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
        self.X, self.Y = (
            self.data.drop(config_handler.targets, axis=1),
            self.data[config_handler.targets]
        )
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