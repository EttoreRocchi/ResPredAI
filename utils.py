__author__ = "Ettore Rocchi"
__email__ = "ettore.rocchi3@unibo.it"

import os

from configparser import ConfigParser
import logging

import pandas as pd

from typing import Iterable

class ConfigHandler:

    def __init__(self, config_path: str):
        """Init class. """
        self.config_path = config_path
        self._setup_config()
        if self.verbosity:
            self.logger = self._setup_logger(
                os.path.join(self.out_folder, self.log_basename)
            )

    def _setup_config(self) -> None:
        config = ConfigParser()
        config.read(self.config_path)
        # Section: Data
        self.data_path = config.get("Data", "data_path")
        self.targets = config.get("Data", "targets").split(",")
        self.continuous_features = config.get("Data", "continuous_features").split(",")
        # Section: Pipeline
        self.models = config.get("Pipeline", "models").split(",")
        self.outer_folds = config.getint("Pipeline", "outer_folds")
        self.inner_folds = config.getint("Pipeline", "inner_folds")
        # Section: Reproducibility
        self.seed = config.getint("Reproducibility", "seed")
        # Section: Log
        self.verbosity = config.getint("Log", "verbosity")
        self.log_basename = config.get("Log", "log_basename")
        # Section: Resources
        self.n_jobs = config.getint("Resources", "n_jobs") #FIXME
        # Section: Output
        self.out_folder = config.get("Output", "out_folder")
    
    @staticmethod
    def _setup_logger(log_file: str) -> logging.Logger:
        """Set up the log file."""
        formatter = logging.Formatter(
            fmt = "%(asctime)s %(levelname)-8s %(message)s",
            datefmt = "%Y-%m-%d %H:%M:%S"
        )
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file, mode="w+")
        handler.setFormatter(formatter)

        logger = logging.getLogger("logger")
        logger.setLevel("INFO")
        logger.addHandler(handler)
        return logger

class DataSetter:

    def __init__(self, config_handler: ConfigHandler):
        """Init class. """
        self.data = self._read_data(config_handler.data_path)
        self._validate_data(self.data, config_handler.targets)
        self.X, self.Y = self.data.drop(config_handler.targets, axis=1), self.data[config_handler.targets]
        self.targets = config_handler.targets
        self.continuous_features = config_handler.continuous_features

    @staticmethod
    def _read_data(data_path) -> pd.DataFrame:
        """Read data."""  
        return pd.read_csv(data_path, sep=",", comment="#")
    
    @staticmethod
    def _validate_data(data: pd.DataFrame, targets: Iterable) -> None:
        """Validate data."""
        # Check no missing values
        assert not data.isnull().values.any(), "there are missing values"
        # Check targets in data
        assert set(targets).issubset(data.columns), "targets not found"

