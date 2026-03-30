"""Utility classes for configuration and data handling."""

import logging
import os
import warnings
from collections.abc import Iterable
from configparser import ConfigParser
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from respredai.core.constants import (
    CALIBRATION_METHODS,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_N_BOOTSTRAP,
    IMPUTATION_ESTIMATORS,
    IMPUTATION_METHODS,
    IMPUTATION_STRATEGIES,
    THRESHOLD_METHODS,
    THRESHOLD_OBJECTIVES,
    VALIDATION_STRATEGIES,
)

# ---------------------------------------------------------------------------
# Domain-specific configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    """Configuration for data paths and column definitions."""

    data_path: str = ""
    targets: list[str] = field(default_factory=list)
    continuous_features: list[str] = field(default_factory=list)


@dataclass
class PipelineConfig:
    """Configuration for the ML pipeline (models, CV, thresholds, calibration)."""

    models: list[str] = field(default_factory=list)
    outer_folds: int = 5
    inner_folds: int = 3
    outer_cv_repeats: int = 1
    calibrate_threshold: bool = False
    threshold_method: str = "auto"
    threshold_objective: str = "youden"
    vme_cost: float = 1.0
    me_cost: float = 1.0
    calibrate_probabilities: bool = False
    probability_calibration_method: str = "sigmoid"
    probability_calibration_cv: int = 5
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP
    compute_feature_direction: bool = False


@dataclass
class ImputationConfig:
    """Configuration for missing data imputation."""

    method: str = "none"
    strategy: str = "mean"
    n_neighbors: int = 5
    estimator: str = "bayesian_ridge"


@dataclass
class ValidationConfig:
    """Configuration for validation strategy (CV, temporal, or both)."""

    strategy: str = "cv"
    temporal_split_date: Optional[str] = None
    temporal_split_ratio: Optional[float] = None


@dataclass
class PreprocessingConfig:
    """Configuration for feature preprocessing."""

    ohe_min_frequency: Optional[float] = None


@dataclass
class OutputConfig:
    """Configuration for output paths and model saving."""

    out_folder: str = "./output/"
    save_models_enable: bool = False
    model_compression: int = 3


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility, logging, and resources."""

    seed: int = 42
    verbosity: int = 1
    log_basename: str = "respredai.log"
    n_jobs: int = -1
    uncertainty_margin: float = 0.1


@dataclass
class MetadataConfig:
    """Configuration for metadata columns (grouping, temporal, subgroup)."""

    group_column: Optional[str] = None
    temporal_column: Optional[str] = None
    subgroup_columns: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# ConfigHandler
# ---------------------------------------------------------------------------


class ConfigHandler:
    """Handle configuration file parsing and validation.

    Stores configuration in domain-specific dataclass instances accessible
    as attributes: ``data_cfg``, ``pipeline``, ``imputation``, ``validation``,
    ``preprocessing``, ``output``, ``reproducibility_cfg``, and ``metadata``.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize configuration handler.

        Parameters
        ----------
        config_path : str
            Path to the configuration file (.ini format)
        """
        self.config_path = config_path
        self.logger: Optional[logging.Logger] = None
        self._setup_config()

    # ------------------------------------------------------------------
    # Setup & parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _get_optional_str(
        config: ConfigParser, section: str, key: str, fallback=None
    ) -> Optional[str]:
        """Get a string value, normalizing empty/whitespace-only to ``None``."""
        val = config.get(section, key, fallback=fallback)
        if val is not None and isinstance(val, str) and not val.strip():
            return None
        return val.strip() if isinstance(val, str) and val else val

    def initialize_logger(self) -> None:
        """Set up the logger using the current out_folder. Call after any CLI overrides."""
        if self.reproducibility_cfg.verbosity and self.logger is None:
            self.logger = self._setup_logger(
                os.path.join(self.output.out_folder, self.reproducibility_cfg.log_basename)
            )

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
        self._parse_metadata_section(config)
        self._parse_validation_section(config)
        self._parse_preprocessing_section(config)
        self._validate_cross_field_constraints()

    def _parse_data_section(self, config: ConfigParser) -> None:
        """Parse [Data] section."""
        self.data_cfg = DataConfig(
            data_path=config.get("Data", "data_path"),
            targets=[t.strip() for t in config.get("Data", "targets").split(",")],
            continuous_features=[
                f.strip() for f in config.get("Data", "continuous_features").split(",")
            ],
        )

    def _parse_pipeline_section(self, config: ConfigParser) -> None:
        """Parse [Pipeline] section including probability calibration."""
        self.pipeline = PipelineConfig(
            models=[m.strip() for m in config.get("Pipeline", "models").split(",")],
            outer_folds=config.getint("Pipeline", "outer_folds"),
            inner_folds=config.getint("Pipeline", "inner_folds"),
            outer_cv_repeats=config.getint("Pipeline", "outer_cv_repeats", fallback=1),
            calibrate_threshold=config.getboolean(
                "Pipeline", "calibrate_threshold", fallback=False
            ),
            threshold_method=config.get("Pipeline", "threshold_method", fallback="auto").lower(),
            threshold_objective=config.get(
                "Pipeline", "threshold_objective", fallback="youden"
            ).lower(),
            vme_cost=config.getfloat("Pipeline", "vme_cost", fallback=1.0),
            me_cost=config.getfloat("Pipeline", "me_cost", fallback=1.0),
            calibrate_probabilities=config.getboolean(
                "Pipeline", "calibrate_probabilities", fallback=False
            ),
            probability_calibration_method=config.get(
                "Pipeline", "probability_calibration_method", fallback="sigmoid"
            ).lower(),
            probability_calibration_cv=config.getint(
                "Pipeline", "probability_calibration_cv", fallback=5
            ),
            confidence_level=config.getfloat(
                "Pipeline", "confidence_level", fallback=DEFAULT_CONFIDENCE_LEVEL
            ),
            n_bootstrap=config.getint("Pipeline", "n_bootstrap", fallback=DEFAULT_N_BOOTSTRAP),
            compute_feature_direction=config.getboolean(
                "Pipeline", "compute_feature_direction", fallback=False
            ),
        )

    def _parse_misc_sections(self, config: ConfigParser) -> None:
        """Parse [Uncertainty], [Reproducibility], [Log], [Resources], [Output], [ModelSaving]."""
        self.reproducibility_cfg = ReproducibilityConfig(
            seed=config.getint("Reproducibility", "seed"),
            verbosity=config.getint("Log", "verbosity"),
            log_basename=config.get("Log", "log_basename"),
            n_jobs=config.getint("Resources", "n_jobs"),
            uncertainty_margin=config.getfloat("Uncertainty", "margin", fallback=0.1),
        )
        self.output = OutputConfig(
            out_folder=config.get("Output", "out_folder"),
            save_models_enable=config.getboolean("ModelSaving", "enable", fallback=False),
            model_compression=config.getint("ModelSaving", "compression", fallback=3),
        )

    def _parse_imputation_section(self, config: ConfigParser) -> None:
        """Parse [Imputation] section."""
        self.imputation = ImputationConfig(
            method=config.get("Imputation", "method", fallback="none").lower(),
            strategy=config.get("Imputation", "strategy", fallback="mean").lower(),
            n_neighbors=config.getint("Imputation", "n_neighbors", fallback=5),
            estimator=config.get("Imputation", "estimator", fallback="bayesian_ridge").lower(),
        )

    def _parse_metadata_section(self, config: ConfigParser) -> None:
        """Parse [Metadata] section, with fallback to legacy locations."""
        if config.has_section("Metadata"):
            group_col = self._get_optional_str(config, "Metadata", "group_column")
            temporal_col = self._get_optional_str(config, "Metadata", "temporal_column")
            subgroup_raw = config.get("Metadata", "subgroup_columns", fallback="")
            subgroup_cols = [s.strip() for s in subgroup_raw.split(",") if s.strip()]
        else:
            # Legacy fallback: read from old locations
            group_col = self._get_optional_str(config, "Data", "group_column")
            temporal_col = self._get_optional_str(config, "Validation", "temporal_split_column")
            subgroup_cols = []
            if group_col or temporal_col:
                warnings.warn(
                    "group_column in [Data] and temporal_split_column in [Validation] are "
                    "deprecated. Move them to a [Metadata] section as group_column and "
                    "temporal_column respectively.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        self.metadata = MetadataConfig(
            group_column=group_col,
            temporal_column=temporal_col,
            subgroup_columns=subgroup_cols,
        )

    def _parse_validation_section(self, config: ConfigParser) -> None:
        """Parse [Validation] section."""
        ratio_str = self._get_optional_str(config, "Validation", "temporal_split_ratio")
        self.validation = ValidationConfig(
            strategy=config.get("Validation", "validation_strategy", fallback="cv").lower(),
            temporal_split_date=self._get_optional_str(config, "Validation", "temporal_split_date"),
            temporal_split_ratio=float(ratio_str) if ratio_str is not None else None,
        )

    def _parse_preprocessing_section(self, config: ConfigParser) -> None:
        """Parse [Preprocessing] section."""
        ohe_min_freq_str = self._get_optional_str(config, "Preprocessing", "ohe_min_frequency")
        if ohe_min_freq_str is not None:
            ohe_val = float(ohe_min_freq_str)
            if ohe_val < 0:
                raise ValueError(f"ohe_min_frequency must be non-negative, got {ohe_val}")
            if ohe_val == 0:
                self.preprocessing = PreprocessingConfig(ohe_min_frequency=None)
                return
            if ohe_val >= 1:
                ohe_val = int(ohe_val)
            self.preprocessing = PreprocessingConfig(ohe_min_frequency=ohe_val)
        else:
            self.preprocessing = PreprocessingConfig(ohe_min_frequency=None)

    def _validate_cross_field_constraints(self) -> None:
        """Validate cross-field constraints after all sections are parsed."""
        if not 1 <= self.output.model_compression <= 9:
            raise ValueError(
                f"Model compression must be between 1 and 9, got {self.output.model_compression}"
            )
        if self.pipeline.threshold_method not in THRESHOLD_METHODS:
            raise ValueError(
                f"Threshold method must be one of {THRESHOLD_METHODS}, "
                f"got '{self.pipeline.threshold_method}'"
            )

        if self.pipeline.threshold_objective not in THRESHOLD_OBJECTIVES:
            raise ValueError(
                f"Threshold objective must be one of {THRESHOLD_OBJECTIVES}, "
                f"got '{self.pipeline.threshold_objective}'"
            )
        if self.pipeline.vme_cost <= 0:
            raise ValueError(f"vme_cost must be positive, got {self.pipeline.vme_cost}")
        if self.pipeline.me_cost <= 0:
            raise ValueError(f"me_cost must be positive, got {self.pipeline.me_cost}")

        if self.pipeline.probability_calibration_method not in CALIBRATION_METHODS:
            raise ValueError(
                f"Probability calibration method must be one of {CALIBRATION_METHODS}, "
                f"got '{self.pipeline.probability_calibration_method}'"
            )
        if self.pipeline.probability_calibration_cv < 2:
            raise ValueError(
                f"probability_calibration_cv must be >= 2, "
                f"got {self.pipeline.probability_calibration_cv}"
            )
        if self.pipeline.outer_cv_repeats < 1:
            raise ValueError(f"outer_cv_repeats must be >= 1, got {self.pipeline.outer_cv_repeats}")
        if not (0.5 < self.pipeline.confidence_level < 1.0):
            raise ValueError(
                f"confidence_level must be between 0.5 and 1.0, "
                f"got {self.pipeline.confidence_level}"
            )
        if self.pipeline.n_bootstrap < 100:
            raise ValueError(f"n_bootstrap must be >= 100, got {self.pipeline.n_bootstrap}")
        if not 0 < self.reproducibility_cfg.uncertainty_margin < 0.5:
            raise ValueError(
                f"Uncertainty margin must be between 0 and 0.5, "
                f"got {self.reproducibility_cfg.uncertainty_margin}"
            )

        # Imputation validation
        if self.imputation.method not in IMPUTATION_METHODS:
            raise ValueError(
                f"Imputation method must be one of {IMPUTATION_METHODS}, "
                f"got '{self.imputation.method}'"
            )
        if self.imputation.strategy not in IMPUTATION_STRATEGIES:
            raise ValueError(
                f"Imputation strategy must be one of {IMPUTATION_STRATEGIES}, "
                f"got '{self.imputation.strategy}'"
            )
        if self.imputation.estimator not in IMPUTATION_ESTIMATORS:
            raise ValueError(
                f"Imputation estimator must be one of {IMPUTATION_ESTIMATORS}, "
                f"got '{self.imputation.estimator}'"
            )

        # Validation strategy constraints
        if self.validation.strategy not in VALIDATION_STRATEGIES:
            raise ValueError(
                f"validation_strategy must be one of {VALIDATION_STRATEGIES}, "
                f"got '{self.validation.strategy}'"
            )
        if self.validation.strategy in ("temporal", "both"):
            if not self.metadata.temporal_column:
                raise ValueError(
                    "temporal_column is required when validation_strategy "
                    f"is '{self.validation.strategy}'"
                )
            has_date = self.validation.temporal_split_date is not None
            has_ratio = self.validation.temporal_split_ratio is not None
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
                and self.validation.temporal_split_ratio is not None
                and not (0 < self.validation.temporal_split_ratio < 1)
            ):
                raise ValueError(
                    f"temporal_split_ratio must be between 0 and 1 exclusive, "
                    f"got {self.validation.temporal_split_ratio}"
                )
            if has_date:
                try:
                    pd.to_datetime(self.validation.temporal_split_date)
                except (ValueError, TypeError) as e:
                    raise ValueError(
                        f"temporal_split_date '{self.validation.temporal_split_date}' "
                        f"is not a valid date: {e}"
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
    """Handle data loading and validation.

    Attributes
    ----------
    data : pd.DataFrame
        Raw loaded dataframe (before dropping metadata columns).
    X : pd.DataFrame
        Feature matrix (targets and metadata columns removed).
    Y : pd.DataFrame
        Target columns.
    targets : list[str]
        Target column names.
    continuous_features : list[str]
        Names of continuous features (used for scaling).
    groups : np.ndarray or None
        Group labels for group-aware cross-validation.
    temporal_column_values : pd.Series or None
        Parsed datetime values for temporal validation splitting.
    subgroup_data : dict[str, pd.Series]
        Maps subgroup column names to their values for subgroup analysis.
    """

    data: pd.DataFrame
    X: pd.DataFrame
    Y: pd.DataFrame
    targets: list[str]
    continuous_features: list[str]
    groups: Optional[np.ndarray]
    temporal_column_values: Optional[pd.Series]
    subgroup_data: dict[str, pd.Series]

    def __init__(self, config_handler: ConfigHandler) -> None:
        """
        Initialize data setter.

        Parameters
        ----------
        config_handler : ConfigHandler
            Configuration handler with data paths and parameters
        """
        self.data = self._read_data(config_handler.data_cfg.data_path)
        self._validate_data(
            self.data, config_handler.data_cfg.targets, config_handler.imputation.method
        )

        # Columns to drop from X (targets + metadata columns)
        cols_to_drop = list(config_handler.data_cfg.targets)

        # Extract groups if group_column is specified
        self.groups = None
        if config_handler.metadata.group_column:
            if config_handler.metadata.group_column not in self.data.columns:
                raise ValueError(
                    f"Group column '{config_handler.metadata.group_column}' not found in data. "
                    f"Available columns: {list(self.data.columns)}"
                )
            self.groups = self.data[config_handler.metadata.group_column].values
            cols_to_drop.append(config_handler.metadata.group_column)

        # Extract temporal column if specified
        self.temporal_column_values = None
        if config_handler.metadata.temporal_column:
            if config_handler.metadata.temporal_column not in self.data.columns:
                raise ValueError(
                    f"Temporal column '{config_handler.metadata.temporal_column}' "
                    f"not found in data. Available columns: {list(self.data.columns)}"
                )
            try:
                self.temporal_column_values = pd.to_datetime(
                    self.data[config_handler.metadata.temporal_column]
                )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Could not parse temporal column "
                    f"'{config_handler.metadata.temporal_column}' as dates: {e}"
                )
            cols_to_drop.append(config_handler.metadata.temporal_column)

        # Extract subgroup columns if specified
        self.subgroup_data = {}
        for col in config_handler.metadata.subgroup_columns:
            if col not in self.data.columns:
                raise ValueError(
                    f"Subgroup column '{col}' not found in data. "
                    f"Available columns: {list(self.data.columns)}"
                )
            self.subgroup_data[col] = self.data[col]
            if col not in cols_to_drop:
                cols_to_drop.append(col)

        self.X = self.data.drop(cols_to_drop, axis=1)
        self.Y = self.data[config_handler.data_cfg.targets]
        self.targets = config_handler.data_cfg.targets
        self.continuous_features = config_handler.data_cfg.continuous_features

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
