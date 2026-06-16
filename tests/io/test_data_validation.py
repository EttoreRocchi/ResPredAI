"""Data validation tests."""

from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from respredai.io.config import ConfigHandler, DataSetter


def _write(tmp_path, df, targets="y", continuous="age, bmi", group=None, imputation="none"):
    """Write a data CSV + config file and return a loaded ConfigHandler."""
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)
    group_line = f"group_column = {group}" if group else ""
    config_text = dedent(f"""
    [Data]
    data_path = {data_path}
    targets = {targets}
    continuous_features = {continuous}

    [Metadata]
    {group_line}

    [Pipeline]
    models = LR
    outer_folds = 2
    inner_folds = 2

    [Reproducibility]
    seed = 42

    [Log]
    verbosity = 0
    log_basename = test.log

    [Resources]
    n_jobs = 1

    [Output]
    out_folder = {tmp_path / "out"}

    [ModelSaving]
    enable = false

    [Imputation]
    method = {imputation}
    strategy = mean
    """).strip()
    config_path = tmp_path / "cfg.ini"
    config_path.write_text(config_text)
    return ConfigHandler(str(config_path))


def _base_df(n=40, seed=0):
    rng = np.random.RandomState(seed)
    df = pd.DataFrame(
        {
            "age": rng.uniform(20, 80, n),
            "bmi": rng.uniform(18, 35, n),
            "sex": rng.choice(["M", "F"], n),
            "group": rng.choice([1, 2, 3, 4, 5], n),
            "y": np.tile([0, 1], n // 2),
        }
    )
    return df


class TestDataValidation:
    def test_valid_binary_loads(self, tmp_path):
        config = _write(tmp_path, _base_df())
        ds = DataSetter(config)
        assert ds.X.shape[0] == 40

    def test_non_binary_target_raises(self, tmp_path):
        df = _base_df()
        df["y"] = np.tile([1, 2], len(df) // 2)
        config = _write(tmp_path, df)
        with pytest.raises(ValueError, match="binary"):
            DataSetter(config)

    def test_single_class_target_raises(self, tmp_path):
        df = _base_df()
        df["y"] = 1
        config = _write(tmp_path, df)
        with pytest.raises(ValueError, match="one class"):
            DataSetter(config)

    def test_target_nan_raises(self, tmp_path):
        df = _base_df()
        df["y"] = df["y"].astype(float)
        df.loc[0, "y"] = np.nan
        config = _write(tmp_path, df)
        with pytest.raises(ValueError, match="missing values"):
            DataSetter(config)

    def test_missing_continuous_feature_raises(self, tmp_path):
        config = _write(tmp_path, _base_df(), continuous="age, nonexistent_feat")
        with pytest.raises(ValueError, match="continuous_features"):
            DataSetter(config)

    def test_group_nan_raises(self, tmp_path):
        # Use imputation != none so the generic missing-value check is skipped and
        # the group-specific NaN check is what fires.
        df = _base_df()
        df["group"] = df["group"].astype(float)
        df.loc[0, "group"] = np.nan
        config = _write(tmp_path, df, group="group", imputation="simple")
        with pytest.raises(ValueError, match="[Gg]roup"):
            DataSetter(config)


class TestBinaryTargetValidationModes:
    """Training requires both classes; evaluation allows a single-class cohort."""

    def test_eval_mode_allows_single_class(self):
        df = pd.DataFrame({"y": [0, 0, 0, 0]})
        # Should not raise: a new-cohort ground truth may be single-class.
        DataSetter._validate_binary_targets(df, ["y"], require_both_classes=False)

    def test_eval_mode_still_rejects_non_binary(self):
        df = pd.DataFrame({"y": [0, 1, 2]})
        with pytest.raises(ValueError, match="binary"):
            DataSetter._validate_binary_targets(df, ["y"], require_both_classes=False)

    def test_eval_mode_still_rejects_nan(self):
        df = pd.DataFrame({"y": [0.0, 1.0, np.nan]})
        with pytest.raises(ValueError, match="missing values"):
            DataSetter._validate_binary_targets(df, ["y"], require_both_classes=False)

    def test_training_mode_rejects_single_class(self):
        df = pd.DataFrame({"y": [1, 1, 1]})
        with pytest.raises(ValueError, match="one class"):
            DataSetter._validate_binary_targets(df, ["y"])
