from textwrap import dedent

import pandas as pd
import pytest

from respredai.io.config import ConfigHandler, DataSetter


class TestDataSetter:
    """Unit tests for DataSetter."""

    def _make_config(self, tmp_path, targets="y", group="group"):
        """Helper to create a temporary config file."""

        config_text = dedent(f"""
        [Data]
        data_path = {{data}}
        targets = {targets}
        continuous_features = age, bmi
        group_column = {group}

        [Pipeline]
        models = lr, rf
        outer_folds = 3
        inner_folds = 2
        calibrate_threshold = True
        threshold_method = auto

        [Reproducibility]
        seed = 42

        [Log]
        verbosity = 0
        log_basename = test.log

        [Resources]
        n_jobs = 1

        [Output]
        out_folder = {{out}}

        [ModelSaving]
        enable = True
        compression = 3
        """).strip()

        data_file = tmp_path / "data.csv"
        out_folder = tmp_path / "output"
        out_folder.mkdir()

        return (
            config_text.format(data=data_file, out=out_folder),
            data_file,
            out_folder,
        )

    def test_data_setter_loads_and_splits(self, tmp_path):
        """Test correct loading, validation, and feature splitting."""

        config_text, data_path, _ = self._make_config(tmp_path)
        config_path = tmp_path / "config.ini"
        config_path.write_text(config_text)
        config = ConfigHandler(str(config_path))

        df = pd.DataFrame(
            {
                "age": [30, 40, 50],
                "bmi": [20.1, 25.3, 30.0],
                "group": [1, 1, 2],
                "y": [0, 1, 0],
            }
        )
        df.to_csv(data_path, index=False)

        ds = DataSetter(config)

        assert isinstance(ds.X, pd.DataFrame)
        assert list(ds.Y.columns) == ["y"]
        assert ds.groups is not None
        assert "group" not in ds.X.columns
        assert "y" not in ds.X.columns

    def test_missing_group_column(self, tmp_path):
        """Test that missing group column raises an error."""

        config_text, data_path, _ = self._make_config(tmp_path)
        config_path = tmp_path / "config.ini"
        config_path.write_text(config_text)
        config = ConfigHandler(str(config_path))

        df = pd.DataFrame(
            {
                "age": [10, 20],
                "bmi": [18.2, 27.5],
                "y": [0, 1],
            }
        )
        df.to_csv(data_path, index=False)

        with pytest.raises(ValueError):
            DataSetter(config)

    def test_data_setter_missing_values(self, tmp_path):
        """Test that missing values raise an assertion error."""

        config_text, data_path, _ = self._make_config(tmp_path)
        config_path = tmp_path / "config.ini"
        config_path.write_text(config_text)
        config = ConfigHandler(str(config_path))

        df = pd.DataFrame(
            {
                "age": [10, None],
                "bmi": [18.2, 27.5],
                "group": [1, 1],
                "y": [0, 1],
            }
        )
        df.to_csv(data_path, index=False)

        with pytest.raises(AssertionError):
            DataSetter(config)

    def test_multiple_targets(self, tmp_path):
        """Test that DataSetter supports multiple target columns."""

        config_text, data_path, _ = self._make_config(tmp_path, targets="y1, y2")
        config_path = tmp_path / "config.ini"
        config_path.write_text(config_text)
        config = ConfigHandler(str(config_path))

        df = pd.DataFrame(
            {
                "age": [30, 40],
                "bmi": [20.1, 25.3],
                "group": [1, 1],
                "y1": [0, 1],
                "y2": [1, 0],
            }
        )
        df.to_csv(data_path, index=False)

        ds = DataSetter(config)

        assert list(ds.Y.columns) == ["y1", "y2"]
        assert all(col not in ds.X.columns for col in ["y1", "y2"])

    def test_nonexistent_data_path(self, tmp_path):
        """Test that missing CSV file raises FileNotFoundError."""

        config_text, data_path, _ = self._make_config(tmp_path)
        config_path = tmp_path / "config.ini"
        config_path.write_text(config_text)

        if data_path.exists():
            data_path.unlink()

        config = ConfigHandler(str(config_path))

        with pytest.raises(FileNotFoundError):
            DataSetter(config)
