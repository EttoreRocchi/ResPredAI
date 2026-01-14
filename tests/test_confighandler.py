from textwrap import dedent

import pytest

from respredai.io.config import ConfigHandler


class TestConfigHandler:
    """Unit tests for ConfigHandler."""

    def _make_config(self, tmp_path):
        """Helper to create a temporary config file."""

        config_text = dedent("""
        [Data]
        data_path = {data}
        targets = y
        continuous_features = age, bmi
        group_column = group

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
        out_folder = {out}

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

    def test_config_handler_parses_config(self, tmp_path):
        """Test that a valid config file is correctly parsed."""

        config_text, _, _ = self._make_config(tmp_path)
        config_path = tmp_path / "config.ini"
        config_path.write_text(config_text)

        config = ConfigHandler(str(config_path))

        assert config.targets == ["y"]
        assert config.continuous_features == ["age", "bmi"]
        assert config.models == ["lr", "rf"]
        assert config.outer_folds == 3
        assert config.seed == 42
        assert config.save_models_enable is True
        assert config.model_compression == 3

    def test_invalid_model_compression(self, tmp_path):
        """Test that invalid compression values raise an error."""

        bad_config = dedent("""
        [Data]
        data_path = foo.csv
        targets = y
        continuous_features = age
        group_column =

        [Pipeline]
        models = lr
        outer_folds = 3
        inner_folds = 2

        [Reproducibility]
        seed = 42

        [Log]
        verbosity = 0
        log_basename = test.log

        [Resources]
        n_jobs = 1

        [Output]
        out_folder = out

        [ModelSaving]
        enable = True
        compression = 99
        """)

        config_path = tmp_path / "bad.ini"
        config_path.write_text(bad_config)

        with pytest.raises(ValueError):
            ConfigHandler(str(config_path))

    def test_threshold_method_validation(self, tmp_path):
        """Test that invalid threshold_method values raise an error."""

        bad_config = dedent("""
        [Data]
        data_path = foo.csv
        targets = y
        continuous_features = age
        group_column =

        [Pipeline]
        models = lr
        outer_folds = 3
        inner_folds = 2
        calibrate_threshold = true
        threshold_method = invalid_method

        [Reproducibility]
        seed = 42

        [Log]
        verbosity = 0
        log_basename = test.log

        [Resources]
        n_jobs = 1

        [Output]
        out_folder = out

        [ModelSaving]
        enable = True
        compression = 3
        """)

        config_path = tmp_path / "bad_threshold.ini"
        config_path.write_text(bad_config)

        with pytest.raises(ValueError, match="Threshold method must be"):
            ConfigHandler(str(config_path))

    @pytest.mark.parametrize("threshold_method", ["auto", "oof", "cv"])
    def test_threshold_method_parsing(self, tmp_path, threshold_method):
        """Test that all valid threshold_method values are correctly parsed."""

        config_text, _, _ = self._make_config(tmp_path)
        config_text = config_text.replace(
            "threshold_method = auto", f"threshold_method = {threshold_method}"
        )
        config_path = tmp_path / "config.ini"
        config_path.write_text(config_text)

        config = ConfigHandler(str(config_path))

        assert config.threshold_method == threshold_method
        assert config.calibrate_threshold is True

    def test_model_parsing(self, tmp_path):
        """Test that model names are correctly parsed from config."""

        config_text, _, _ = self._make_config(tmp_path)
        config_path = tmp_path / "config.ini"
        config_path.write_text(config_text)

        config = ConfigHandler(str(config_path))

        assert "lr" in config.models
        assert "rf" in config.models
        assert len(config.models) == 2
