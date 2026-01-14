import shutil
import subprocess
import sys

import pytest


def get_cli_command():
    """Get the CLI command (prefer respredai, fallback to python -m)."""
    if shutil.which("respredai") is not None:
        return ["respredai"]
    return [sys.executable, "-m", "respredai.cli"]


def is_cli_available():
    """Check if respredai CLI is available."""
    try:
        cmd = get_cli_command()
        result = subprocess.run(cmd + ["--version"], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except Exception:
        return False


skip_if_no_cli = pytest.mark.skipif(not is_cli_available(), reason="respredai CLI not available")


def run_cli(*args, **kwargs):
    """Run CLI command with appropriate method."""
    cmd = get_cli_command() + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


@skip_if_no_cli
class TestCLI:
    """Integration tests for CLI commands."""

    @pytest.mark.slow
    def test_list_models_command(self):
        """Test that list-models command runs successfully."""
        result = run_cli("list-models")

        assert result.returncode == 0
        assert "LR" in result.stdout
        assert "XGB" in result.stdout
        assert "RF" in result.stdout

    @pytest.mark.slow
    def test_info_command(self):
        """Test that info command runs successfully."""
        result = run_cli("info")

        assert result.returncode == 0
        assert "ResPredAI" in result.stdout

    def test_version_command(self):
        """Test that --version command runs successfully."""
        result = run_cli("--version")

        assert result.returncode == 0
        assert any(char.isdigit() for char in result.stdout)

    @pytest.mark.slow
    def test_create_config_command(self, tmp_path):
        """Test that create-config command creates a valid config file."""
        config_path = tmp_path / "test_config.ini"

        result = run_cli("create-config", str(config_path))

        assert result.returncode == 0
        assert config_path.exists()

        config_text = config_path.read_text()
        assert "[Data]" in config_text
        assert "[Pipeline]" in config_text
        assert "[Reproducibility]" in config_text
        assert "threshold_method" in config_text
        assert "calibrate_threshold" in config_text

    @pytest.mark.slow
    def test_create_config_invalid_extension(self, tmp_path):
        """Test that create-config fails with invalid file extension."""
        config_path = tmp_path / "test_config.txt"

        result = run_cli("create-config", str(config_path))

        assert result.returncode != 0

    @pytest.mark.slow
    def test_run_command_missing_config(self):
        """Test that run command fails gracefully with missing config."""
        result = run_cli("run", "--config", "nonexistent_config.ini")

        assert result.returncode != 0

    @pytest.mark.slow
    def test_feature_importance_missing_arguments(self):
        """Test that feature-importance command fails without required arguments."""
        result = run_cli("feature-importance")

        assert result.returncode != 0

    @pytest.mark.slow
    def test_validate_config_command(self, tmp_path):
        """Test that validate-config command works with valid config."""
        config_path = tmp_path / "test_config.ini"

        # First create a config
        run_cli("create-config", str(config_path))

        # Then validate it
        result = run_cli("validate-config", str(config_path))

        assert result.returncode == 0
        assert "valid" in result.stdout.lower()

    @pytest.mark.slow
    def test_validate_config_missing_file(self):
        """Test that validate-config fails with missing config file."""
        result = run_cli("validate-config", "nonexistent.ini")

        assert result.returncode != 0

    @pytest.mark.slow
    def test_train_command_missing_config(self):
        """Test that train command fails gracefully with missing config."""
        result = run_cli("train", "--config", "nonexistent_config.ini")

        assert result.returncode != 0

    @pytest.mark.slow
    def test_evaluate_command_missing_args(self):
        """Test that evaluate command fails without required arguments."""
        result = run_cli("evaluate")

        assert result.returncode != 0

    @pytest.mark.slow
    def test_evaluate_command_missing_models_dir(self, tmp_path):
        """Test that evaluate command fails with nonexistent models directory."""
        result = run_cli(
            "evaluate",
            "--models-dir",
            str(tmp_path / "nonexistent"),
            "--data",
            "data.csv",
            "--output",
            str(tmp_path / "out"),
        )

        assert result.returncode != 0


@skip_if_no_cli
class TestCLIIntegration:
    """Integration tests that run full pipeline commands."""

    @pytest.mark.slow
    def test_full_pipeline_run(self):
        """Test that the full pipeline runs on example config.

        This test actually runs the full pipeline and can take several minutes.
        Run with: pytest -v -m slow
        """
        result = run_cli("run", "--config", "example/config_example.ini", timeout=300)

        assert result.returncode == 0

    @pytest.mark.slow
    def test_train_command_with_example(self, tmp_path):
        """Test that train command runs successfully on example data.

        This test trains models on the full dataset.
        Run with: pytest -v -m slow
        """
        output_dir = tmp_path / "train_output"

        result = run_cli(
            "train",
            "--config",
            "example/config_example.ini",
            "--output",
            str(output_dir),
            timeout=300,
        )

        assert result.returncode == 0, f"Train failed: {result.stderr}"

        # Verify output files were created
        trained_models_dir = output_dir / "trained_models"
        assert trained_models_dir.exists(), "trained_models directory not created"
        assert (trained_models_dir / "training_metadata.json").exists(), "metadata not created"

        # Check that at least one model file was created
        model_files = list(trained_models_dir.glob("*.joblib"))
        assert len(model_files) > 0, "No model files created"

    @pytest.mark.slow
    def test_evaluate_command_with_example(self, tmp_path):
        """Test that evaluate command runs successfully with trained models.

        This test first trains models, then evaluates them on the same data.
        Run with: pytest -v -m slow
        """
        train_output_dir = tmp_path / "train_output"
        eval_output_dir = tmp_path / "eval_output"

        # First train models
        train_result = run_cli(
            "train",
            "--config",
            "example/config_example.ini",
            "--output",
            str(train_output_dir),
            timeout=300,
        )
        assert train_result.returncode == 0, f"Train failed: {train_result.stderr}"

        # Then evaluate on the same data (as ground truth)
        eval_result = run_cli(
            "evaluate",
            "--models-dir",
            str(train_output_dir / "trained_models"),
            "--data",
            "example/data_example.csv",
            "--output",
            str(eval_output_dir),
            timeout=120,
        )

        assert eval_result.returncode == 0, f"Evaluate failed: {eval_result.stderr}"

        # Verify output files were created
        assert eval_output_dir.exists(), "Output directory not created"
        assert (eval_output_dir / "evaluation_summary.csv").exists(), "Summary not created"

        # Check predictions directory
        predictions_dir = eval_output_dir / "predictions"
        assert predictions_dir.exists(), "predictions directory not created"
        prediction_files = list(predictions_dir.glob("*.csv"))
        assert len(prediction_files) > 0, "No prediction files created"
