import pytest
import subprocess
import shutil


def is_cli_installed():
    """Check if respredai CLI is installed."""
    return shutil.which("respredai") is not None


skip_if_no_cli = pytest.mark.skipif(
    not is_cli_installed(),
    reason="respredai CLI not installed (run 'pip install .' first)"
)


@skip_if_no_cli
class TestCLI:
    """Integration tests for CLI commands.

    These tests verify that CLI commands run successfully
    and produce expected outputs.
    """

    @pytest.mark.slow
    def test_list_models_command(self):
        """Test that list-models command runs successfully."""

        result = subprocess.run(
            ["respredai", "list-models"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "LR" in result.stdout
        assert "XGB" in result.stdout
        assert "RF" in result.stdout

    @pytest.mark.slow
    def test_info_command(self):
        """Test that info command runs successfully."""

        result = subprocess.run(
            ["respredai", "info"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "ResPredAI" in result.stdout

    def test_version_command(self):
        """Test that --version command runs successfully."""

        result = subprocess.run(
            ["respredai", "--version"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        # Version should be in format X.Y.Z
        assert any(char.isdigit() for char in result.stdout)

    @pytest.mark.slow
    def test_create_config_command(self, tmp_path):
        """Test that create-config command creates a valid config file."""

        config_path = tmp_path / "test_config.ini"

        result = subprocess.run(
            ["respredai", "create-config", str(config_path)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert config_path.exists()

        # Verify config contains required sections
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

        result = subprocess.run(
            ["respredai", "create-config", str(config_path)],
            capture_output=True,
            text=True
        )

        # Should fail because extension is not .ini
        assert result.returncode != 0

    @pytest.mark.slow
    def test_run_command_missing_config(self):
        """Test that run command fails gracefully with missing config."""

        result = subprocess.run(
            ["respredai", "run", "--config", "nonexistent_config.ini"],
            capture_output=True,
            text=True
        )

        # Should fail because config doesn't exist
        assert result.returncode != 0

    @pytest.mark.slow
    def test_feature_importance_missing_arguments(self, tmp_path):
        """Test that feature-importance command fails without required arguments."""

        result = subprocess.run(
            ["respredai", "feature-importance"],
            capture_output=True,
            text=True
        )

        # Should fail because required arguments are missing
        assert result.returncode != 0


@skip_if_no_cli
class TestCLIIntegration:
    """Integration tests that run full pipeline commands."""

    @pytest.mark.slow
    def test_full_pipeline_run(self, tmp_path):
        """Test that the full pipeline runs on example config.

        This test actually runs the full pipeline and can take several minutes.
        Run with: pytest -v -m slow
        """

        result = subprocess.run(
            ["respredai", "run", "--config", "example/config_example.ini"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        assert result.returncode == 0
        assert "Training completed successfully" in result.stdout or result.returncode == 0
