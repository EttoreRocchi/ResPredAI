"""Integration tests for the full pipeline with synthetic data."""

import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

from respredai.core.pipeline import perform_evaluation, perform_pipeline, perform_training
from respredai.io.config import ConfigHandler, DataSetter

# Suppress expected sklearn warnings for small synthetic test data
pytestmark = pytest.mark.filterwarnings(
    "ignore::sklearn.exceptions.ConvergenceWarning",
    "ignore:.*max_iter was reached.*:UserWarning",
)


def create_synthetic_data(n_samples=100, seed=42):
    """Create synthetic data for testing.

    Creates a simple dataset with:
    - 2 continuous features
    - 2 categorical features
    - 1 binary target
    - 1 group column

    The target is somewhat predictable from the features.
    """
    np.random.seed(seed)

    # Continuous features
    age = np.random.uniform(20, 80, n_samples)
    bmi = np.random.uniform(18, 35, n_samples)

    # Categorical features
    sex = np.random.choice(["M", "F"], n_samples)
    ward = np.random.choice(["ICU", "General", "Emergency"], n_samples)

    # Group column (patient IDs with some repeated samples)
    n_patients = n_samples // 2
    patient_ids = np.random.choice(range(1, n_patients + 1), n_samples)

    # Target: make it somewhat predictable
    # Higher risk if: age > 60, BMI > 30, or in ICU
    risk_score = (
        (age > 60).astype(float) * 0.3
        + (bmi > 30).astype(float) * 0.2
        + (ward == "ICU").astype(float) * 0.3
        + np.random.uniform(0, 0.5, n_samples)
    )
    target = (risk_score > 0.5).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "bmi": bmi,
            "sex": sex,
            "ward": ward,
            "patient_id": patient_ids,
            "resistant": target,
        }
    )

    return df


def create_test_config(tmp_path, data_path, with_groups=True):
    """Create a test configuration file."""
    group_line = "patient_id" if with_groups else ""

    config_text = dedent(f"""
    [Data]
    data_path = {data_path}
    targets = resistant
    continuous_features = age, bmi
    group_column = {group_line}

    [Pipeline]
    models = LR
    outer_folds = 3
    inner_folds = 2
    calibrate_threshold = true
    threshold_method = oof

    [Reproducibility]
    seed = 42

    [Log]
    verbosity = 0
    log_basename = test.log

    [Resources]
    n_jobs = 1

    [Output]
    out_folder = {tmp_path / "output"}

    [ModelSaving]
    enable = true
    compression = 3

    [Imputation]
    method = none
    strategy = mean
    n_neighbors = 5
    estimator = bayesian_ridge
    """).strip()

    config_path = tmp_path / "test_config.ini"
    config_path.write_text(config_text)
    return config_path


class TestPerformPipelineIntegration:
    """Integration tests for perform_pipeline."""

    @pytest.mark.slow
    def test_pipeline_runs_with_synthetic_data(self, tmp_path):
        """Test that perform_pipeline completes successfully with synthetic data."""
        # Create synthetic data
        df = create_synthetic_data(n_samples=60, seed=42)
        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)

        # Create config
        config_path = create_test_config(tmp_path, data_path, with_groups=True)

        # Load config and data
        config = ConfigHandler(str(config_path))
        datasetter = DataSetter(config)

        # Run pipeline
        perform_pipeline(
            datasetter=datasetter,
            models=config.models,
            config_handler=config,
            progress_callback=None,
        )

        # Verify outputs
        output_dir = Path(config.out_folder)
        assert output_dir.exists()

        # Check metrics directory
        metrics_dir = output_dir / "metrics"
        assert metrics_dir.exists()
        assert (metrics_dir / "resistant").exists()
        assert (metrics_dir / "resistant" / "LR_metrics_detailed.csv").exists()

        # Check confusion matrices
        cm_dir = output_dir / "confusion_matrices"
        assert cm_dir.exists()

        # Check models directory
        models_dir = output_dir / "models"
        assert models_dir.exists()

    @pytest.mark.slow
    def test_pipeline_without_groups(self, tmp_path):
        """Test pipeline without group stratification."""
        df = create_synthetic_data(n_samples=60, seed=42)
        df = df.drop(columns=["patient_id"])
        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)

        config_path = create_test_config(tmp_path, data_path, with_groups=False)

        config = ConfigHandler(str(config_path))
        datasetter = DataSetter(config)

        perform_pipeline(
            datasetter=datasetter,
            models=config.models,
            config_handler=config,
            progress_callback=None,
        )

        output_dir = Path(config.out_folder)
        assert output_dir.exists()
        assert (output_dir / "metrics" / "resistant" / "LR_metrics_detailed.csv").exists()

    @pytest.mark.slow
    def test_pipeline_metrics_content(self, tmp_path):
        """Test that metrics file contains expected content."""
        df = create_synthetic_data(n_samples=60, seed=42)
        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)

        config_path = create_test_config(tmp_path, data_path)

        config = ConfigHandler(str(config_path))
        datasetter = DataSetter(config)

        perform_pipeline(
            datasetter=datasetter,
            models=config.models,
            config_handler=config,
        )

        # Load and verify metrics
        metrics_path = Path(config.out_folder) / "metrics" / "resistant" / "LR_metrics_detailed.csv"
        metrics_df = pd.read_csv(metrics_path)

        # Check expected columns
        assert "Metric" in metrics_df.columns
        assert "Mean" in metrics_df.columns
        assert "Std" in metrics_df.columns
        assert "CI95_lower" in metrics_df.columns
        assert "CI95_upper" in metrics_df.columns

        # Check expected metrics
        metric_names = metrics_df["Metric"].tolist()
        assert "F1 (weighted)" in metric_names
        assert "MCC" in metric_names
        assert "AUROC" in metric_names
        assert "Balanced Acc" in metric_names


class TestPerformTrainingIntegration:
    """Integration tests for perform_training."""

    @pytest.mark.slow
    def test_training_creates_model_files(self, tmp_path):
        """Test that perform_training creates model files."""
        df = create_synthetic_data(n_samples=60, seed=42)
        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)

        config_path = create_test_config(tmp_path, data_path)

        config = ConfigHandler(str(config_path))
        datasetter = DataSetter(config)

        perform_training(
            datasetter=datasetter,
            models=config.models,
            config_handler=config,
        )

        # Verify outputs
        trained_models_dir = Path(config.out_folder) / "trained_models"
        assert trained_models_dir.exists()

        # Check metadata file
        metadata_path = trained_models_dir / "training_metadata.json"
        assert metadata_path.exists()

        with open(metadata_path) as f:
            metadata = json.load(f)

        assert "features" in metadata
        assert "targets" in metadata
        assert "continuous_features" in metadata
        assert "categorical_features" in metadata

        # Check model file
        model_files = list(trained_models_dir.glob("*.joblib"))
        assert len(model_files) > 0

    @pytest.mark.slow
    def test_training_with_threshold_calibration(self, tmp_path):
        """Test that training correctly calibrates threshold."""
        df = create_synthetic_data(n_samples=60, seed=42)
        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)

        config_path = create_test_config(tmp_path, data_path)

        config = ConfigHandler(str(config_path))
        datasetter = DataSetter(config)

        perform_training(
            datasetter=datasetter,
            models=config.models,
            config_handler=config,
        )

        # Load model and check threshold
        import joblib

        trained_models_dir = Path(config.out_folder) / "trained_models"
        model_files = list(trained_models_dir.glob("*.joblib"))

        for model_file in model_files:
            bundle = joblib.load(model_file)
            assert "threshold" in bundle
            # Threshold should be between 0 and 1
            assert 0 < bundle["threshold"] < 1


class TestPerformEvaluationIntegration:
    """Integration tests for perform_evaluation."""

    @pytest.mark.slow
    def test_evaluation_on_trained_models(self, tmp_path):
        """Test full train -> evaluate workflow."""
        # Create and save training data
        train_df = create_synthetic_data(n_samples=60, seed=42)
        train_path = tmp_path / "train_data.csv"
        train_df.to_csv(train_path, index=False)

        # Create test data (same structure, different samples)
        test_df = create_synthetic_data(n_samples=30, seed=123)
        test_path = tmp_path / "test_data.csv"
        test_df.to_csv(test_path, index=False)

        # Train models
        config_path = create_test_config(tmp_path, train_path)
        config = ConfigHandler(str(config_path))
        datasetter = DataSetter(config)

        perform_training(
            datasetter=datasetter,
            models=config.models,
            config_handler=config,
        )

        # Evaluate on test data
        eval_output_dir = tmp_path / "eval_output"
        trained_models_dir = Path(config.out_folder) / "trained_models"

        results = perform_evaluation(
            models_dir=trained_models_dir,
            data_path=test_path,
            output_dir=eval_output_dir,
            verbose=False,
        )

        # Verify results
        assert len(results) > 0

        for key, metrics in results.items():
            assert "F1 (weighted)" in metrics
            assert "MCC" in metrics
            assert "AUROC" in metrics

        # Verify output files
        assert eval_output_dir.exists()
        assert (eval_output_dir / "evaluation_summary.csv").exists()
        assert (eval_output_dir / "predictions").exists()

    @pytest.mark.slow
    def test_evaluation_predictions_format(self, tmp_path):
        """Test that evaluation predictions have correct format."""
        train_df = create_synthetic_data(n_samples=60, seed=42)
        train_path = tmp_path / "train_data.csv"
        train_df.to_csv(train_path, index=False)

        test_df = create_synthetic_data(n_samples=30, seed=123)
        test_path = tmp_path / "test_data.csv"
        test_df.to_csv(test_path, index=False)

        config_path = create_test_config(tmp_path, train_path)
        config = ConfigHandler(str(config_path))
        datasetter = DataSetter(config)

        perform_training(
            datasetter=datasetter,
            models=config.models,
            config_handler=config,
        )

        eval_output_dir = tmp_path / "eval_output"
        trained_models_dir = Path(config.out_folder) / "trained_models"

        perform_evaluation(
            models_dir=trained_models_dir,
            data_path=test_path,
            output_dir=eval_output_dir,
            verbose=False,
        )

        # Check predictions file format
        pred_files = list((eval_output_dir / "predictions").glob("*.csv"))
        assert len(pred_files) > 0

        for pred_file in pred_files:
            pred_df = pd.read_csv(pred_file)
            assert "y_true" in pred_df.columns
            assert "y_pred" in pred_df.columns
            assert "y_prob" in pred_df.columns
            # Predictions should be binary
            assert set(pred_df["y_pred"].unique()).issubset({0, 1})
            # Probabilities should be between 0 and 1
            assert pred_df["y_prob"].min() >= 0
            assert pred_df["y_prob"].max() <= 1


class TestPipelineWithImputation:
    """Integration tests for pipeline with imputation."""

    @pytest.mark.slow
    def test_pipeline_with_missing_values_and_imputation(self, tmp_path):
        """Test pipeline handles missing values with imputation."""
        # Create data with missing values
        df = create_synthetic_data(n_samples=60, seed=42)
        # Introduce some missing values
        np.random.seed(42)
        mask = np.random.random(60) < 0.1
        df.loc[mask, "age"] = np.nan
        mask = np.random.random(60) < 0.1
        df.loc[mask, "bmi"] = np.nan

        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)

        # Create config with imputation enabled
        config_text = dedent(f"""
        [Data]
        data_path = {data_path}
        targets = resistant
        continuous_features = age, bmi
        group_column = patient_id

        [Pipeline]
        models = LR
        outer_folds = 3
        inner_folds = 2
        calibrate_threshold = false

        [Reproducibility]
        seed = 42

        [Log]
        verbosity = 0
        log_basename = test.log

        [Resources]
        n_jobs = 1

        [Output]
        out_folder = {tmp_path / "output"}

        [ModelSaving]
        enable = true
        compression = 3

        [Imputation]
        method = simple
        strategy = mean
        n_neighbors = 5
        estimator = bayesian_ridge
        """).strip()

        config_path = tmp_path / "test_config.ini"
        config_path.write_text(config_text)

        config = ConfigHandler(str(config_path))
        datasetter = DataSetter(config)

        # Pipeline should complete without errors
        perform_pipeline(
            datasetter=datasetter,
            models=config.models,
            config_handler=config,
        )

        # Verify outputs exist
        assert (
            Path(config.out_folder) / "metrics" / "resistant" / "LR_metrics_detailed.csv"
        ).exists()
