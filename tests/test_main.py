import numpy as np
import pandas as pd

from respredai.core.metrics import metric_dict, save_metrics_summary
from respredai.core.models import generate_summary_report


class TestMetricDict:
    """Unit tests for metric_dict function."""

    def test_metric_dict_binary_classification(self):
        """Test metrics calculation for binary classification."""

        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0.9, 0.1], [0.6, 0.4]])

        metrics = metric_dict(y_true, y_pred, y_prob)

        assert "Precision (0)" in metrics
        assert "Precision (1)" in metrics
        assert "Recall (0)" in metrics
        assert "Recall (1)" in metrics
        assert "F1 (weighted)" in metrics
        assert "MCC" in metrics
        assert "Balanced Acc" in metrics
        assert "AUROC" in metrics

        # Values should be in valid range
        for key, value in metrics.items():
            if key != "MCC":
                assert -1 <= value <= 1
            else:
                assert 0 <= value <= 1

    def test_metric_dict_perfect_predictions(self):
        """Test metrics with perfect predictions."""

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

        metrics = metric_dict(y_true, y_pred, y_prob)

        assert metrics["F1 (weighted)"] == 1.0
        assert metrics["MCC"] == 1.0
        assert metrics["Balanced Acc"] == 1.0


class TestSaveMetricsSummary:
    """Unit tests for save_metrics_summary function."""

    def test_save_metrics_summary_creates_file(self, tmp_path):
        """Test that save_metrics_summary creates a CSV file."""

        metrics_dict = [
            {"Precision (0)": 0.8, "Recall (0)": 0.9, "F1 (0)": 0.85},
            {"Precision (0)": 0.82, "Recall (0)": 0.88, "F1 (0)": 0.84},
            {"Precision (0)": 0.78, "Recall (0)": 0.92, "F1 (0)": 0.86},
        ]

        # Sample-level predictions for bootstrap CI
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0])
        y_prob = np.array(
            [
                [0.8, 0.2],
                [0.3, 0.7],
                [0.2, 0.8],
                [0.1, 0.9],
                [0.9, 0.1],
                [0.6, 0.4],
                [0.7, 0.3],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.8, 0.2],
            ]
        )

        output_path = tmp_path / "metrics" / "test_metrics.csv"

        save_metrics_summary(
            metrics_dict=metrics_dict,
            output_path=output_path,
            n_bootstrap=100,  # Reduced for faster test
            y_true_all=y_true,
            y_pred_all=y_pred,
            y_prob_all=y_prob,
        )

        assert output_path.exists()

        df = pd.read_csv(output_path)
        assert "Metric" in df.columns
        assert "Mean" in df.columns
        assert "Std" in df.columns
        assert "CI95_lower" in df.columns
        assert "CI95_upper" in df.columns


class TestGenerateSummaryReport:
    """Unit tests for generate_summary_report function."""

    def test_generate_summary_report_creates_files(self, tmp_path):
        """Test that summary report creates expected CSV files."""

        # Create mock metrics structure
        metrics_dir = tmp_path / "metrics"
        target_dir = metrics_dir / "Target1"
        target_dir.mkdir(parents=True)

        # Create mock metrics files
        lr_metrics = pd.DataFrame(
            {
                "Metric": ["Precision (0)", "F1 (weighted)", "MCC"],
                "Mean": [0.8, 0.75, 0.3],
                "Std": [0.02, 0.03, 0.05],
            }
        )
        lr_metrics.to_csv(target_dir / "LR_metrics_detailed.csv", index=False)

        rf_metrics = pd.DataFrame(
            {
                "Metric": ["Precision (0)", "F1 (weighted)", "MCC"],
                "Mean": [0.82, 0.77, 0.35],
                "Std": [0.01, 0.02, 0.04],
            }
        )
        rf_metrics.to_csv(target_dir / "RF_metrics_detailed.csv", index=False)

        # Generate summary
        generate_summary_report(
            output_folder=str(tmp_path), models=["LR", "RF"], targets=["Target1"]
        )

        # Check files were created
        assert (target_dir / "summary.csv").exists()
        assert (metrics_dir / "summary_all.csv").exists()

        # Check content
        summary_df = pd.read_csv(target_dir / "summary.csv")
        assert len(summary_df) == 2
        assert "Model" in summary_df.columns

    def test_generate_summary_report_handles_missing_files(self, tmp_path):
        """Test that summary report handles missing metric files gracefully."""

        metrics_dir = tmp_path / "metrics"
        target_dir = metrics_dir / "Target1"
        target_dir.mkdir(parents=True)

        # Only create one metrics file
        lr_metrics = pd.DataFrame({"Metric": ["F1 (weighted)"], "Mean": [0.75], "Std": [0.03]})
        lr_metrics.to_csv(target_dir / "LR_metrics_detailed.csv", index=False)

        # Generate summary - RF file doesn't exist
        generate_summary_report(
            output_folder=str(tmp_path), models=["LR", "RF"], targets=["Target1"]
        )

        summary_df = pd.read_csv(target_dir / "summary.csv")
        # Should only have LR since RF file doesn't exist
        assert len(summary_df) == 1
        assert summary_df["Model"].iloc[0] == "LR"

    def test_generate_summary_report_multiple_targets(self, tmp_path):
        """Test summary report with multiple targets."""

        metrics_dir = tmp_path / "metrics"

        for target in ["Target1", "Target2"]:
            target_dir = metrics_dir / target
            target_dir.mkdir(parents=True)

            metrics = pd.DataFrame({"Metric": ["F1 (weighted)"], "Mean": [0.75], "Std": [0.03]})
            metrics.to_csv(target_dir / "LR_metrics_detailed.csv", index=False)

        generate_summary_report(
            output_folder=str(tmp_path), models=["LR"], targets=["Target1", "Target2"]
        )

        # Check global summary has both targets
        all_summary = pd.read_csv(metrics_dir / "summary_all.csv")
        assert len(all_summary) == 2
        assert "Target" in all_summary.columns
        assert set(all_summary["Target"]) == {"Target1", "Target2"}
