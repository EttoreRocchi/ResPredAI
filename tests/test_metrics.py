"""Unit tests for metrics.py - comprehensive metrics and bootstrap CI tests."""

import numpy as np
import pandas as pd

from respredai.core.metrics import (
    METRIC_FUNCTIONS,
    bootstrap_ci_samples,
    metric_dict,
    save_metrics_summary,
    youden_j_score,
)


class TestYoudenJScore:
    """Unit tests for Youden's J statistic."""

    def test_youden_j_perfect_predictions(self):
        """Test Youden's J with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        j = youden_j_score(y_true, y_pred)
        assert j == 1.0

    def test_youden_j_random_predictions(self):
        """Test Youden's J with completely wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        j = youden_j_score(y_true, y_pred)
        assert j == -1.0

    def test_youden_j_partial_correct(self):
        """Test Youden's J with partially correct predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        j = youden_j_score(y_true, y_pred)
        # TPR = 1/2 = 0.5, TNR = 1/2 = 0.5, J = 0.5 + 0.5 - 1 = 0
        assert j == 0.0

    def test_youden_j_all_positive_predictions(self):
        """Test Youden's J when predicting all positive."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        j = youden_j_score(y_true, y_pred)
        # TPR = 1, TNR = 0, J = 0
        assert j == 0.0

    def test_youden_j_all_negative_predictions(self):
        """Test Youden's J when predicting all negative."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        j = youden_j_score(y_true, y_pred)
        # TPR = 0, TNR = 1, J = 0
        assert j == 0.0


class TestMetricDict:
    """Unit tests for metric_dict function."""

    def test_metric_dict_returns_all_metrics(self):
        """Test that metric_dict returns all expected metrics."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4]])

        metrics = metric_dict(y_true, y_pred, y_prob)

        expected_keys = [
            "Precision (0)",
            "Precision (1)",
            "Recall (0)",
            "Recall (1)",
            "F1 (0)",
            "F1 (1)",
            "F1 (weighted)",
            "MCC",
            "Balanced Acc",
            "AUROC",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_metric_dict_values_in_range(self):
        """Test that all metrics are in valid ranges."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.1, 0.9], [0.9, 0.1], [0.6, 0.4]])

        metrics = metric_dict(y_true, y_pred, y_prob)

        for key, value in metrics.items():
            if key == "MCC":
                assert -1 <= value <= 1, f"{key} out of range: {value}"
            else:
                assert 0 <= value <= 1, f"{key} out of range: {value}"

    def test_metric_dict_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

        metrics = metric_dict(y_true, y_pred, y_prob)

        assert metrics["F1 (weighted)"] == 1.0
        assert metrics["MCC"] == 1.0
        assert metrics["Balanced Acc"] == 1.0
        assert metrics["AUROC"] == 1.0

    def test_metric_dict_handles_single_class_auroc(self):
        """Test that AUROC is NaN when only one class present."""
        import warnings

        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        y_prob = np.array([[0.1, 0.9], [0.2, 0.8], [0.1, 0.9], [0.0, 1.0]])

        # Suppress expected warning about single label in confusion matrix
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            metrics = metric_dict(y_true, y_pred, y_prob)

        assert np.isnan(metrics["AUROC"])


class TestBootstrapCISamples:
    """Unit tests for bootstrap_ci_samples function."""

    def test_bootstrap_ci_returns_tuple(self):
        """Test that bootstrap_ci_samples returns a tuple of two values."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
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
            ]
        )

        lower, upper = bootstrap_ci_samples(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_fn=METRIC_FUNCTIONS["F1 (weighted)"],
            n_bootstrap=100,
            random_state=42,
        )

        assert isinstance(lower, float)
        assert isinstance(upper, float)

    def test_bootstrap_ci_lower_less_than_upper(self):
        """Test that lower bound is less than or equal to upper bound."""
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

        lower, upper = bootstrap_ci_samples(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_fn=METRIC_FUNCTIONS["MCC"],
            n_bootstrap=100,
            random_state=42,
        )

        assert lower <= upper

    def test_bootstrap_ci_confidence_level(self):
        """Test that different confidence levels produce different intervals."""
        np.random.seed(42)
        n = 100
        y_true = np.random.randint(0, 2, n)
        y_pred = np.random.randint(0, 2, n)
        y_prob = np.column_stack([1 - np.random.rand(n), np.random.rand(n)])

        lower_90, upper_90 = bootstrap_ci_samples(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_fn=METRIC_FUNCTIONS["F1 (weighted)"],
            confidence=0.90,
            n_bootstrap=500,
            random_state=42,
        )

        lower_95, upper_95 = bootstrap_ci_samples(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_fn=METRIC_FUNCTIONS["F1 (weighted)"],
            confidence=0.95,
            n_bootstrap=500,
            random_state=42,
        )

        # 95% CI should be wider than or equal to 90% CI
        ci_width_90 = upper_90 - lower_90
        ci_width_95 = upper_95 - lower_95
        assert ci_width_95 >= ci_width_90 - 0.01  # Allow small tolerance

    def test_bootstrap_ci_reproducibility(self):
        """Test that same random_state produces same results."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 0, 1])
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
            ]
        )

        result1 = bootstrap_ci_samples(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_fn=METRIC_FUNCTIONS["MCC"],
            n_bootstrap=100,
            random_state=123,
        )

        result2 = bootstrap_ci_samples(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_fn=METRIC_FUNCTIONS["MCC"],
            n_bootstrap=100,
            random_state=123,
        )

        assert result1 == result2

    def test_bootstrap_ci_all_metrics(self):
        """Test bootstrap CI calculation for all defined metrics."""
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

        for metric_name, metric_fn in METRIC_FUNCTIONS.items():
            lower, upper = bootstrap_ci_samples(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                metric_fn=metric_fn,
                n_bootstrap=50,
                random_state=42,
            )
            assert not np.isnan(lower), f"NaN lower bound for {metric_name}"
            assert not np.isnan(upper), f"NaN upper bound for {metric_name}"

    def test_bootstrap_ci_handles_empty_bootstrap(self):
        """Test that bootstrap handles case where all samples are single-class."""
        # Very small dataset where bootstrap might produce single-class samples
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        y_prob = np.array([[0.9, 0.1], [0.1, 0.9]])

        # This might return NaN if all bootstrap samples are single-class
        lower, upper = bootstrap_ci_samples(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            metric_fn=METRIC_FUNCTIONS["F1 (weighted)"],
            n_bootstrap=10,
            random_state=42,
        )

        # Should handle gracefully (either return valid bounds or NaN)
        assert isinstance(lower, float)
        assert isinstance(upper, float)


class TestSaveMetricsSummary:
    """Unit tests for save_metrics_summary function."""

    def test_save_metrics_summary_file_content(self, tmp_path):
        """Test that saved file contains expected columns and data."""
        metrics_dict = [
            {
                "Precision (0)": 0.8,
                "Precision (1)": 0.7,
                "Recall (0)": 0.9,
                "Recall (1)": 0.6,
                "F1 (0)": 0.85,
                "F1 (1)": 0.65,
                "F1 (weighted)": 0.75,
                "MCC": 0.5,
                "Balanced Acc": 0.75,
                "AUROC": 0.8,
            },
            {
                "Precision (0)": 0.82,
                "Precision (1)": 0.72,
                "Recall (0)": 0.88,
                "Recall (1)": 0.62,
                "F1 (0)": 0.84,
                "F1 (1)": 0.66,
                "F1 (weighted)": 0.76,
                "MCC": 0.52,
                "Balanced Acc": 0.76,
                "AUROC": 0.82,
            },
        ]

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

        output_path = tmp_path / "metrics.csv"

        result_df = save_metrics_summary(
            metrics_dict=metrics_dict,
            output_path=output_path,
            n_bootstrap=50,
            y_true_all=y_true,
            y_pred_all=y_pred,
            y_prob_all=y_prob,
        )

        # Check returned dataframe
        assert "Metric" in result_df.columns
        assert "Mean" in result_df.columns
        assert "Std" in result_df.columns
        assert "CI95_lower" in result_df.columns
        assert "CI95_upper" in result_df.columns

        # Check file was created
        assert output_path.exists()

        # Check file content
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 10  # 10 metrics

    def test_save_metrics_summary_creates_parent_dirs(self, tmp_path):
        """Test that save_metrics_summary creates parent directories."""
        metrics_dict = [{"F1 (weighted)": 0.75}, {"F1 (weighted)": 0.76}]

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4]])

        # Nested path that doesn't exist
        output_path = tmp_path / "nested" / "dir" / "metrics.csv"

        save_metrics_summary(
            metrics_dict=metrics_dict,
            output_path=output_path,
            n_bootstrap=10,
            y_true_all=y_true,
            y_pred_all=y_pred,
            y_prob_all=y_prob,
        )

        assert output_path.exists()
