"""Unit tests for calibration metrics module."""

import numpy as np

from respredai.core.calibration import (
    CALIBRATION_METRIC_FUNCTIONS,
    brier_score,
    calibration_metrics_dict,
    compute_reliability_curve,
    expected_calibration_error,
    maximum_calibration_error,
)


class TestBrierScore:
    """Unit tests for Brier score calculation."""

    def test_perfect_predictions(self):
        """Brier score should be 0 for perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        assert brier_score(y_true, y_prob) == 0.0

    def test_worst_predictions(self):
        """Brier score should be 1 for completely wrong predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])
        assert brier_score(y_true, y_prob) == 1.0

    def test_random_predictions(self):
        """Brier score should be between 0 and 1 for random predictions."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        bs = brier_score(y_true, y_prob)
        assert 0 <= bs <= 1

    def test_all_positive(self):
        """Test with all positive samples."""
        y_true = np.array([1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.7, 0.6])
        bs = brier_score(y_true, y_prob)
        expected = np.mean((y_prob - y_true) ** 2)
        assert np.isclose(bs, expected)


class TestExpectedCalibrationError:
    """Unit tests for Expected Calibration Error (ECE)."""

    def test_ece_range(self):
        """ECE should be between 0 and 1."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        ece = expected_calibration_error(y_true, y_prob)
        assert 0 <= ece <= 1

    def test_ece_empty_input(self):
        """ECE should handle empty input gracefully."""
        y_true = np.array([])
        y_prob = np.array([])
        ece = expected_calibration_error(y_true, y_prob)
        assert ece == 0.0

    def test_ece_constant_predictions(self):
        """ECE should handle constant predictions (all same probability)."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        ece = expected_calibration_error(y_true, y_prob)
        # With all predictions at 0.5 and 40% positive rate, ECE = |0.4 - 0.5| = 0.1
        assert 0 <= ece <= 1

    def test_ece_different_bins(self):
        """ECE should work with different numbers of bins."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)

        ece_5 = expected_calibration_error(y_true, y_prob, n_bins=5)
        ece_10 = expected_calibration_error(y_true, y_prob, n_bins=10)
        ece_20 = expected_calibration_error(y_true, y_prob, n_bins=20)

        # All should be valid
        assert 0 <= ece_5 <= 1
        assert 0 <= ece_10 <= 1
        assert 0 <= ece_20 <= 1


class TestMaximumCalibrationError:
    """Unit tests for Maximum Calibration Error (MCE)."""

    def test_mce_range(self):
        """MCE should be between 0 and 1."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        mce = maximum_calibration_error(y_true, y_prob)
        assert 0 <= mce <= 1

    def test_mce_gte_ece(self):
        """MCE should always be greater than or equal to ECE."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        ece = expected_calibration_error(y_true, y_prob)
        mce = maximum_calibration_error(y_true, y_prob)
        assert mce >= ece

    def test_mce_empty_input(self):
        """MCE should handle empty input gracefully."""
        y_true = np.array([])
        y_prob = np.array([])
        mce = maximum_calibration_error(y_true, y_prob)
        assert mce == 0.0

    def test_mce_constant_predictions(self):
        """MCE should handle constant predictions."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        mce = maximum_calibration_error(y_true, y_prob)
        assert 0 <= mce <= 1


class TestComputeReliabilityCurve:
    """Unit tests for reliability curve computation."""

    def test_returns_correct_shapes(self):
        """Returned arrays should have compatible shapes."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)

        prob_true, prob_pred, bin_counts = compute_reliability_curve(y_true, y_prob)

        assert len(prob_true) == len(prob_pred)
        # bin_counts may have different length due to empty bins being filtered

    def test_empty_input(self):
        """Should handle empty input."""
        y_true = np.array([])
        y_prob = np.array([])

        prob_true, prob_pred, bin_counts = compute_reliability_curve(y_true, y_prob)

        assert len(prob_true) == 0
        assert len(prob_pred) == 0
        assert len(bin_counts) == 0

    def test_constant_predictions(self):
        """Should handle constant predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])

        prob_true, prob_pred, bin_counts = compute_reliability_curve(y_true, y_prob)

        # Should return at least one bin
        assert len(prob_true) >= 1


class TestCalibrationMetricsDict:
    """Unit tests for calibration_metrics_dict function."""

    def test_returns_all_metrics(self):
        """Should return dictionary with all three calibration metrics."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.3, 0.4, 0.6, 0.7])
        metrics = calibration_metrics_dict(y_true, y_prob)

        assert "Brier Score" in metrics
        assert "ECE" in metrics
        assert "MCE" in metrics

    def test_metric_values_valid(self):
        """All metric values should be valid floats."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.rand(100)
        metrics = calibration_metrics_dict(y_true, y_prob)

        for name, value in metrics.items():
            assert isinstance(value, float)
            assert not np.isnan(value)
            assert 0 <= value <= 1


class TestCalibrationMetricFunctions:
    """Unit tests for CALIBRATION_METRIC_FUNCTIONS wrappers."""

    def test_all_functions_present(self):
        """All calibration metrics should have wrapper functions."""
        assert "Brier Score" in CALIBRATION_METRIC_FUNCTIONS
        assert "ECE" in CALIBRATION_METRIC_FUNCTIONS
        assert "MCE" in CALIBRATION_METRIC_FUNCTIONS

    def test_wrappers_work_with_2d_prob(self):
        """Wrappers should work with 2D probability arrays (sklearn format)."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        # 2D array with probabilities for both classes
        y_prob = np.array([[0.7, 0.3], [0.6, 0.4], [0.4, 0.6], [0.3, 0.7]])

        for name, fn in CALIBRATION_METRIC_FUNCTIONS.items():
            value = fn(y_true, y_pred, y_prob)
            assert isinstance(value, float)
            assert 0 <= value <= 1
