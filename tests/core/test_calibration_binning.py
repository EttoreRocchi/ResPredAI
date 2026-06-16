"""Tests for calibration quantile-edge deduplication."""

import numpy as np

from respredai.core.calibration import (
    compute_reliability_curve,
    expected_calibration_error,
)


class TestQuantileDedupe:
    """Tied probabilities must not silently collapse quantile bins."""

    def test_tied_probabilities_no_crash_and_dedup(self):
        y_prob = np.concatenate([np.full(90, 0.2), np.linspace(0.6, 0.9, 10)])
        y_true = np.concatenate([np.zeros(90, dtype=int), np.ones(10, dtype=int)])

        prob_true, prob_pred, counts = compute_reliability_curve(
            y_true, y_prob, n_bins=10, strategy="quantile"
        )
        assert len(prob_true) == len(prob_pred) == len(counts)
        # Far fewer effective bins than requested because of the ties
        assert 1 <= len(prob_true) < 10
        # Every sample is accounted for across the (deduped) bins
        assert counts.sum() == len(y_true)

    def test_ece_quantile_tied_finite(self):
        y_prob = np.concatenate([np.full(90, 0.2), np.linspace(0.6, 0.9, 10)])
        y_true = np.concatenate([np.zeros(90, dtype=int), np.ones(10, dtype=int)])
        ece = expected_calibration_error(y_true, y_prob, n_bins=10, strategy="quantile")
        assert 0.0 <= ece <= 1.0


class TestCIMetricFnHonorsNBins:
    """The ECE/MCE bootstrap CI must use the same bin count as the point estimate."""

    def test_ece_ci_fn_uses_n_bins(self):
        from respredai.core.metrics import METRIC_FUNCTIONS, _ci_metric_fn

        rng = np.random.RandomState(0)
        y_true = rng.randint(0, 2, 200)
        p1 = rng.rand(200)
        y_prob = np.column_stack([1 - p1, p1])

        ece_5 = _ci_metric_fn("ECE", 5)(y_true, None, y_prob)
        ece_50 = _ci_metric_fn("ECE", 50)(y_true, None, y_prob)
        assert ece_5 != ece_50  # bin count changes the ECE estimate

        # A non-calibration metric falls back to the standard fixed function
        assert _ci_metric_fn("AUROC", 5) is METRIC_FUNCTIONS["AUROC"]
