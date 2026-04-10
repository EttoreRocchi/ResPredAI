"""Unit tests for metrics.py - comprehensive metrics and bootstrap CI tests."""

import warnings

import numpy as np
import pandas as pd

from respredai.core.metrics import (
    METRIC_FUNCTIONS,
    bootstrap_ci_samples,
    compute_conformal_qhat,
    conformal_coverage_report,
    conformal_prediction_sets,
    cost_sensitive_score,
    f1_threshold_score,
    f2_threshold_score,
    get_threshold_scorer,
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
            "VME",
            "ME",
            "FOR",
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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
                "VME": 0.4,
                "ME": 0.1,
                "FOR": 0.2,
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
                "VME": 0.38,
                "ME": 0.12,
                "FOR": 0.18,
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
        assert len(loaded_df) == 13

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


class TestThresholdScorers:
    """Unit tests for threshold scoring functions."""

    def test_f1_threshold_score_perfect(self):
        """Test F1 threshold scorer with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        score = f1_threshold_score(y_true, y_pred)
        assert score == 1.0

    def test_f1_threshold_score_no_true_positives(self):
        """Test F1 threshold scorer with no true positives."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        score = f1_threshold_score(y_true, y_pred)
        assert score == 0.0

    def test_f2_threshold_score_perfect(self):
        """Test F2 threshold scorer with perfect predictions."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        score = f2_threshold_score(y_true, y_pred)
        assert score == 1.0

    def test_f2_weights_recall_higher(self):
        """Test that F2 weights recall higher than precision."""
        y_true = np.array([0, 0, 1, 1, 1, 1])
        # High recall (3/4), lower precision (3/4)
        y_pred_high_recall = np.array([0, 1, 1, 1, 1, 0])
        # Lower recall (2/4), high precision (2/2)
        y_pred_high_precision = np.array([0, 0, 1, 1, 0, 0])

        f2_high_recall = f2_threshold_score(y_true, y_pred_high_recall)
        f2_high_precision = f2_threshold_score(y_true, y_pred_high_precision)

        # F2 should favor higher recall
        assert f2_high_recall > f2_high_precision

    def test_cost_sensitive_score_equal_weights(self):
        """Test cost sensitive scorer with equal weights."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        score = cost_sensitive_score(y_true, y_pred, vme_cost=1.0, me_cost=1.0)
        assert score == 0.0  # Perfect predictions, no cost

    def test_cost_sensitive_score_vme_weighted(self):
        """Test that higher VME cost penalizes false negatives more."""
        y_true = np.array([0, 0, 1, 1])
        # One FN (VME) - predicting susceptible when resistant
        y_pred_fn = np.array([0, 0, 0, 1])
        # One FP (ME) - predicting resistant when susceptible
        y_pred_fp = np.array([0, 1, 1, 1])

        # With higher VME cost, FN should have lower (more negative) score
        score_fn = cost_sensitive_score(y_true, y_pred_fn, vme_cost=5.0, me_cost=1.0)
        score_fp = cost_sensitive_score(y_true, y_pred_fp, vme_cost=5.0, me_cost=1.0)

        assert score_fn < score_fp

    def test_get_threshold_scorer_youden(self):
        """Test get_threshold_scorer returns youden_j_score."""
        scorer = get_threshold_scorer("youden")
        assert scorer == youden_j_score

    def test_get_threshold_scorer_f1(self):
        """Test get_threshold_scorer returns f1 scorer."""
        scorer = get_threshold_scorer("f1")
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert scorer(y_true, y_pred) == 1.0

    def test_get_threshold_scorer_f2(self):
        """Test get_threshold_scorer returns f2 scorer."""
        scorer = get_threshold_scorer("f2")
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert scorer(y_true, y_pred) == 1.0

    def test_get_threshold_scorer_cost_sensitive(self):
        """Test get_threshold_scorer returns cost sensitive scorer."""
        scorer = get_threshold_scorer("cost_sensitive", vme_cost=5.0, me_cost=1.0)
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        assert scorer(y_true, y_pred) == 0.0

    def test_get_threshold_scorer_invalid(self):
        """Test get_threshold_scorer raises on invalid objective."""
        import pytest

        with pytest.raises(ValueError, match="Unknown threshold objective"):
            get_threshold_scorer("invalid")


class TestConformalPrediction:
    """Unit tests for Mondrian conformal prediction."""

    def test_compute_conformal_qhat_perfect_predictions(self):
        """Perfect predictions should produce low q_hat (low nonconformity)."""
        rng = np.random.RandomState(42)
        n = 50
        y_true = np.array([0] * n + [1] * n)
        # Near-perfect predictions: high probability for the true class
        y_prob = np.zeros((2 * n, 2))
        y_prob[:n, 0] = 0.85 + 0.10 * rng.rand(n)  # class 0: p(0) in [0.85, 0.95]
        y_prob[:n, 1] = 1 - y_prob[:n, 0]
        y_prob[n:, 1] = 0.85 + 0.10 * rng.rand(n)  # class 1: p(1) in [0.85, 0.95]
        y_prob[n:, 0] = 1 - y_prob[n:, 1]

        q_hat = compute_conformal_qhat(y_true, y_prob, alpha=0.1)
        assert 0 in q_hat and 1 in q_hat
        assert q_hat[0] < 0.5  # low nonconformity for good predictions
        assert q_hat[1] < 0.5

    def test_compute_conformal_qhat_random_predictions(self):
        """Random predictions should produce high q_hat."""
        rng = np.random.RandomState(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_prob = rng.dirichlet([1, 1], size=100)
        q_hat = compute_conformal_qhat(y_true, y_prob, alpha=0.1)
        assert q_hat[0] > 0.5
        assert q_hat[1] > 0.5

    def test_compute_conformal_qhat_empty_class(self):
        """Empty class should return conservative q_hat = 1.0."""
        y_true = np.array([0, 0, 0])
        y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]])
        q_hat = compute_conformal_qhat(y_true, y_prob, alpha=0.1)
        assert q_hat[1] == 1.0  # no class-1 samples

    def test_conformal_prediction_sets_certain(self):
        """Confident predictions with tight q_hat should have set_size=1."""
        y_prob = np.array([[0.95, 0.05], [0.10, 0.90]])
        q_hat = {0: 0.2, 1: 0.2}
        set_sizes, is_uncertain = conformal_prediction_sets(y_prob, q_hat)
        np.testing.assert_array_equal(set_sizes, [1, 1])
        np.testing.assert_array_equal(is_uncertain, [False, False])

    def test_conformal_prediction_sets_uncertain(self):
        """With q_hat=1.0 for both classes, all samples should be uncertain."""
        y_prob = np.array([[0.5, 0.5], [0.6, 0.4]])
        q_hat = {0: 1.0, 1: 1.0}
        set_sizes, is_uncertain = conformal_prediction_sets(y_prob, q_hat)
        np.testing.assert_array_equal(set_sizes, [2, 2])
        np.testing.assert_array_equal(is_uncertain, [True, True])

    def test_conformal_coverage_report_keys(self):
        """Coverage report should contain all expected keys."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7]])
        q_hat = compute_conformal_qhat(y_true, y_prob, alpha=0.1)
        report = conformal_coverage_report(y_true, y_prob, q_hat, alpha=0.1)

        expected_keys = {
            "empirical_coverage_overall",
            "guaranteed_coverage",
            "avg_set_size",
            "fraction_uncertain",
            "fraction_empty",
            "empirical_coverage_class_0",
            "empirical_coverage_class_1",
        }
        assert set(report.keys()) == expected_keys

    def test_conformal_coverage_report_values_in_range(self):
        """All coverage report values should be in [0, 1] or [1, 2]."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array(
            [
                [0.8, 0.2],
                [0.7, 0.3],
                [0.2, 0.8],
                [0.3, 0.7],
                [0.9, 0.1],
                [0.1, 0.9],
            ]
        )
        q_hat = compute_conformal_qhat(y_true, y_prob, alpha=0.1)
        report = conformal_coverage_report(y_true, y_prob, q_hat, alpha=0.1)

        assert 0 <= report["empirical_coverage_overall"] <= 1
        assert 0 <= report["empirical_coverage_class_0"] <= 1
        assert 0 <= report["empirical_coverage_class_1"] <= 1
        assert 0 <= report["fraction_uncertain"] <= 1
        assert 0 <= report["fraction_empty"] <= 1
        assert 1 <= report["avg_set_size"] <= 2
        # Split conformal guarantee: 1 - alpha
        assert report["guaranteed_coverage"] == 0.9

    def test_conformal_cv_plus_guarantee(self):
        """CV+ guarantee should be 1-2*alpha, not 1-alpha."""
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.8, 0.2], [0.7, 0.3], [0.2, 0.8], [0.3, 0.7]])
        q_hat = compute_conformal_qhat(y_true, y_prob, alpha=0.1)
        report = conformal_coverage_report(y_true, y_prob, q_hat, alpha=0.1, is_cv_plus=True)
        assert report["guaranteed_coverage"] == 0.8


class TestNadeauBengioSE:
    """Verify that the Nadeau-Bengio corrected SE matches the analytical formula."""

    def test_se_matches_analytical_formula(self, tmp_path):
        """SE should equal sqrt( (1/k + n2/n1) * sigma_hat^2 ) per Nadeau & Bengio (2003)."""
        # 5 folds with known metric values
        fold_values = [0.80, 0.85, 0.78, 0.82, 0.90]
        k = 5
        metrics_dict = [{"F1 (weighted)": v} for v in fold_values]

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4]])

        output_path = tmp_path / "se_test.csv"
        result_df = save_metrics_summary(
            metrics_dict=metrics_dict,
            output_path=output_path,
            n_bootstrap=50,
            y_true_all=y_true,
            y_pred_all=y_pred,
            y_prob_all=y_prob,
            n_folds=k,
            n_repeats=1,
        )

        correction = 1.0 / k + 1.0 / (k - 1)
        sigma_hat_sq = np.var(fold_values, ddof=0)  # biased variance
        expected_se = np.sqrt(correction * sigma_hat_sq)

        se_row = result_df[result_df["Metric"] == "F1 (weighted)"]
        actual_se = se_row["SE"].values[0]

        np.testing.assert_allclose(actual_se, expected_se, rtol=1e-10)

        # Also verify SE is LARGER than standard SE (s/sqrt(k)), not smaller
        standard_se = np.std(fold_values, ddof=1) / np.sqrt(k)
        assert actual_se > standard_se, (
            f"Corrected SE ({actual_se:.6f}) should be larger than "
            f"standard SE ({standard_se:.6f}) for k={k}"
        )

    def test_se_repeated_cv(self, tmp_path):
        """SE for repeated CV should use repeat-level means and biased variance."""
        # 2 repeats x 3 folds = 6 fold values
        fold_values = [0.80, 0.85, 0.78, 0.82, 0.90, 0.76]
        k = 3
        n_repeats = 2
        metrics_dict = [{"F1 (weighted)": v} for v in fold_values]

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4]])

        output_path = tmp_path / "se_repeated.csv"
        result_df = save_metrics_summary(
            metrics_dict=metrics_dict,
            output_path=output_path,
            n_bootstrap=50,
            y_true_all=y_true,
            y_pred_all=y_pred,
            y_prob_all=y_prob,
            n_folds=k,
            n_repeats=n_repeats,
        )

        # Repeat-level means
        repeat_means = [np.mean(fold_values[:k]), np.mean(fold_values[k:])]
        correction = 1.0 / k + 1.0 / (k - 1)
        sigma_hat_sq = np.var(repeat_means, ddof=0)  # biased variance
        expected_se = np.sqrt(correction * sigma_hat_sq)

        se_row = result_df[result_df["Metric"] == "F1 (weighted)"]
        actual_se = se_row["SE"].values[0]

        np.testing.assert_allclose(actual_se, expected_se, rtol=1e-10)


class TestMakeNanMetricsCompleteness:
    """Verify that _make_nan_metrics covers all keys from metric_dict."""

    def test_nan_metrics_keys_match_metric_dict(self):
        """_make_nan_metrics must return the same keys as metric_dict."""
        from respredai.core.workflow import _make_nan_metrics

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])
        y_prob = np.array([[0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.6, 0.4]])

        expected_keys = set(metric_dict(y_true, y_pred, y_prob).keys())
        nan_keys = set(_make_nan_metrics().keys())

        assert nan_keys == expected_keys, (
            f"Missing keys in _make_nan_metrics: {expected_keys - nan_keys}. "
            f"Extra keys: {nan_keys - expected_keys}"
        )
