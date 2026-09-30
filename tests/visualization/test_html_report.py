"""Tests for the HTML report generation."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from respredai.core.constants import DIR_METRICS
from respredai.visualization.html_report import _collect_metrics_data, generate_html_report


def _write_metrics_csv(output: Path, model: str, target: str, ci_pct: int) -> None:
    metrics_dir = output / DIR_METRICS / target
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["AUROC", "F1 (weighted)", "MCC", "Balanced Acc", "VME", "ME", "Brier Score", "ECE"]
    df = pd.DataFrame(
        {
            "Metric": metrics,
            "Mean": [0.8, 0.7, 0.4, 0.75, 0.2, 0.3, 0.15, 0.05],
            "Std": [0.05] * len(metrics),
            "SE": [0.03] * len(metrics),
            f"CI{ci_pct}_lower": [0.7, 0.6, 0.3, 0.65, 0.1, 0.2, 0.1, 0.02],
            f"CI{ci_pct}_upper": [0.9, 0.8, 0.5, 0.85, 0.3, 0.4, 0.2, 0.08],
        }
    )
    df.to_csv(metrics_dir / f"{model}_metrics_detailed.csv", index=False)


def _fake_config(tmp_path: Path, confidence_level: float) -> SimpleNamespace:
    """The subset of ConfigHandler the report reads, without loading data."""
    return SimpleNamespace(
        config_path="config.ini",
        data_cfg=SimpleNamespace(data_path="data.csv", targets=["y"]),
        pipeline=SimpleNamespace(
            models=["LR"],
            outer_folds=3,
            inner_folds=2,
            outer_cv_repeats=1,
            calibrate_threshold=False,
            threshold_method="auto",
            threshold_objective="youden",
            vme_cost=1.0,
            me_cost=1.0,
            calibrate_probabilities=False,
            probability_calibration_method="sigmoid",
            probability_calibration_cv=5,
            confidence_level=confidence_level,
            n_bootstrap=1000,
        ),
        imputation=SimpleNamespace(
            method="none", strategy="mean", n_neighbors=5, estimator="bayesian_ridge"
        ),
        output=SimpleNamespace(out_folder=str(tmp_path)),
        reproducibility_cfg=SimpleNamespace(seed=42, n_jobs=1, conformal_alpha=0.1),
    )


class TestConfidenceIntervalColumns:
    """The report must pick up the CI columns whatever the confidence level."""

    def test_collect_metrics_detects_any_ci_level(self, tmp_path):
        _write_metrics_csv(tmp_path, "LR", "y", ci_pct=90)
        data = _collect_metrics_data(tmp_path, ["LR"], ["y"])
        assert data["LR_y"]["AUROC_ci_lower"] == 0.7
        assert data["LR_y"]["AUROC_ci_upper"] == 0.9

    def test_report_labels_and_shows_non_default_ci(self, tmp_path):
        _write_metrics_csv(tmp_path, "LR", "y", ci_pct=90)
        report = generate_html_report(str(tmp_path), ["LR"], ["y"], _fake_config(tmp_path, 0.9))
        html = report.read_text()
        assert "[90% CI]" in html
        assert "95% CI" not in html
        assert "[0.700-0.900]" in html
