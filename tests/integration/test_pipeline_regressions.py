"""Integration regressions: checkpoint resume, training with groups, temporal failures, progress."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import TunedThresholdClassifierCV

import respredai.core.workflow as workflow_module
from respredai.core.models import get_model_path
from respredai.core.workflow import (
    perform_pipeline,
    perform_temporal_validation,
    perform_training,
)
from respredai.io.config import ConfigHandler, DataSetter

pytestmark = [
    pytest.mark.slow,
    pytest.mark.filterwarnings(
        "ignore::sklearn.exceptions.ConvergenceWarning",
        "ignore:.*Mean of empty slice.*:RuntimeWarning",
        "ignore:.*Degrees of freedom.*:RuntimeWarning",
        "ignore:.*Bootstrap CI.*:UserWarning",
        "ignore:.*failed.*:UserWarning",
    ),
]


def _make_data(n=150, seed=3, groups=False, temporal=False):
    rng = np.random.RandomState(seed)
    age = rng.uniform(20, 80, n)
    bmi = rng.uniform(18, 35, n)
    sex = rng.choice(["M", "F"], n)
    score = (age - 50) / 30 + (bmi - 25) / 10 + rng.normal(0, 0.3, n)
    y = (score > 0).astype(int)
    y[0], y[1] = 0, 1  # guarantee both classes
    df = pd.DataFrame({"age": age, "bmi": bmi, "sex": sex, "y": y})
    if groups:
        df["patient_id"] = [f"P{i // 3}" for i in range(n)]
    if temporal:
        df["date"] = pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d")
    return df


def _write_config(
    tmp_path,
    data_path,
    out_name,
    *,
    groups=False,
    temporal=False,
    save_models=False,
    pipeline=None,
):
    """Write a config file line by line (no dedent, so inserted lines stay valid)."""
    pipeline_opts = {
        "models": "LR",
        "outer_folds": 3,
        "inner_folds": 2,
        "calibrate_threshold": "false",
        "calibrate_probabilities": "false",
        "n_bootstrap": 100,
    }
    pipeline_opts.update(pipeline or {})
    lines = [
        "[Data]",
        f"data_path = {data_path}",
        "targets = y",
        "continuous_features = age, bmi",
        "",
        "[Metadata]",
    ]
    if groups:
        lines.append("group_column = patient_id")
    if temporal:
        lines.append("temporal_column = date")
    lines += ["", "[Pipeline]"] + [f"{k} = {v}" for k, v in pipeline_opts.items()]
    lines += [
        "",
        "[Reproducibility]",
        "seed = 42",
        "",
        "[Log]",
        "verbosity = 0",
        "log_basename = test.log",
        "",
        "[Resources]",
        "n_jobs = 1",
        "",
        "[Output]",
        f"out_folder = {tmp_path / out_name}",
        "",
        "[ModelSaving]",
        f"enable = {'true' if save_models else 'false'}",
    ]
    if temporal:
        lines += [
            "",
            "[Validation]",
            "validation_strategy = temporal",
            "temporal_split_ratio = 0.7",
        ]
    path = tmp_path / f"cfg_{out_name}.ini"
    path.write_text("\n".join(lines) + "\n")
    return ConfigHandler(str(path))


def _write_data(tmp_path, name, **kwargs):
    data_path = tmp_path / f"{name}.csv"
    _make_data(**kwargs).to_csv(data_path, index=False)
    return data_path


class _RecordingCallback:
    """Progress callback that records every call it receives."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def _record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return _record

    def of(self, name):
        return [(args, kwargs) for called, args, kwargs in self.calls if called == name]

    @property
    def names(self):
        return [called for called, _, _ in self.calls]


class TestCheckpointResume:
    def test_resume_from_partial_checkpoint(self, tmp_path):
        data_path = _write_data(tmp_path, "d")
        config = _write_config(tmp_path, data_path, "resume", save_models=True)
        perform_pipeline(DataSetter(config), config.pipeline.models, config)

        model_path = get_model_path(config.output.out_folder, "LR", "y")
        data = joblib.load(model_path)
        assert data["completed_folds"] == 3

        # Truncate the checkpoint to one completed fold, as an interrupted run leaves it
        n_first = len(data["fold_test_data"][0][0])
        per_fold = (
            "fold_models",
            "fold_transformers",
            "fold_ohe_transformers",
            "fold_thresholds",
            "fold_hyperparams",
            "fold_test_data",
        )
        for key in per_fold:
            data[key] = data[key][:1]
        for key in (
            "all_metrics",
            "f1scores",
            "mccs",
            "cms",
            "aurocs",
            "fold_y_true_calib",
            "fold_y_prob_calib",
        ):
            data["metrics"][key] = data["metrics"][key][:1]
        for key in ("all_y_true", "all_y_pred", "all_y_prob", "all_test_indices"):
            data["metrics"][key] = data["metrics"][key][:n_first]
        data["completed_folds"] = 1
        joblib.dump(data, model_path, compress=3)

        out = Path(config.output.out_folder)
        metrics_csv = out / "metrics" / "y" / "LR_metrics_detailed.csv"
        curve = out / "calibration" / "reliability_curve_LR_y.png"
        metrics_csv.unlink()
        curve.unlink()

        perform_pipeline(DataSetter(config), config.pipeline.models, config)

        resumed = joblib.load(model_path)
        assert resumed["completed_folds"] == 3
        assert len(resumed["metrics"]["all_metrics"]) == 3
        assert all(model is not None for model in resumed["fold_models"])
        assert len(resumed["metrics"]["fold_y_true_calib"]) == 3
        summary = pd.read_csv(metrics_csv).set_index("Metric")
        assert not np.isnan(summary.loc["AUROC", "Mean"])
        assert curve.exists()


class TestTrainingWithGroups:
    def test_train_with_groups_and_probability_calibration(self, tmp_path):
        data_path = _write_data(tmp_path, "d", groups=True)
        config = _write_config(
            tmp_path,
            data_path,
            "train",
            groups=True,
            pipeline={
                "calibrate_probabilities": "true",
                "calibrate_threshold": "true",
                "threshold_method": "auto",
            },
        )
        perform_training(DataSetter(config), config.pipeline.models, config)

        bundle_path = Path(config.output.out_folder) / "trained_models" / "LR_y.joblib"
        bundle = joblib.load(bundle_path)
        assert set(bundle["conformal_q_hat"]) == {0, 1}

        metadata = json.loads((bundle_path.parent / "training_metadata.json").read_text())
        # The resolved method is recorded, not the "auto" placeholder
        assert metadata["config"]["threshold_method"] == "oof"
        assert metadata["config"]["calibration_bins"] == 10


class TestTemporalValidation:
    def test_all_targets_failing_raises(self, tmp_path, monkeypatch):
        data_path = _write_data(tmp_path, "d", temporal=True)
        config = _write_config(tmp_path, data_path, "temporal_fail", temporal=True)

        def _boom(*args, **kwargs):
            raise RuntimeError("forced failure")

        monkeypatch.setattr(workflow_module, "metric_dict", _boom)
        with pytest.raises(RuntimeError, match="Temporal validation failed"):
            perform_temporal_validation(DataSetter(config), config.pipeline.models, config)

    def test_reports_progress(self, tmp_path):
        data_path = _write_data(tmp_path, "d", temporal=True)
        config = _write_config(tmp_path, data_path, "temporal_progress", temporal=True)
        callback = _RecordingCallback()
        perform_temporal_validation(
            DataSetter(config), config.pipeline.models, config, progress_callback=callback
        )
        assert callback.names[:3] == ["start", "start_model", "start_target"]
        assert "complete_fold" in callback.names
        assert callback.names[-2:] == ["complete_model", "stop"]


class TestProgressTotals:
    def test_repeated_cv_counts_every_iteration(self, tmp_path):
        data_path = _write_data(tmp_path, "d")
        config = _write_config(
            tmp_path, data_path, "repeats", save_models=True, pipeline={"outer_cv_repeats": 2}
        )
        callback = _RecordingCallback()
        perform_pipeline(
            DataSetter(config), config.pipeline.models, config, progress_callback=callback
        )
        assert callback.of("start_target")[0][1]["total_folds"] == 6
        assert {args[1] for args, _ in callback.of("start_fold")} == {6}
        assert len(callback.of("complete_fold")) == 6

        # A second run loads the saved models and must skip all six iterations at once
        second = _RecordingCallback()
        perform_pipeline(
            DataSetter(config), config.pipeline.models, config, progress_callback=second
        )
        assert second.of("skip_target")[0][0][1] == 6


class TestCvThresholdWithGroups:
    """The cv threshold method must work with a group column (sklearn rejects groups= on the tuner)."""

    @pytest.mark.parametrize("calibrate_probabilities", ["false", "true"])
    def test_run_tunes_threshold_on_group_aware_splits(self, tmp_path, calibrate_probabilities):
        data_path = _write_data(tmp_path, "d", groups=True)
        config = _write_config(
            tmp_path,
            data_path,
            f"cv_threshold_{calibrate_probabilities}",
            groups=True,
            save_models=True,
            pipeline={
                "calibrate_threshold": "true",
                "threshold_method": "cv",
                "calibrate_probabilities": calibrate_probabilities,
            },
        )
        perform_pipeline(DataSetter(config), config.pipeline.models, config)

        summary = pd.read_csv(
            Path(config.output.out_folder) / "metrics" / "y" / "LR_metrics_detailed.csv"
        ).set_index("Metric")
        assert not np.isnan(summary.loc["AUROC", "Mean"])

        checkpoint = joblib.load(get_model_path(config.output.out_folder, "LR", "y"))
        assert all(model is not None for model in checkpoint["fold_models"])
        assert all(0.0 <= t <= 1.0 for t in checkpoint["fold_thresholds"])
        assert not any(
            isinstance(model, TunedThresholdClassifierCV) for model in checkpoint["fold_models"]
        )

    def test_train_with_cv_threshold_and_groups(self, tmp_path):
        data_path = _write_data(tmp_path, "d", groups=True)
        config = _write_config(
            tmp_path,
            data_path,
            "train_cv_threshold",
            groups=True,
            pipeline={
                "calibrate_threshold": "true",
                "threshold_method": "cv",
                "calibrate_probabilities": "true",
            },
        )
        perform_training(DataSetter(config), config.pipeline.models, config)

        bundle_path = Path(config.output.out_folder) / "trained_models" / "LR_y.joblib"
        bundle = joblib.load(bundle_path)
        assert 0.0 <= bundle["threshold"] <= 1.0
        metadata = json.loads((bundle_path.parent / "training_metadata.json").read_text())
        assert metadata["config"]["threshold_method"] == "cv"
