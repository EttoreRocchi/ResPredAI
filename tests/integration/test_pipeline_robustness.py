"""Integration regressions for pipeline failure-surfacing, leakage, and determinism."""

from pathlib import Path
from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

import respredai.core.workflow as workflow_module
from respredai.core.workflow import perform_pipeline
from respredai.io.config import ConfigHandler, DataSetter

pytestmark = [
    pytest.mark.slow,
    pytest.mark.filterwarnings(
        "ignore::sklearn.exceptions.ConvergenceWarning",
        "ignore:.*Mean of empty slice.*:RuntimeWarning",
        "ignore:.*Degrees of freedom.*:RuntimeWarning",
        "ignore:.*Bootstrap CI.*:UserWarning",
        "ignore:.*failed for.*:UserWarning",
        "ignore:.*All folds failed.*:UserWarning",
    ),
]


def _make_data(n=160, seed=42, signal=True):
    rng = np.random.RandomState(seed)
    age = rng.uniform(20, 80, n)
    bmi = rng.uniform(18, 35, n)
    sex = rng.choice(["M", "F"], n)
    if signal:
        score = (age - 50) / 30 + (bmi - 25) / 10 + rng.normal(0, 0.3, n)
        y = (score > 0).astype(int)
    else:
        y = rng.randint(0, 2, n)
    y[0], y[1] = 0, 1  # guarantee both classes
    return pd.DataFrame({"age": age, "bmi": bmi, "sex": sex, "y": y})


def _config(tmp_path, data_path, out_name="out", seed=42):
    config_text = dedent(f"""
    [Data]
    data_path = {data_path}
    targets = y
    continuous_features = age, bmi

    [Pipeline]
    models = LR
    outer_folds = 3
    inner_folds = 2
    calibrate_threshold = false

    [Reproducibility]
    seed = {seed}

    [Log]
    verbosity = 0
    log_basename = test.log

    [Resources]
    n_jobs = 1

    [Output]
    out_folder = {tmp_path / out_name}

    [ModelSaving]
    enable = false
    """).strip()
    p = tmp_path / f"cfg_{out_name}.ini"
    p.write_text(config_text)
    return ConfigHandler(str(p))


def _auroc_mean(out_folder):
    csv = Path(out_folder) / "metrics" / "y" / "LR_metrics_detailed.csv"
    df = pd.read_csv(csv).set_index("Metric")
    return float(df.loc["AUROC", "Mean"])


class TestFailureSurfacing:
    def test_all_folds_failing_raises(self, tmp_path, monkeypatch):
        df = _make_data()
        data_path = tmp_path / "d.csv"
        df.to_csv(data_path, index=False)
        config = _config(tmp_path, data_path)
        datasetter = DataSetter(config)

        def _boom(*args, **kwargs):
            raise RuntimeError("forced failure")

        # Force every fold's metric computation to fail.
        monkeypatch.setattr(workflow_module, "metric_dict", _boom)

        with pytest.raises(RuntimeError, match="Training failed"):
            perform_pipeline(
                datasetter=datasetter,
                models=config.pipeline.models,
                config_handler=config,
            )


class TestLeakageAndDeterminism:
    def test_shuffled_labels_give_chance_auroc(self, tmp_path):
        df = _make_data(signal=True)
        rng = np.random.RandomState(0)
        df["y"] = rng.permutation(df["y"].values)  # destroy any signal
        df.loc[0, "y"], df.loc[1, "y"] = 0, 1
        data_path = tmp_path / "d.csv"
        df.to_csv(data_path, index=False)
        config = _config(tmp_path, data_path)
        perform_pipeline(
            datasetter=DataSetter(config),
            models=config.pipeline.models,
            config_handler=config,
        )
        # A leak-free pipeline scores near chance on shuffled labels.
        assert 0.3 <= _auroc_mean(config.output.out_folder) <= 0.7

    def test_signal_data_beats_chance(self, tmp_path):
        df = _make_data(signal=True)
        data_path = tmp_path / "d.csv"
        df.to_csv(data_path, index=False)
        config = _config(tmp_path, data_path)
        perform_pipeline(
            datasetter=DataSetter(config),
            models=config.pipeline.models,
            config_handler=config,
        )
        assert _auroc_mean(config.output.out_folder) > 0.7

    def test_determinism_same_seed(self, tmp_path):
        df = _make_data(signal=True)
        data_path = tmp_path / "d.csv"
        df.to_csv(data_path, index=False)

        c1 = _config(tmp_path, data_path, out_name="run1")
        perform_pipeline(datasetter=DataSetter(c1), models=c1.pipeline.models, config_handler=c1)
        c2 = _config(tmp_path, data_path, out_name="run2")
        perform_pipeline(datasetter=DataSetter(c2), models=c2.pipeline.models, config_handler=c2)

        csv1 = (
            Path(c1.output.out_folder) / "metrics" / "y" / "LR_metrics_detailed.csv"
        ).read_text()
        csv2 = (
            Path(c2.output.out_folder) / "metrics" / "y" / "LR_metrics_detailed.csv"
        ).read_text()
        assert csv1 == csv2


class TestCalibrationWithGroups:
    def test_calibrate_probabilities_with_groups_non_nan(self, tmp_path):
        # Regression: calibrate_probabilities + group_column together used to make
        # every fold fail ("indices are out-of-bounds") in the conformal step,
        # silently yielding empty metrics. Assert real (non-NaN) metrics now.
        rng = np.random.RandomState(7)
        n = 160
        age = rng.uniform(20, 80, n)
        bmi = rng.uniform(18, 35, n)
        score = (age - 50) / 30 + (bmi - 25) / 10 + rng.normal(0, 0.3, n)
        y = (score > 0).astype(int)
        y[0], y[1] = 0, 1
        pid = rng.randint(1, n // 2 + 1, n)  # repeated patients within groups
        df = pd.DataFrame({"age": age, "bmi": bmi, "patient_id": pid, "y": y})
        data_path = tmp_path / "d.csv"
        df.to_csv(data_path, index=False)

        config_text = dedent(f"""
        [Data]
        data_path = {data_path}
        targets = y
        continuous_features = age, bmi

        [Metadata]
        group_column = patient_id

        [Pipeline]
        models = LR
        outer_folds = 3
        inner_folds = 2
        calibrate_probabilities = true
        probability_calibration_method = sigmoid
        probability_calibration_cv = 3

        [Reproducibility]
        seed = 42

        [Log]
        verbosity = 0
        log_basename = test.log

        [Resources]
        n_jobs = 1

        [Output]
        out_folder = {tmp_path / "out"}

        [ModelSaving]
        enable = false
        """).strip()
        p = tmp_path / "cfg.ini"
        p.write_text(config_text)
        config = ConfigHandler(str(p))
        perform_pipeline(
            datasetter=DataSetter(config),
            models=config.pipeline.models,
            config_handler=config,
        )
        assert not np.isnan(_auroc_mean(config.output.out_folder))
