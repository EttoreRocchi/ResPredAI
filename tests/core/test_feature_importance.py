"""Tests for SHAP shape handling and feature-importance aggregation."""

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TunedThresholdClassifierCV

from respredai.core.models import get_model_path
from respredai.visualization.feature_importance import (
    _importance_type_label,
    _shap_to_class1,
    compute_feature_direction,
    compute_shap_importance,
    extract_feature_importance_from_models,
    get_feature_importance,
    get_linear_intercept,
    iter_inner_estimators,
    process_feature_importance,
    save_feature_importance_csv,
    save_feature_importance_per_fold_csv,
    unwrap_calibrated_model,
)


def _make_binary_data(n: int = 200, seed: int = 0):
    """Binary problem with one risk, one protective and one weak feature."""
    rng = np.random.RandomState(seed)
    names = ["risk", "protect", "weak"]
    X = pd.DataFrame(rng.randn(n, 3), columns=names)
    y = ((X["risk"] * 2 - X["protect"] * 1.5 + rng.randn(n) * 0.3) > 0).astype(int)
    return X, y, names


def _calibrated_lr(X, y, cv: int = 3, seed: int = 0) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, random_state=seed), cv=cv, method="sigmoid"
    ).fit(X, y)


class TestShapToClass1:
    """Reducing SHAP output of any shape to a 2D positive-class array."""

    def test_list_input_selects_class1(self):
        sv = [np.zeros((5, 3)), np.ones((5, 3))]
        out = _shap_to_class1(sv)
        assert out.shape == (5, 3)
        assert np.allclose(out, 1.0)

    def test_3d_input_selects_class1(self):
        arr = np.zeros((5, 3, 2))
        arr[:, :, 1] = 2.0
        out = _shap_to_class1(arr)
        assert out.shape == (5, 3)
        assert np.allclose(out, 2.0)

    def test_2d_input_passthrough(self):
        arr = np.arange(15).reshape(5, 3).astype(float)
        out = _shap_to_class1(arr)
        assert out.shape == (5, 3)
        assert np.allclose(out, arr)


class TestRandomForestDirection:
    """Regression guard: RandomForest feature direction must not silently fail.

    RandomForest direction used to silently return None because the 3-D
    TreeExplainer output of newer SHAP versions was not handled.
    """

    def test_rf_direction_not_none(self):
        rng = np.random.RandomState(0)
        names = ["risk", "protect", "weak"]
        X = pd.DataFrame(rng.randn(150, 3), columns=names)
        y = ((X["risk"] * 2 - X["protect"] * 1.5 + rng.randn(150) * 0.3) > 0).astype(int)
        model = RandomForestClassifier(n_estimators=40, random_state=0).fit(X, y)

        result = compute_feature_direction(model, X.values, names, model_name="RF")
        assert result is not None
        assert len(result) == 3
        assert result.notna().all()


class TestImportanceTypeLabel:
    def test_labels(self):
        assert _importance_type_label("RF", "native") == "impurity (MDI)"
        assert _importance_type_label("XGB", "native") == "gain"
        assert _importance_type_label("LR", "native") == "coefficient"
        assert _importance_type_label("MLP", "shap") == "mean_abs_shap"


class TestSaveImportanceCsv:
    """NaN-skip aggregation + Importance_Type / N_folds_present columns."""

    def test_nan_skip_and_columns(self, tmp_path):
        # Feature 'b' is absent (NaN) in fold 0; NaN-skip averages only the folds
        # where it is present, so its mean is 4.0 (not 2.0 from fillna-0).
        df = pd.DataFrame(
            [
                {"a": 1.0, "b": np.nan, "c": 0.5},
                {"a": 3.0, "b": 4.0, "c": 0.5},
            ]
        )
        out = tmp_path / "imp.csv"
        save_feature_importance_csv(df, out, method="native", importance_type="gain")
        saved = pd.read_csv(out).set_index("Feature")

        assert "Importance_Type" in saved.columns
        assert "N_folds_present" in saved.columns
        assert (saved["Importance_Type"] == "gain").all()
        assert saved.loc["b", "N_folds_present"] == 1
        assert saved.loc["a", "N_folds_present"] == 2
        assert np.isclose(saved.loc["b", "Mean_Importance"], 4.0)


class TestInnerEstimators:
    """Walking calibration wrappers must expose every inner estimator."""

    def test_plain_model_yields_itself(self):
        X, y, _ = _make_binary_data()
        lr = LogisticRegression(max_iter=1000).fit(X, y)
        assert list(iter_inner_estimators(lr)) == [lr]
        assert unwrap_calibrated_model(lr) is lr

    def test_nested_wrappers_yield_every_sub_model(self):
        X, y, _ = _make_binary_data()
        calibrated = CalibratedClassifierCV(
            LogisticRegression(max_iter=1000), cv=3, method="sigmoid"
        )
        tuned = TunedThresholdClassifierCV(calibrated, cv=2).fit(X, y)

        inner = list(iter_inner_estimators(tuned))
        assert len(inner) == 3
        assert all(isinstance(m, LogisticRegression) for m in inner)
        assert unwrap_calibrated_model(tuned) is inner[0]


class TestLinearCoefficients:
    """Binary linear coefficients must keep their sign and cover all sub-models."""

    def test_binary_lr_coefficients_keep_sign(self):
        X, y, names = _make_binary_data()
        lr = LogisticRegression(max_iter=1000).fit(X, y)

        imp = get_feature_importance(lr, names, model_name="LR")
        assert imp["risk"] > 0
        assert imp["protect"] < 0
        np.testing.assert_allclose(imp.values, lr.coef_[0])
        np.testing.assert_allclose(get_linear_intercept(lr, "LR"), lr.intercept_[0])

    def test_calibrated_lr_averages_all_sub_models(self):
        X, y, names = _make_binary_data()
        model = _calibrated_lr(X, y, cv=4)
        subs = [cc.estimator for cc in model.calibrated_classifiers_]
        expected_coef = np.mean([s.coef_[0] for s in subs], axis=0)
        expected_intercept = np.mean([s.intercept_[0] for s in subs])

        imp = get_feature_importance(model, names, model_name="LR")
        np.testing.assert_allclose(imp.values, expected_coef)
        assert imp["protect"] < 0
        # sub-models are fit on different subsets, so the first one alone differs
        assert not np.allclose(subs[0].coef_[0], expected_coef)
        np.testing.assert_allclose(get_linear_intercept(model, "LR"), expected_intercept)

    def test_direction_uses_all_sub_models(self):
        X, y, names = _make_binary_data()
        model = _calibrated_lr(X, y, cv=4)
        expected = np.mean([cc.estimator.coef_[0] for cc in model.calibrated_classifiers_], axis=0)

        signed = compute_feature_direction(model, X.values, names, model_name="LR")
        np.testing.assert_allclose(signed.values, expected)

    def test_intercept_none_for_non_linear(self):
        X, y, _ = _make_binary_data()
        rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, y)
        assert get_linear_intercept(rf, "RF") is None
        assert get_linear_intercept(None, "LR") is None


class TestTreeImportanceAveraging:
    """Tree importances and TreeExplainer SHAP average over calibrated sub-models."""

    def test_calibrated_rf_averages_sub_models(self):
        X, y, names = _make_binary_data(n=150)
        model = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=15, random_state=0), cv=3
        ).fit(X, y)
        subs = [cc.estimator for cc in model.calibrated_classifiers_]
        expected = np.mean([s.feature_importances_ for s in subs], axis=0)

        imp = get_feature_importance(model, names, model_name="RF")
        np.testing.assert_allclose(imp.values, expected)
        assert not np.allclose(subs[0].feature_importances_, expected)

    def test_calibrated_rf_shap_averages_sub_models(self):
        import shap

        X, y, names = _make_binary_data(n=120)
        model = CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=10, random_state=0), cv=2
        ).fit(X, y)
        X_test = X.values[:20]
        per_sub = [
            _shap_to_class1(
                shap.TreeExplainer(cc.estimator).shap_values(pd.DataFrame(X_test, columns=names))
            )
            for cc in model.calibrated_classifiers_
        ]
        expected = np.abs(np.mean(per_sub, axis=0)).mean(axis=0)

        imp = compute_shap_importance(model, X_test, names, model_name="RF")
        np.testing.assert_allclose(imp.values, expected)


class TestPerFoldCsv:
    def test_per_fold_csv_columns(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, 3.0], "b": [-0.5, -0.7]}, index=pd.Index([1, 3], name="Fold"))
        intercepts = pd.Series([0.1, 0.2], index=df.index)
        out = tmp_path / "per_fold.csv"

        save_feature_importance_per_fold_csv(df, out, intercepts=intercepts)

        saved = pd.read_csv(out)
        assert list(saved.columns) == ["Fold", "Intercept", "a", "b"]
        assert saved["Fold"].tolist() == [1, 3]
        np.testing.assert_allclose(saved["Intercept"], [0.1, 0.2])
        np.testing.assert_allclose(saved["b"], [-0.5, -0.7])

    def test_per_fold_csv_without_intercept(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, 3.0]})
        out = tmp_path / "per_fold.csv"

        save_feature_importance_per_fold_csv(df, out)

        saved = pd.read_csv(out)
        assert list(saved.columns) == ["Fold", "a"]
        assert saved["Fold"].tolist() == [1, 2]


class TestExtractFromSavedModels:
    """End-to-end on a saved model file shaped like the pipeline output."""

    @staticmethod
    def _save_model_file(path, X, y, names):
        fold_models, fold_test_data = [], []
        for seed in range(2):
            fold_models.append(_calibrated_lr(X, y, cv=3, seed=seed))
            fold_test_data.append((X.values[:40], names))
        # a failed fold in the middle must be skipped without shifting fold numbers
        fold_models.insert(1, None)
        fold_test_data.insert(1, None)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "fold_models": fold_models,
                "fold_test_data": fold_test_data,
                "fold_transformers": [None] * len(fold_models),
            },
            path,
        )
        return fold_models

    def test_calibrated_lr_signed_and_per_fold(self, tmp_path):
        X, y, names = _make_binary_data(n=240)
        path = tmp_path / "LR_T_models.joblib"
        fold_models = self._save_model_file(path, X, y, names)

        result = extract_feature_importance_from_models(path, model_name="LR", use_shap=False)

        assert result is not None
        importances_df, feature_names, method, directions, intercepts = result
        assert method == "native"
        assert directions is None
        assert importances_df.index.tolist() == [1, 3]
        assert set(feature_names) == set(names)
        assert (importances_df["risk"] > 0).all()
        assert (importances_df["protect"] < 0).all()
        assert intercepts is not None
        assert intercepts.index.tolist() == [1, 3]

        expected_fold1 = np.mean(
            [cc.estimator.coef_[0] for cc in fold_models[0].calibrated_classifiers_], axis=0
        )
        np.testing.assert_allclose(importances_df.loc[1, names].values, expected_fold1)

    def test_process_writes_summary_and_per_fold_csv(self, tmp_path):
        X, y, names = _make_binary_data(n=240)
        path = get_model_path(str(tmp_path), "LR", "T")
        self._save_model_file(path, X, y, names)

        result = process_feature_importance(str(tmp_path), "LR", "T", save_plot=False)

        assert result is not None
        fi_dir = tmp_path / "feature_importance" / "T"
        summary = pd.read_csv(fi_dir / "LR_feature_importance.csv").set_index("Feature")
        assert summary.loc["protect", "Mean_Importance"] < 0
        assert summary.loc["risk", "Mean_Importance"] > 0
        assert (summary["Importance_Type"] == "coefficient").all()

        per_fold = pd.read_csv(fi_dir / "LR_feature_importance_per_fold.csv")
        assert per_fold["Fold"].tolist() == [1, 3]
        assert list(per_fold.columns[:2]) == ["Fold", "Intercept"]
        assert set(per_fold.columns[2:]) == set(names)
        assert (per_fold["protect"] < 0).all()
