"""Tests for SHAP shape handling and feature-importance aggregation."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from respredai.visualization.feature_importance import (
    _importance_type_label,
    _shap_to_class1,
    compute_feature_direction,
    save_feature_importance_csv,
)


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
