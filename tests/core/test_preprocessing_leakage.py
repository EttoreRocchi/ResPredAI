"""Tests that per-fold one-hot encoding is fit on training data only."""

import pandas as pd

from respredai.core.workflow import _apply_ohe_and_clean, _build_ohe_transformer


class TestOheFitOnTrainOnly:
    """The OHE must derive its columns from the training split only."""

    def test_test_only_category_not_leaked(self):
        # Train has categories A, B; test introduces an unseen C. Fitting on train
        # only means no 'cat_C' column appears, and test columns align exactly to
        # the train-derived columns (the no-leakage guarantee).
        X_train = pd.DataFrame({"cat": ["A", "B", "A", "B"], "num": [1.0, 2.0, 3.0, 4.0]})
        X_test = pd.DataFrame({"cat": ["A", "C"], "num": [5.0, 6.0]})

        ohe = _build_ohe_transformer(["cat"])
        X_train_ohe, X_test_ohe = _apply_ohe_and_clean(ohe, X_train, X_test)

        # Columns come from train only and match exactly across train/test
        assert list(X_train_ohe.columns) == list(X_test_ohe.columns)
        # The unseen test-only category must not create a column
        assert "cat_C" not in X_test_ohe.columns
        # Rows are preserved
        assert len(X_test_ohe) == 2
