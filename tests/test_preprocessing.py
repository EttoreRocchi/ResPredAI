"""Unit tests for preprocessing consistency and data handling."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


class TestPreprocessingConsistency:
    """Tests to ensure preprocessing is consistent across pipeline functions."""

    def test_one_hot_encoding_consistency(self):
        """Test that OneHotEncoder produces consistent encoding."""
        # Simulate the categorical encoding used in perform_pipeline
        categorical_cols = ["cat1", "cat2"]

        df = pd.DataFrame(
            {
                "cat1": ["A", "B", "A", "C"],
                "cat2": ["X", "Y", "X", "Y"],
                "continuous": [1.0, 2.0, 3.0, 4.0],
            }
        )

        ohe_transformer = ColumnTransformer(
            transformers=[
                (
                    "ohe",
                    OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"),
                    categorical_cols,
                )
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

        encoded = ohe_transformer.fit_transform(df)

        # Check that all categories are present
        assert "cat1_A" in encoded.columns
        assert "cat1_B" in encoded.columns
        assert "cat1_C" in encoded.columns
        assert "cat2_X" in encoded.columns
        assert "cat2_Y" in encoded.columns
        assert "continuous" in encoded.columns

    def test_feature_name_sanitization(self):
        """Test that feature names are sanitized consistently."""
        # Create column names with < and > characters
        df = pd.DataFrame(
            {
                "age<30": [1, 0, 1],
                "bmi>25": [0, 1, 1],
                "normal_col": [1, 2, 3],
            }
        )

        # Apply same sanitization as in pipeline
        df.columns = df.columns.str.replace("<", "_lt_", regex=False)
        df.columns = df.columns.str.replace(">", "_gt_", regex=False)

        assert "age_lt_30" in df.columns
        assert "bmi_gt_25" in df.columns
        assert "normal_col" in df.columns

    def test_categorical_identification(self):
        """Test that categorical columns are correctly identified."""
        continuous_features = ["age", "bmi"]

        df = pd.DataFrame(
            {
                "age": [30, 40, 50],
                "bmi": [20.1, 25.3, 30.0],
                "sex": ["M", "F", "M"],
                "ward": ["ICU", "General", "ICU"],
            }
        )

        categorical_cols = [col for col in df.columns if col not in continuous_features]

        assert "sex" in categorical_cols
        assert "ward" in categorical_cols
        assert "age" not in categorical_cols
        assert "bmi" not in categorical_cols

    def test_one_hot_encoding_unknown_category(self):
        """Test that unknown categories are handled during transform."""
        categorical_cols = ["category"]

        train_df = pd.DataFrame({"category": ["A", "B", "A"], "value": [1, 2, 3]})

        test_df = pd.DataFrame(
            {
                "category": ["A", "C"],  # C is unknown
                "value": [4, 5],
            }
        )

        ohe_transformer = ColumnTransformer(
            transformers=[
                (
                    "ohe",
                    OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"),
                    categorical_cols,
                )
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        ).set_output(transform="pandas")

        ohe_transformer.fit(train_df)
        encoded_test = ohe_transformer.transform(test_df)

        # Unknown category should be encoded as all zeros
        assert "category_A" in encoded_test.columns
        assert "category_B" in encoded_test.columns
        # Row with category C should have all zeros for categorical columns
        assert encoded_test.loc[1, "category_A"] == 0.0
        assert encoded_test.loc[1, "category_B"] == 0.0

    def test_pd_get_dummies_vs_ohe_consistency(self):
        """Test that pd.get_dummies with drop_first=False matches OneHotEncoder drop=None."""
        df = pd.DataFrame({"category": ["A", "B", "A", "C"]})

        # Method 1: OneHotEncoder
        ohe = OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore")
        _ = ohe.fit_transform(df[["category"]])
        ohe_columns = [f"category_{cat}" for cat in ohe.categories_[0]]

        # Method 2: pd.get_dummies
        dummies_result = pd.get_dummies(df, columns=["category"], drop_first=False)

        # Both should have same number of columns for category
        assert len(ohe_columns) == len(
            [c for c in dummies_result.columns if c.startswith("category")]
        )

        # Values should match (after sorting columns)
        for col in ohe_columns:
            assert col in dummies_result.columns


class TestDataValidation:
    """Tests for data validation and edge cases."""

    def test_empty_continuous_features(self):
        """Test handling when no continuous features specified."""
        continuous_features = []
        df = pd.DataFrame({"cat1": ["A", "B"], "cat2": ["X", "Y"]})

        categorical_cols = [col for col in df.columns if col not in continuous_features]

        assert len(categorical_cols) == 2
        assert "cat1" in categorical_cols
        assert "cat2" in categorical_cols

    def test_all_continuous_features(self):
        """Test handling when all features are continuous."""
        df = pd.DataFrame({"age": [30, 40], "bmi": [20.0, 25.0], "temp": [36.5, 37.0]})
        continuous_features = list(df.columns)

        categorical_cols = [col for col in df.columns if col not in continuous_features]

        assert len(categorical_cols) == 0

    def test_missing_values_detection(self):
        """Test that missing values can be detected."""
        df = pd.DataFrame({"col1": [1, np.nan, 3], "col2": [4, 5, 6]})

        has_missing = df.isna().any().any()
        missing_cols = df.columns[df.isna().any()].tolist()

        assert has_missing
        assert "col1" in missing_cols
        assert "col2" not in missing_cols

    def test_binary_target_validation(self):
        """Test validation of binary target column."""
        valid_target = pd.Series([0, 1, 0, 1, 1])
        invalid_target = pd.Series([0, 1, 2, 1, 0])

        assert set(valid_target.unique()) == {0, 1}
        assert set(invalid_target.unique()) != {0, 1}

    def test_feature_column_alignment(self):
        """Test alignment of columns between train and test data."""
        train_columns = ["feat_A", "feat_B", "feat_C"]
        test_df = pd.DataFrame({"feat_A": [1], "feat_D": [2]})  # Missing B, C; extra D

        # Add missing columns as 0
        for col in train_columns:
            if col not in test_df.columns:
                test_df[col] = 0

        # Reorder to match training
        test_df = test_df[train_columns]

        assert list(test_df.columns) == train_columns
        assert test_df["feat_B"].iloc[0] == 0
        assert test_df["feat_C"].iloc[0] == 0


class TestGroupHandling:
    """Tests for group-based stratification."""

    def test_group_column_extraction(self):
        """Test that group column is correctly extracted."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4],
                "feature2": [5, 6, 7, 8],
                "group_id": [1, 1, 2, 2],
                "target": [0, 1, 0, 1],
            }
        )

        group_column = "group_id"
        groups = df[group_column]
        X = df.drop(columns=[group_column, "target"])

        assert len(groups) == 4
        assert group_column not in X.columns
        assert list(groups.unique()) == [1, 2]

    def test_stratified_group_split(self):
        """Test that groups are respected in stratified split."""
        from sklearn.model_selection import StratifiedGroupKFold

        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]])
        y = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        groups = np.array([1, 1, 1, 1, 2, 2, 2, 2])

        sgkf = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)

        for train_idx, test_idx in sgkf.split(X, y, groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            # Groups should not overlap between train and test
            assert len(train_groups & test_groups) == 0
