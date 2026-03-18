"""Unit tests for CV utilities module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from respredai.core.cv_utils import RepeatedStratifiedGroupKFold, get_outer_cv, get_temporal_split


class TestRepeatedStratifiedGroupKFold:
    """Unit tests for RepeatedStratifiedGroupKFold cross-validator."""

    def test_correct_number_of_splits(self):
        """Should return n_splits * n_repeats iterations."""
        cv = RepeatedStratifiedGroupKFold(n_splits=3, n_repeats=2, random_state=42)
        X = np.arange(30).reshape(-1, 1)
        y = np.array([0] * 15 + [1] * 15)
        groups = np.array([i // 3 for i in range(30)])  # 10 groups

        splits = list(cv.split(X, y, groups))
        assert len(splits) == 6  # 3 * 2

    def test_get_n_splits(self):
        """get_n_splits should return correct count."""
        cv = RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=3, random_state=42)
        assert cv.get_n_splits() == 15

    def test_groups_kept_together(self):
        """All samples from same group should be in same fold."""
        cv = RepeatedStratifiedGroupKFold(n_splits=3, n_repeats=2, random_state=42)
        X = np.arange(30).reshape(-1, 1)
        y = np.array([0] * 15 + [1] * 15)
        groups = np.array([i // 3 for i in range(30)])  # 10 groups

        for train_idx, test_idx in cv.split(X, y, groups):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            # No group should appear in both train and test
            assert len(train_groups & test_groups) == 0

    def test_reproducibility(self):
        """Same random_state should produce identical splits."""
        X = np.arange(30).reshape(-1, 1)
        y = np.array([0] * 15 + [1] * 15)
        groups = np.array([i // 3 for i in range(30)])

        cv1 = RepeatedStratifiedGroupKFold(n_splits=3, n_repeats=2, random_state=42)
        cv2 = RepeatedStratifiedGroupKFold(n_splits=3, n_repeats=2, random_state=42)

        splits1 = list(cv1.split(X, y, groups))
        splits2 = list(cv2.split(X, y, groups))

        for (train1, test1), (train2, test2) in zip(splits1, splits2):
            assert np.array_equal(train1, train2)
            assert np.array_equal(test1, test2)

    def test_different_random_states_produce_different_splits(self):
        """Different random_states should produce different splits."""
        X = np.arange(30).reshape(-1, 1)
        y = np.array([0] * 15 + [1] * 15)
        groups = np.array([i // 3 for i in range(30)])

        cv1 = RepeatedStratifiedGroupKFold(n_splits=3, n_repeats=2, random_state=42)
        cv2 = RepeatedStratifiedGroupKFold(n_splits=3, n_repeats=2, random_state=123)

        splits1 = list(cv1.split(X, y, groups))
        splits2 = list(cv2.split(X, y, groups))

        # At least some splits should be different
        any_different = False
        for (train1, _), (train2, _) in zip(splits1, splits2):
            if not np.array_equal(train1, train2):
                any_different = True
                break
        assert any_different

    def test_raises_without_groups(self):
        """Should raise ValueError if groups is None."""
        cv = RepeatedStratifiedGroupKFold(n_splits=3, n_repeats=2, random_state=42)
        X = np.arange(30).reshape(-1, 1)
        y = np.array([0] * 15 + [1] * 15)

        with pytest.raises(ValueError, match="groups must be provided"):
            list(cv.split(X, y, groups=None))

    def test_each_repeat_is_different(self):
        """Different repeats should not produce identical test folds."""
        cv = RepeatedStratifiedGroupKFold(n_splits=3, n_repeats=2, random_state=42)
        X = np.arange(30).reshape(-1, 1)
        y = np.array([0] * 15 + [1] * 15)
        groups = np.array([i // 3 for i in range(30)])

        splits = list(cv.split(X, y, groups))
        # First repeat: splits[0:3], Second repeat: splits[3:6]
        first_repeat = [set(test_idx) for _, test_idx in splits[:3]]
        second_repeat = [set(test_idx) for _, test_idx in splits[3:]]

        # The splits within each repeat should be different
        # (unless by chance they're the same, which is unlikely)
        # More robust: check that at least one pair differs
        any_different = False
        for s1, s2 in zip(first_repeat, second_repeat):
            if s1 != s2:
                any_different = True
                break
        # Note: This could theoretically fail if random seeds align perfectly
        # but that's extremely unlikely
        assert any_different


class TestGetOuterCV:
    """Unit tests for get_outer_cv factory function."""

    def test_standard_stratified_kfold(self):
        """Should return StratifiedKFold for n_repeats=1, no groups."""
        cv = get_outer_cv(n_splits=5, n_repeats=1, use_groups=False, random_state=42)
        assert isinstance(cv, StratifiedKFold)
        assert cv.n_splits == 5

    def test_standard_stratified_group_kfold(self):
        """Should return StratifiedGroupKFold for n_repeats=1, with groups."""
        cv = get_outer_cv(n_splits=5, n_repeats=1, use_groups=True, random_state=42)
        assert isinstance(cv, StratifiedGroupKFold)
        assert cv.n_splits == 5

    def test_repeated_stratified_kfold(self):
        """Should return RepeatedStratifiedKFold for n_repeats>1, no groups."""
        cv = get_outer_cv(n_splits=5, n_repeats=3, use_groups=False, random_state=42)
        assert isinstance(cv, RepeatedStratifiedKFold)

    def test_repeated_stratified_group_kfold(self):
        """Should return RepeatedStratifiedGroupKFold for n_repeats>1, with groups."""
        cv = get_outer_cv(n_splits=5, n_repeats=3, use_groups=True, random_state=42)
        assert isinstance(cv, RepeatedStratifiedGroupKFold)

    def test_random_state_passed_correctly(self):
        """Random state should be passed to the CV object."""
        cv = get_outer_cv(n_splits=5, n_repeats=1, use_groups=False, random_state=123)
        assert cv.random_state == 123

    def test_shuffle_enabled(self):
        """Shuffle should be enabled for non-repeated CV."""
        cv = get_outer_cv(n_splits=5, n_repeats=1, use_groups=False, random_state=42)
        assert cv.shuffle is True

    def test_repeated_cv_produces_more_splits(self):
        """Repeated CV should produce more splits than standard CV."""
        X = np.arange(100).reshape(-1, 1)
        y = np.array([0] * 50 + [1] * 50)

        cv_standard = get_outer_cv(n_splits=5, n_repeats=1, use_groups=False)
        cv_repeated = get_outer_cv(n_splits=5, n_repeats=3, use_groups=False)

        splits_standard = list(cv_standard.split(X, y))
        splits_repeated = list(cv_repeated.split(X, y))

        assert len(splits_standard) == 5
        assert len(splits_repeated) == 15


class TestGetTemporalSplit:
    """Unit tests for get_temporal_split function."""

    def test_split_by_date(self):
        """Split with a date cutoff produces correct indices."""
        dates = pd.Series(pd.to_datetime(["2022-01", "2022-06", "2023-01", "2023-06"]))
        train_idx, test_idx = get_temporal_split(dates, split_date="2023-01-01")

        np.testing.assert_array_equal(sorted(train_idx), [0, 1])
        np.testing.assert_array_equal(sorted(test_idx), [2, 3])

    def test_split_by_ratio(self):
        """Split with a ratio produces correct proportions."""
        dates = pd.Series(pd.to_datetime([f"2020-{i:02d}-01" for i in range(1, 11)]))
        train_idx, test_idx = get_temporal_split(dates, split_ratio=0.5)

        assert len(train_idx) > 0
        assert len(test_idx) > 0
        assert len(train_idx) + len(test_idx) == 10

    def test_group_aware_split(self):
        """Groups spanning the boundary are assigned to test."""
        dates = pd.Series(pd.to_datetime(["2022-01", "2022-06", "2023-06", "2022-03"]))
        # Group 0: samples 0, 3 (dates 2022-01, 2022-03) -> all before cutoff -> train
        # Group 1: samples 1, 2 (dates 2022-06, 2023-06) -> max date is after cutoff -> test
        groups = np.array([0, 1, 1, 0])

        train_idx, test_idx = get_temporal_split(dates, split_date="2023-01-01", groups=groups)

        # Group 0 (samples 0, 3) should be in train
        assert 0 in train_idx
        assert 3 in train_idx
        # Group 1 (samples 1, 2) should be in test (max date 2023-06 >= cutoff)
        assert 1 in test_idx
        assert 2 in test_idx

    def test_empty_train_raises(self):
        """Split that produces empty train set raises ValueError."""
        dates = pd.Series(pd.to_datetime(["2023-01", "2023-06"]))

        with pytest.raises(ValueError, match="empty training set"):
            get_temporal_split(dates, split_date="2020-01-01")

    def test_empty_test_raises(self):
        """Split that produces empty test set raises ValueError."""
        dates = pd.Series(pd.to_datetime(["2020-01", "2020-06"]))

        with pytest.raises(ValueError, match="empty test set"):
            get_temporal_split(dates, split_date="2025-01-01")

    def test_both_date_and_ratio_raises(self):
        """Providing both split_date and split_ratio raises ValueError."""
        dates = pd.Series(pd.to_datetime(["2022-01", "2023-01"]))

        with pytest.raises(ValueError, match="Exactly one"):
            get_temporal_split(dates, split_date="2022-06-01", split_ratio=0.5)

    def test_neither_date_nor_ratio_raises(self):
        """Providing neither split_date nor split_ratio raises ValueError."""
        dates = pd.Series(pd.to_datetime(["2022-01", "2023-01"]))

        with pytest.raises(ValueError, match="Exactly one"):
            get_temporal_split(dates)

    def test_imbalanced_split_warns(self):
        """Highly imbalanced split produces a warning."""
        # 20 dates, cutoff after 1st -> 1 train / 19 test = 5% train
        dates = pd.Series(
            pd.to_datetime(
                [f"2020-{i:02d}-01" for i in range(1, 13)]
                + [f"2021-{i:02d}-01" for i in range(1, 9)]
            )
        )

        with pytest.warns(UserWarning, match="highly imbalanced"):
            get_temporal_split(dates, split_date="2020-02-01")

    def test_returns_numpy_arrays(self):
        """Returned indices are numpy arrays."""
        dates = pd.Series(pd.to_datetime(["2022-01", "2022-06", "2023-01", "2023-06"]))
        train_idx, test_idx = get_temporal_split(dates, split_date="2023-01-01")

        assert isinstance(train_idx, np.ndarray)
        assert isinstance(test_idx, np.ndarray)
