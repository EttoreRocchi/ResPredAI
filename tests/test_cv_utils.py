"""Unit tests for CV utilities module."""

import numpy as np
import pytest
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)

from respredai.core.cv_utils import RepeatedStratifiedGroupKFold, get_outer_cv


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
