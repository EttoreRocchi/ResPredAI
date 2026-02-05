"""Cross-validation utilities for ResPredAI."""

from typing import Generator, Optional

import numpy as np
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.model_selection._split import BaseCrossValidator


class RepeatedStratifiedGroupKFold(BaseCrossValidator):
    """
    Repeated Stratified Group K-Fold cross-validator.

    Repeats StratifiedGroupKFold n_repeats times with different random shuffles.
    Ensures groups are kept together within each split across all repeats.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.
    n_repeats : int, default=1
        Number of times cross-validator needs to be repeated.
    random_state : int, optional
        Controls the randomness of each repeated cross-validation instance.
        Pass an int for reproducible output across multiple function calls.

    Notes
    -----
    sklearn does not provide a RepeatedStratifiedGroupKFold, so this class
    implements it by wrapping StratifiedGroupKFold with different random seeds
    for each repetition.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 1,
        random_state: Optional[int] = None,
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(
        self,
        X,
        y,
        groups=None,
    ) -> Generator:
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.

        Raises
        ------
        ValueError
            If groups is None.
        """
        if groups is None:
            raise ValueError("groups must be provided for RepeatedStratifiedGroupKFold")

        np_groups = np.asarray(groups)
        if np_groups.shape[0] != len(X):
            raise ValueError("groups must have the same length as X")

        rng = np.random.default_rng(self.random_state)

        for _repeat in range(self.n_repeats):
            # Generate different random state for each repeat
            repeat_seed = rng.integers(np.iinfo(np.int32).max)

            cv = StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=repeat_seed,
            )

            for train_idx, test_idx in cv.split(X, y, groups):
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """
        Return the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations (n_splits * n_repeats).
        """
        return self.n_splits * self.n_repeats


def get_outer_cv(
    n_splits: int,
    n_repeats: int = 1,
    use_groups: bool = False,
    random_state: int = 42,
):
    """
    Get the appropriate outer cross-validation splitter.

    Factory function that returns the correct cross-validator based on
    whether groups are used and whether repeated CV is requested.

    Parameters
    ----------
    n_splits : int
        Number of folds.
    n_repeats : int, default=1
        Number of repetitions. Use 1 for standard CV, >1 for repeated CV.
    use_groups : bool, default=False
        Whether to use group-aware CV (StratifiedGroupKFold variants).
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    cv : cross-validator
        The appropriate cross-validator instance:
        - n_repeats=1, use_groups=False: StratifiedKFold
        - n_repeats=1, use_groups=True: StratifiedGroupKFold
        - n_repeats>1, use_groups=False: RepeatedStratifiedKFold
        - n_repeats>1, use_groups=True: RepeatedStratifiedGroupKFold
    """
    if n_repeats == 1:
        # Standard (non-repeated) CV
        if use_groups:
            return StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
        else:
            return StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=random_state,
            )
    else:
        # Repeated CV
        if use_groups:
            return RepeatedStratifiedGroupKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )
        else:
            return RepeatedStratifiedKFold(
                n_splits=n_splits,
                n_repeats=n_repeats,
                random_state=random_state,
            )
