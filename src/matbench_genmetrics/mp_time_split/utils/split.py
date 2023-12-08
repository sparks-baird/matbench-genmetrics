from warnings import warn

import numpy as np
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.model_selection._split import _BaseKFold
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

AVAILABLE_MODES = ["TimeSeriesSplit", "TimeSeriesOverflowSplit", "TimeKFold"]


def mp_time_splitter(
    X,
    mode="TimeSeriesSplit",
    use_trainval_test: bool = True,
    n_cv_splits: int = 5,
    max_train_size=None,
    test_size=None,
    gap=0,
):
    """Split into trainval and test sets, and optionally return test_sets.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame of Materials Project data to be split.
    mode : str, optional
        One of {"TimeSeriesSplit", "TimeSeriesOverflowSplit", "TimeKFold"}, by default
        "TimeSeriesSplit"
    use_trainval_test : bool, optional
        Whether to use a trainval-test split vs. just a train-test split. The idea is
        that you should tune your hyperparameters using training and validation sets
        and then keep a held-out test set that you "only ever touch once" (e.g., run
        immediately before and only once prior to manuscript submission). By default
        True.
    n_cv_splits : int, optional
        Number of cross-validation splits to perform, by default 5
    max_train_size : int, optional
        Maximum size for a single training set, by default None
    test_size : int, optional
        Used to limit the size of the test set, by default None
    gap : int, optional
         Number of samples to exclude from the end of each training set before the test
         set, by default 0

    Returns
    -------
    list of (tuples of arrays)
        The training and test indices for that split.
    list of (tuples of arrays), 2-element tuple of arrays
        Returned when use_trainval_test is True. The (training and validation) and test
        indices for that split.

    Raises
    ------
    NotImplementedError
        mode={mode} not implemented. Use one of {AVAILABLE_MODES}
    NotImplementedError
        non-zero `gap` specified, not implemented for TimeKFold
    NotImplementedError
        non-None `max_train_size` specified, not implemented for TimeKFold
    NotImplementedError
        non-None `test_size` specified, not implemented for TimeKFold

    Examples
    --------
    >>> mpt = MPTimeSplit(num_sites=num_sites, elements=elements)
    >>> data = mpt.load(dummy=True)
    >>> trainval_splits, test_split = mp_time_split(data, use_trainval_test=True)
    >>> print(trainval_splits)
    [
        (array([0]), array([1])),
        (array([0, 1]), array([2])),
        (array([0, 1, 2]), array([3])),
        (array([0, 1, 2, 3]), array([4])),
        (array([0, 1, 2, 3, 4]), array([5])),
    ]
    >>> print(test_split)
    (array([0, 1, 2, 3, 4, 5]), array([6, 7]))
    >>>
    >>> # **no held-out test set**
    >>> trainval_splits = mp_time_split(data, use_trainval_test=False)
    >>> print(trainval_splits)
    [
        (array([0, 1, 2]), array([3])),
        (array([0, 1, 2, 3]), array([4])),
        (array([0, 1, 2, 3, 4]), array([5])),
        (array([0, 1, 2, 3, 4, 5]), array([6])),
        (array([0, 1, 2, 3, 4, 5, 6]), array([7])),
    ]
    """
    if mode not in AVAILABLE_MODES:
        raise NotImplementedError(
            f"mode={mode} not implemented. Use one of {AVAILABLE_MODES}"
        )

    if use_trainval_test:
        # NOTE: the test indices get assigned later using `np.arange`
        X_trainval, _ = train_test_split(X, shuffle=False, test_size=0.2)
    else:
        X_trainval = X

    if mode == "TimeSeriesSplit":
        splitter = TimeSeriesSplit(
            n_splits=n_cv_splits,
            max_train_size=max_train_size,
            test_size=test_size,
            gap=0,
        )
    elif mode == "TimeSeriesOverflowSplit":
        splitter = TimeSeriesOverflowSplit(
            n_splits=n_cv_splits,
            max_train_size=max_train_size,
            test_size=test_size,
            gap=0,
        )
    elif mode == "TimeKFold":
        if gap != 0:
            raise NotImplementedError(
                "non-zero `gap` specified, not implemented for TimeKFold"
            )
        if max_train_size is not None:
            raise NotImplementedError(
                "non-None `max_train_size` specified, not implemented for TimeKFold"
            )
        if test_size is not None:
            raise NotImplementedError(
                "non-None `test_size` specified, not implemented for TimeKFold"
            )
        splitter = TimeKFold(n_splits=n_cv_splits)
    trainval_splits = list(splitter.split(X_trainval))

    if use_trainval_test:
        num_samples = X.shape[0]
        n_trainval = X_trainval.shape[0]
        test_split = (np.arange(0, n_trainval), np.arange(n_trainval, num_samples))
        return trainval_splits, test_split
    else:
        return trainval_splits


class TimeSeriesOverflowSplit(_BaseKFold):
    """Time Series cross-validator that always uses remainder of data as test data."""

    def __init__(self, n_splits=5, *, max_train_size=None, test_size=None, gap=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_size = max_train_size
        self.test_size = test_size
        self.gap = gap

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        groups : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        gap = self.gap
        test_size = (
            self.test_size if self.test_size is not None else n_samples // n_folds
        )

        all_index = list(range(n_samples))
        tscv = TimeSeriesSplit(
            gap=gap,
            n_splits=n_splits,
            test_size=test_size,
            max_train_size=self.max_train_size,
        )
        train_indices = []
        test_indices = []
        for tri, _ in tscv.split(X):
            train_indices.append(tri)
            # use remainder of data rather than default `test_index`
            test_indices.append(np.setdiff1d(all_index, tri))

        splits = list(zip(train_indices, test_indices))

        for train_index, test_index in splits:
            yield train_index, test_index


class TimeKFold(_BaseKFold):
    """Time Series K-Folds cross-validator

    TODO: update docstring

    Provides train/test indices to split data in train/test sets. Split
    dataset into k consecutive folds (without shuffling by default).

    Each fold is then used once as a validation while the k - 1 remaining
    folds form the training set.

    Read more in the :ref:`User Guide <k_fold>`.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds. Must be at least 2.

        .. versionchanged:: 0.22
            ``n_splits`` default value changed from 3 to 5.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.model_selection import KFold
    >>> X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
    >>> y = np.array([1, 2, 3, 4])
    >>> kf = KFold(n_splits=2)
    >>> kf.get_n_splits(X)
    2
    >>> print(kf)
    KFold(n_splits=2, random_state=None, shuffle=False)
    >>> for train_index, test_index in kf.split(X):
    ...     print("TRAIN:", train_index, "TEST:", test_index)
    ...     X_train, X_test = X[train_index], X[test_index]
    ...     y_train, y_test = y[train_index], y[test_index]
    TRAIN: [2 3] TEST: [0 1]
    TRAIN: [0 1] TEST: [2 3]

    Notes
    -----
    The first ``n_samples % n_splits`` folds have size
    ``n_samples // n_splits + 1``, other folds have size
    ``n_samples // n_splits``, where ``n_samples`` is the number of samples.

    Randomized CV splitters may return different results for each call of
    split. You can make the results identical by setting `random_state`
    to an integer.

    See Also
    --------
    StratifiedKFold : Takes group information into account to avoid building
        folds with imbalanced class distributions (for binary or multiclass
        classification tasks).

    GroupKFold : K-fold iterator variant with non-overlapping groups.

    RepeatedKFold : Repeats K-Fold n times.
    """

    def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
        if shuffle or random_state is not None:
            warn(
                "`shuffle` and `random_state` for compatibility only. These are fixed to `False` and `None`, respectively."  # noqa: E501
            )
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.

        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        # an extra split to ensure that last `text_index` is not empty
        kf = KFold(n_splits=self.n_splits + 1)
        splits = [indices[1] for indices in kf.split(X)]
        splits.pop(-1)

        running_index = np.empty(0, dtype=int)
        all_index = list(range(n_samples))
        for s in splits:
            running_index = np.concatenate((running_index, s))
            train_index = running_index
            test_index = np.setdiff1d(all_index, running_index)
            yield train_index, test_index
