"""Core functionality for matbench-genmetrics (generative materials benchmarking)"""
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from pymatgen.core.structure import Structure
from pystow import ensure_csv
from scipy.stats import wasserstein_distance

from matbench_genmetrics.core.utils.featurize import (
    featurize_comp_struct,
    mod_petti_contributions,
)
from matbench_genmetrics.core.utils.match import (
    ALLOWED_MATCH_TYPES,
    cdvae_cov_compstruct_match_matrix,
    get_structure_match_matrix,
)
from matbench_genmetrics.mp_time_split.splitter import MPTimeSplit

# causes pytest to fail (tests not found, DLL load error)
# from matbench_genmetrics.cdvae.metrics import RecEval, GenEval, OptEval


__author__ = "sgbaird"
__copyright__ = "sgbaird"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


IN_COLAB = "google.colab" in sys.modules

FULL_COMP_NAME = "comp_fingerprints.csv"
DUMMY_COMP_NAME = "dummy_comp_fingerprints.csv"
FULL_STRUCT_NAME = "struct_fingerprints.csv"
DUMMY_STRUCT_NAME = "dummy_struct_fingerprints.csv"
FULL_SPG_NAME = "space_group_number.csv"
DUMMY_SPG_NAME = "dummy_space_group_number.csv"
FULL_MODPETTI_NAME = "mod_petti_contributions.csv"
DUMMY_MODPETTI_NAME = "dummy_mod_petti_contributions.csv"

FULL_COMP_URL = "https://figshare.com/ndownloader/files/36581838"
DUMMY_COMP_URL = "https://figshare.com/ndownloader/files/36582174"
FULL_STRUCT_URL = "https://figshare.com/ndownloader/files/36581841"
DUMMY_STRUCT_URL = "https://figshare.com/ndownloader/files/36582177"
FULL_SPG_URL = "https://figshare.com/ndownloader/files/36620538"
DUMMY_SPG_URL = "https://figshare.com/ndownloader/files/36620544"
FULL_MODPETTI_URL = "https://figshare.com/ndownloader/files/36620535"
DUMMY_MODPETTI_URL = "https://figshare.com/ndownloader/files/36620541"

DATA_HOME = "matbench-genmetrics"


class GenMatcher(object):
    """
    A class for matching generated structures to test structures.

    Parameters
    ----------
    test_structures : List[pymatgen.Structure]
        A list of test structures.
    gen_structures : List[pymatgen.Structure]
        A list of generated structures.
    match_type : str, optional
        The type of matching algorithm to use. Default is "StructureMatcher".
    match_kwargs : dict, optional
        Additional keyword arguments to pass to the matching algorithm.

    Attributes
    ----------
    match_matrix : numpy.ndarray
        A matrix of match scores between test and generated structures.
    """

    def __init__(
        self,
        test_structures,
        gen_structures: Optional[List[Structure]] = None,
        test_comp_fingerprints: Optional[np.ndarray] = None,
        gen_comp_fingerprints: Optional[np.ndarray] = None,
        test_struct_fingerprints: Optional[np.ndarray] = None,
        gen_struct_fingerprints: Optional[np.ndarray] = None,
        verbose=True,
        match_type="cdvae_coverage",
        **match_kwargs,
    ) -> None:
        self.test_structures = test_structures
        self.test_comp_fingerprints = test_comp_fingerprints
        self.test_struct_fingerprints = test_struct_fingerprints
        self.verbose = verbose
        assert (
            match_type in ALLOWED_MATCH_TYPES
        ), f"type must be one of {ALLOWED_MATCH_TYPES}"
        self.match_type = match_type
        self.match_kwargs = match_kwargs

        if gen_structures is None:
            self.gen_structures = test_structures
            self.symmetric = True
        else:
            self.gen_structures = gen_structures
            self.symmetric = False

        # featurize test and/or gen structures if features not provided
        if self.match_type == "cdvae_coverage":
            if test_comp_fingerprints is None or test_struct_fingerprints is None:
                (
                    self.test_comp_fingerprints,
                    self.test_struct_fingerprints,
                ) = featurize_comp_struct(self.test_structures)

            if self.symmetric:
                self.gen_comp_fingerprints, self.gen_struct_fingerprints = (
                    self.test_comp_fingerprints,
                    self.test_struct_fingerprints,
                )
            elif gen_comp_fingerprints is None or gen_struct_fingerprints is None:
                (
                    self.gen_comp_fingerprints,
                    self.gen_struct_fingerprints,
                ) = featurize_comp_struct(self.gen_structures)
            else:
                self.gen_comp_fingerprints = gen_comp_fingerprints
                self.gen_struct_fingerprints = gen_struct_fingerprints

        self.num_test = len(self.test_structures)
        self.num_gen = len(self.gen_structures)

        self._match_matrix = None

    @property
    def match_matrix(self):
        """A matrix of match scores between test and generated structures.

        match_matrix : numpy.ndarray
            The element at position (i, j) represents the match score between the ith
            test structure and the jth generated structure. The match score is
            calculated using the specified matching algorithm and any additional keyword
            arguments passed to the `GenMatcher` class.

            Examples
            --------
            >>> from pymatgen.core.structure import Structure
            >>> from pymatgen.core.lattice import Lattice
            >>> test_structures = [
            ...     Structure(
            ...         Lattice.cubic(3.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]
            ...     ),
            ...     Structure(
            ...         Lattice.cubic(3.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]
            ...     ),
            ... ]
            >>> gen_structures = [
            ...     Structure(
            ...         Lattice.cubic(3.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]
            ...     ),
            ...     Structure(
            ...         Lattice.cubic(3.0), ["Si", "Si"], [[0, 0, 0], [0.5, 0.5, 0.5]]
            ...     ),
            ... ]
            >>> matcher = GenMatcher(test_structures, gen_structures)
            >>> matcher.match_matrix
            array([[1., 1.],
                [1., 1.]])
        """
        if self._match_matrix is not None:
            return self._match_matrix

        if self.match_type == "StructureMatcher":
            match_matrix = get_structure_match_matrix(
                self.test_structures,
                self.gen_structures,
                match_type=self.match_type,
                symmetric=self.symmetric,
                verbose=self.verbose,
                **self.match_kwargs,
            )
        elif self.match_type == "cdvae_coverage":
            match_matrix = cdvae_cov_compstruct_match_matrix(
                self.test_comp_fingerprints,
                self.gen_comp_fingerprints,
                self.test_struct_fingerprints,
                self.gen_struct_fingerprints,
                symmetric=self.symmetric,
                verbose=self.verbose,
                **self.match_kwargs,
            )

        self._match_matrix = match_matrix

        return match_matrix

    @property
    def match_counts(self):
        return np.sum(self.match_matrix, axis=0)

    @property
    def match_count(self):
        return np.sum(self.match_counts > 0)

    @property
    def match_rate(self):
        return self.match_count / self.num_test

    @property
    def duplicity_counts(self):
        """Get number of duplicates within the match matrix ignoring self-comparison."""
        if self.num_test != self.num_gen:
            raise ValueError("Test and gen sets should be identical.")
            # TODO: assert that test and gen sets are identical

        # minus one is subtraction of the diagonal that got summed
        return np.clip(self.match_counts - 1, 0, None)

    @property
    def duplicity_count(self):
        return np.sum(self.duplicity_counts)

    @property
    def duplicity_rate(self):
        """Get number of duplicates divided by number of possible duplicate locations.

        A set with 4 instances of the same structure will score lower than 2 repeat
        instances each for 2 structures. In other words, the metric favors a larger
        of unique structures, even if repeat structures exist.
        """
        num_possible = self.num_test**2 - self.num_test
        return self.duplicity_count / num_possible


class GenMetrics(object):
    """
    Evaluate the performance of a generative model on a set of training and test data.

    Parameters
    ----------
    train_structures : List[pymatgen.Structure]
        A list of training structures.
    test_structures : List[pymatgen.Structure]
        A list of test structures.
    gen_structures : List[pymatgen.Structure]
        A list of generated structures.
    test_pred_structures : List[pymatgen.Structure]
        A list of test structures predicted by the machine learning model.
    match_type : str, optional
        The type of matching algorithm to use. Default is "StructureMatcher".
    match_kwargs : dict, optional
        Additional keyword arguments to pass to the matching algorithm.

    Attributes
    ----------
    train_test_match_matrix : numpy.ndarray
        A matrix of match scores between training and test structures. The element at
        position (i, j) represents the match score between the ith training structure
        and the jth test structure. The match score is calculated using the specified
        matching algorithm and any additional keyword arguments passed to the
        `GenMetrics` class.
    test_pred_match_matrix : numpy.ndarray
        A matrix of match scores between test structures and predicted test structures.
        The element at position (i, j) represents the match score between the ith test
        structure and the jth predicted test structure. The match score is calculated
        using the specified matching algorithm and any additional keyword arguments
        passed to the `GenMetrics` class.
    train_gen_match_matrix : numpy.ndarray
        A matrix of match scores between training structures and generated structures.
        The element at position (i, j) represents the match score between the ith
        training structure and the jth generated structure. The match score is
        calculated using the specified matching algorithm and any additional keyword
        arguments passed to the `GenMetrics` class.
    test_gen_match_matrix : numpy.ndarray
        A matrix of match scores between test structures and generated structures. The
        element at position (i, j) represents the match score between the ith test
        structure and the jth generated structure. The match score is calculated using
        the specified matching algorithm and any additional keyword arguments passed to
        the `GenMetrics` class.
    """

    def __init__(
        self,
        train_structures: List[Structure],
        test_structures: List[Structure],
        gen_structures: List[Structure],
        train_comp_fingerprints=None,
        test_comp_fingerprints=None,
        train_struct_fingerprints=None,
        test_struct_fingerprints=None,
        train_test_spg=None,
        train_test_modpetti_df=None,
        test_pred_structures=None,
        verbose=True,
        match_type="cdvae_coverage",
        **match_kwargs,
    ):
        self.train_structures = train_structures
        self.test_structures = test_structures
        self.gen_structures = gen_structures
        self.train_comp_fingerprints = train_comp_fingerprints
        self.test_comp_fingerprints = test_comp_fingerprints
        self.train_struct_fingerprints = train_struct_fingerprints
        self.test_struct_fingerprints = test_struct_fingerprints
        self.train_test_spg = train_test_spg
        self.train_test_modpetti_df = train_test_modpetti_df
        self.test_pred_structures = test_pred_structures
        self.verbose = verbose
        self.match_type = match_type
        self.match_kwargs = match_kwargs

        (
            self.gen_comp_fingerprints,
            self.gen_struct_fingerprints,
        ) = featurize_comp_struct(self.gen_structures)

        self._cdvae_metrics = None
        self._mpts_metrics = None

    @property
    def validity(self):
        """Scaled Wasserstein distance between real (train/test) and gen structures."""
        # TODO: implement notion of compositional validity, since this is only structure
        train_test_structures = self.train_structures + self.test_structures

        def try_get_space_group_info(structure):
            try:
                spg_tmp = structure.get_space_group_info()
            except TypeError as e:
                _logger.debug(e)
                spg_tmp = ("P", 1)
            return spg_tmp

        if self.train_test_spg is None:
            self.train_test_spg = [
                try_get_space_group_info(ts)[1] for ts in train_test_structures
            ]

        gen_spg = [try_get_space_group_info(ts)[1] for ts in self.gen_structures]

        if self.train_test_modpetti_df is None:
            self.train_test_modpetti_df = mod_petti_contributions(train_test_structures)
        gen_modpetti_df = mod_petti_contributions(self.gen_structures)

        dummy_spg_case = wasserstein_distance(self.train_test_spg, [1])
        spg_distance = wasserstein_distance(self.train_test_spg, gen_spg)

        dummy_modpetti_case = wasserstein_distance(
            self.train_test_modpetti_df.mod_petti.values,
            [1],
            u_weights=self.train_test_modpetti_df.contribution.values,
            v_weights=[1],
        )
        modpetti_distance = wasserstein_distance(
            self.train_test_modpetti_df.mod_petti.values,
            gen_modpetti_df.mod_petti.values,
            u_weights=self.train_test_modpetti_df.contribution.values,
            v_weights=gen_modpetti_df.contribution.values,
        )

        spg_scaled_distance = spg_distance / dummy_spg_case
        modpetti_scaled_distance = modpetti_distance / dummy_modpetti_case

        self.spg_validity = 1 - spg_scaled_distance
        self.modpetti_validity = 1 - modpetti_scaled_distance

        return 0.5 * (self.spg_validity + self.modpetti_validity)

    @property
    def coverage(self):
        """Match rate between test structures and generated structures."""
        self.coverage_matcher = GenMatcher(
            self.test_structures,
            self.gen_structures,
            test_comp_fingerprints=self.test_comp_fingerprints,
            test_struct_fingerprints=self.test_struct_fingerprints,
            gen_comp_fingerprints=self.gen_comp_fingerprints,
            gen_struct_fingerprints=self.gen_struct_fingerprints,
            verbose=self.verbose,
            match_type=self.match_type,
            **self.match_kwargs,
        )
        return self.coverage_matcher.match_rate

    @property
    def novelty(self):
        """One minus match rate between train structures and generated structures."""
        self.similarity_matcher = GenMatcher(
            self.train_structures,
            self.gen_structures,
            test_comp_fingerprints=self.train_comp_fingerprints,
            test_struct_fingerprints=self.train_struct_fingerprints,
            gen_comp_fingerprints=self.gen_comp_fingerprints,
            gen_struct_fingerprints=self.gen_struct_fingerprints,
            verbose=self.verbose,
            match_type=self.match_type,
            **self.match_kwargs,
        )
        similarity = (
            self.similarity_matcher.match_count / self.similarity_matcher.num_gen
        )
        return 1.0 - similarity

    @property
    def uniqueness(self):
        """One minus duplicity rate within generated structures."""
        self.commonality_matcher = GenMatcher(
            self.gen_structures,
            self.gen_structures,
            test_comp_fingerprints=self.gen_comp_fingerprints,
            test_struct_fingerprints=self.gen_struct_fingerprints,
            gen_comp_fingerprints=self.gen_comp_fingerprints,
            gen_struct_fingerprints=self.gen_struct_fingerprints,
            verbose=self.verbose,
            match_type=self.match_type,
            **self.match_kwargs,
        )
        commonality = self.commonality_matcher.duplicity_rate
        return 1.0 - commonality

    @property
    def metrics(self):
        """Return validity, coverage, novelty, and uniqueness metrics as a dict."""
        return {
            "validity": self.validity,
            "coverage": self.coverage,
            "novelty": self.novelty,
            "uniqueness": self.uniqueness,
        }


class MPTSMetrics(object):
    """
    Evaluate the performance of a crystal generative model using MP Time Split (MPTS).

    Parameters
    ----------
    dummy : bool, optional
        Whether to use dummy data for testing purposes. Default is False.
    verbose : bool, optional
        Whether to print out the results of the evaluation. Default is True.
    num_gen : int, optional
        The number of generated structures to use for the evaluation. Default is None.
    save_dir : str, optional
        The directory to save the generated structures to. Default is "results".
    match_type : str, optional
        The type of matching algorithm to use. Default is "StructureMatcher".

    Attributes
    ----------
    train_scores : List[Dict[str, float]]
        A list of dictionaries containing the evaluation results for each fold of the
        training data.
    val_scores : List[Dict[str, float]]
        A list of dictionaries containing the evaluation results for each fold of the
        validation data.
    test_scores : List[Dict[str, float]]
        A list of dictionaries containing the evaluation results for each fold of the
        test data.
    gen_scores : List[Dict[str, float]]
        A list of dictionaries containing the evaluation results for each fold of the
        generated data.
    test_pred_scores : List[Dict[str, float]]
        A list of dictionaries containing the evaluation results for each fold of the
        predicted test data.
    **match_kwargs : Dict[str, Any]
        Keyword arguments passed to GenMetrics.

    Notes
    -----
    This class assumes that the data is split into training, validation, and test sets
    using the MP Time Split (MPTS) protocol. The `get_train_and_val_data` method is used
    to retrieve the training and validation data for a given fold, and the
    `evaluate_and_record` method is used to evaluate the performance of the model on the
    generated structures and record the results. The evaluation results are stored in
    the `train_scores`, `val_scores`, `test_scores`, `gen_scores`, and
    `test_pred_scores` attributes.
    """

    def __init__(
        self,
        dummy=False,
        verbose=True,
        num_gen=None,
        save_dir="results",
        match_type="cdvae_coverage",
        **match_kwargs,
    ):
        self.dummy = dummy
        self.verbose = verbose
        self.num_gen = num_gen
        self.save_dir = save_dir
        self.match_type = match_type
        self.match_kwargs = match_kwargs

        Path(self.save_dir).mkdir(exist_ok=True, parents=True)

        self.mpt = MPTimeSplit(target="energy_above_hull")
        self.folds = self.mpt.folds
        self.gms: List[Optional[GenMetrics]] = [None] * len(self.folds)
        self.recorded_metrics = {}

    def load_fingerprints(self, dummy=False):
        """Load precalculated fingerprints from FigShare.

        Parameters
        ----------
        dummy : bool, optional
            Whether to load a small, dummy dataset, by default False

        Returns
        -------
        DataFrame
            Compositional fingerprints

        Examples
        --------
        >>> mptm = MPTSMetrics()
        >>> comp_fingerprints_df, struct_fingerprints_df = mptm.load_fingerprints(
        ...     dummy=False
        ... )
        """
        comp_url = DUMMY_COMP_URL if dummy else FULL_COMP_URL
        struct_url = DUMMY_STRUCT_URL if dummy else FULL_STRUCT_URL
        comp_name = DUMMY_COMP_NAME if dummy else FULL_COMP_NAME
        struct_name = DUMMY_STRUCT_NAME if dummy else FULL_STRUCT_NAME

        read_csv_kwargs = dict(index_col="material_id", sep=",")
        self.comp_fingerprints_df = ensure_csv(
            DATA_HOME,
            name=comp_name,
            url=comp_url,
            read_csv_kwargs=read_csv_kwargs,
        )
        self.struct_fingerprints_df = ensure_csv(
            DATA_HOME,
            name=struct_name,
            url=struct_url,
            read_csv_kwargs=read_csv_kwargs,
        )
        # REVIEW: consider doing checksum validation

        return self.comp_fingerprints_df, self.struct_fingerprints_df

    def load_space_group_and_mod_petti(self, dummy=False):
        """Load space groups and modified pettifor encodings from FigShare.

        Parameters
        ----------
        dummy : bool, optional
            Whether to load a small, dummy dataset, by default False

        Returns
        -------
        DataFrame
            space group numbers
        DataFrame
            modified pettifor encodings

        Examples
        --------
        >>> mptm = MPTSMetrics()
        >>> spg_df, modpetti_df = mptm.load_space_group_and_mod_petti(dummy=False)
        """
        spg_name = DUMMY_SPG_NAME if dummy else FULL_SPG_NAME
        spg_url = DUMMY_SPG_URL if dummy else FULL_SPG_URL
        modpetti_name = DUMMY_MODPETTI_NAME if dummy else FULL_MODPETTI_NAME
        modpetti_url = DUMMY_MODPETTI_URL if dummy else FULL_MODPETTI_URL

        self.spg_df = ensure_csv(
            DATA_HOME,
            name=spg_name,
            url=spg_url,
            read_csv_kwargs=dict(index_col="material_id", sep=","),
        )
        self.modpetti_df = ensure_csv(
            DATA_HOME,
            name=modpetti_name,
            url=modpetti_url,
            read_csv_kwargs=dict(index_col="symbol", sep=","),
        )
        return self.spg_df, self.modpetti_df

    def get_train_and_val_data(self, fold: int, return_test=False):
        """Get the MPTimeSplit fingerprints, sp.grp numbers, and modPetti info.

        Parameters
        ----------
        fold : int
            Which of the 5 folds to use for training and validation (0-4)
        return_test : bool, optional
            Whether to return the test data in addition to the training and
            validation data, by default False

        Returns
        -------
        DataFrame
            Training and validation inputs
        DataFrame
            Test inputs. Only returned if `return_test` is True.


        Examples
        --------
        >>> mptm = MPTSMetrics()
        >>> train_and_val_inputs = mptm.get_train_and_val_data(fold, return_test=False)
        """
        if self.recorded_metrics == {}:
            self.mpt.load(dummy=self.dummy)
        (
            self.train_and_val_inputs,
            self.test_inputs,
            self.train_and_val_outputs,
            self.test_outputs,
        ) = self.mpt.get_train_and_val_data(fold)

        spg_df, modpetti_df = self.load_space_group_and_mod_petti(dummy=self.dummy)
        self.spg = spg_df.space_group_number.values
        self.modpetti_df = modpetti_df

        if self.match_type == "cdvae_coverage":
            comp_fps, struct_fps = self.load_fingerprints(dummy=self.dummy)

            self.train_comp_fingerprints, self.val_comp_fingerprints = [
                comp_fps.iloc[tvs].values for tvs in self.mpt.trainval_splits[fold]
            ]

            self.train_struct_fingerprints, self.val_struct_fingerprints = [
                struct_fps.iloc[tvs].values for tvs in self.mpt.trainval_splits[fold]
            ]
        elif self.match_type == "StructureMatcher":
            self.train_comp_fingerprints = None
            self.val_comp_fingerprints = None
            self.train_struct_fingerprints = None
            self.val_struct_fingerprints = None

        if return_test:
            return self.train_and_val_inputs, self.test_inputs

        return self.train_and_val_inputs

    def evaluate_and_record(self, fold: int, gen_structures, test_pred_structures=None):
        """Evaluate generated structures and record metrics.

        Parameters
        ----------
        fold : int
            Fold number.
        gen_structures : list of pymatgen Structure
        List of generated structures.
        test_pred_structures : list of pymatgen Structure, optional
            List of predicted structures for the test set. If not provided, the
            test set is assumed to be the same as the validation set.
        """
        if self.num_gen is not None and self.num_gen != len(gen_structures):
            raise ValueError(
                f"Number of generated structures ({len(gen_structures)}) does not match expected number ({self.num_gen})."  # noqa: E501
            )
        self.gms[fold] = GenMetrics(
            self.train_and_val_inputs.tolist(),
            self.test_inputs.tolist(),
            gen_structures,
            train_comp_fingerprints=self.train_comp_fingerprints,
            test_comp_fingerprints=self.val_comp_fingerprints,
            train_struct_fingerprints=self.train_struct_fingerprints,
            test_struct_fingerprints=self.val_struct_fingerprints,
            train_test_spg=self.spg,
            train_test_modpetti_df=self.modpetti_df,
            test_pred_structures=test_pred_structures,
            verbose=self.verbose,
            match_type=self.match_type,
            **self.match_kwargs,
        )

        self.recorded_metrics[fold] = self.gms[fold].metrics

        # i.e. store the values for the current fold for testing purposes
        for metric, value in self.recorded_metrics[fold].items():
            setattr(self, metric, value)

    def save(self, fpath_stem):
        with open(fpath_stem + ".pkl", "wb") as f:
            pickle.dump(self, f)

        with open(fpath_stem + ".json", "w") as fp:
            json.dump(self.recorded_metrics, fp)

    def load(self, fpath):
        with open(fpath, "rb") as f:
            return pickle.load(f)


class MPTSMetrics10(MPTSMetrics):
    """Benchmark class for MPTSMetrics with 10 generated structures."""

    def __init__(self, dummy=False, verbose=True):
        MPTSMetrics.__init__(
            self, dummy=dummy, verbose=verbose, num_gen=10, match_type="cdvae_coverage"
        )


class MPTSMetrics100(MPTSMetrics):
    """Benchmark class for MPTSMetrics with 100 generated structures."""

    def __init__(self, dummy=False, verbose=True):
        MPTSMetrics.__init__(
            self, dummy=dummy, verbose=verbose, num_gen=100, match_type="cdvae_coverage"
        )


class MPTSMetrics1000(MPTSMetrics):
    """Benchmark class for MPTSMetrics with 1000 generated structures."""

    def __init__(self, dummy=False, verbose=True):
        MPTSMetrics.__init__(
            self,
            dummy=dummy,
            verbose=verbose,
            num_gen=1000,
            match_type="cdvae_coverage",
        )


class MPTSMetrics10000(MPTSMetrics):
    """Benchmark class for MPTSMetrics with 10000 generated structures."""

    def __init__(self, dummy=False, verbose=True):
        MPTSMetrics.__init__(
            self,
            dummy=dummy,
            verbose=verbose,
            num_gen=10000,
            match_type="cdvae_coverage",
        )
