"""Core functionality for matbench-genmetrics (generative materials benchmarking)"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from mp_time_split.core import MPTimeSplit
from pymatgen.core.structure import Structure
from pystow import ensure_csv
from scipy.stats import wasserstein_distance

from matbench_genmetrics import __version__
from matbench_genmetrics.utils.featurize import featurize_comp_struct
from matbench_genmetrics.utils.match import (
    ALLOWED_MATCH_TYPES,
    cdvae_cov_compstruct_match_matrix,
    get_structure_match_matrix,
)

# causes pytest to fail (tests not found, DLL load error)
# from matbench_genmetrics.cdvae.metrics import RecEval, GenEval, OptEval


__author__ = "sgbaird"
__copyright__ = "sgbaird"
__license__ = "MIT"

_logger = logging.getLogger(__name__)


# ---- Python API ----
# The functions defined in this section can be imported by users in their
# Python scripts/interactive interpreter, e.g. via
# `from matbench_genmetrics.skeleton import fib`,
# when using this Python module as a library.


def fib(n):
    """Fibonacci example function

    Args:
      n (int): integer

    Returns:
      int: n-th Fibonacci number
    """
    assert n > 0
    a, b = 1, 1
    for _i in range(n - 1):
        a, b = b, a + b
    return a


IN_COLAB = "google.colab" in sys.modules

FULL_COMP_NAME = "comp_fingerprints.csv"
DUMMY_COMP_NAME = "dummy_comp_fingerprints.csv"
FULL_STRUCT_NAME = "struct_fingerprints.csv"
DUMMY_STRUCT_NAME = "dummy_struct_fingerprints.csv"

FULL_COMP_CHECKSUM_FROZEN = "0d714081a8f0bc53af84b0ce96d3536f"
DUMMY_COMP_CHECKSUM_FROZEN = "5630a3bfc7cbeac0cb3d7897b02aae9f"
FULL_STRUCT_CHECKSUM_FROZEN = "312a4a282c57d80aed19a07dd2760ad9"
DUMMY_STRUCT_CHECKSUM_FROZEN = "d402abc2ba383e6b18b24413bdd96a7e"

FULL_COMP_URL = "https://figshare.com/ndownloader/files/36581838"
DUMMY_COMP_URL = "https://figshare.com/ndownloader/files/36582174"
FULL_STRUCT_URL = "https://figshare.com/ndownloader/files/36581841"
DUMMY_STRUCT_URL = "https://figshare.com/ndownloader/files/36582177"


DATA_HOME = "matbench-genmetrics"


class GenMatcher(object):
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
    def __init__(
        self,
        train_structures,
        test_structures,
        gen_structures,
        train_comp_fingerprints=None,
        test_comp_fingerprints=None,
        train_struct_fingerprints=None,
        test_struct_fingerprints=None,
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

    # @property
    # def cdvae_metrics(self):
    #     # FIXME: update with CDVAE structures and handle 3 dataset types
    #     if self._cdvae_metrics is not None:
    #         return self._cdvae_metrics

    #     rec_eval = RecEval(self.test_pred_structures, self.test_structures)
    #     reconstruction_metrics = rec_eval.get_metrics()

    #     gen_eval = GenEval(self.gen_structures, self.test_structures)
    #     generation_metrics = gen_eval.get_metrics()

    #     opt_eval = OptEval(self.gen_structures, self.test_structures)
    #     optimization_metrics = opt_eval.get_metrics()

    #     self._cdvae_metrics = (
    #         reconstruction_metrics,
    #         generation_metrics,
    #         optimization_metrics,
    #     )

    #     return self._cdvae_metrics

    @property
    def validity(self):
        """Scaled Wasserstein distance between real (train/test) and gen structures."""
        # TODO: implement notion of compositional validity, since this is only structure
        train_test_structures = self.train_structures + self.test_structures
        train_test_spg = [ts.get_space_group_info()[1] for ts in train_test_structures]
        gen_spg = [ts.get_space_group_info()[1] for ts in self.gen_structures]
        dummy_case = wasserstein_distance(train_test_spg, [1])
        return 1 - wasserstein_distance(train_test_spg, gen_spg) / dummy_case

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

        return self.comp_fingerprints_df, self.struct_fingerprints_df

    def get_train_and_val_data(self, fold, include_val=False):

        if self.recorded_metrics == {}:
            self.mpt.load(dummy=self.dummy)
        (
            self.train_inputs,
            self.val_inputs,
            self.train_outputs,
            self.val_outputs,
        ) = self.mpt.get_train_and_val_data(fold)

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

        if include_val:
            return self.train_inputs, self.val_inputs

        return self.train_inputs

    def evaluate_and_record(self, fold, gen_structures, test_pred_structures=None):
        if self.num_gen is not None and self.num_gen != len(gen_structures):
            raise ValueError(
                f"Number of generated structures ({len(gen_structures)}) does not match expected number ({self.num_gen})."  # noqa: E501
            )
        self.gms[fold] = GenMetrics(
            self.train_inputs.tolist(),
            self.val_inputs.tolist(),
            gen_structures,
            train_comp_fingerprints=self.train_comp_fingerprints,
            test_comp_fingerprints=self.val_comp_fingerprints,
            train_struct_fingerprints=self.train_struct_fingerprints,
            test_struct_fingerprints=self.val_struct_fingerprints,
            test_pred_structures=test_pred_structures,
            verbose=self.verbose,
            match_type=self.match_type,
            **self.match_kwargs,
        )

        self.recorded_metrics[fold] = self.gms[fold].metrics

        # i.e. store the values for the current fold for testing purposes
        for metric, value in self.recorded_metrics[fold].items():
            setattr(self, metric, value)


class MPTSMetrics10(MPTSMetrics):
    def __init__(self, dummy=False, verbose=True):
        MPTSMetrics.__init__(
            self, dummy=dummy, verbose=verbose, num_gen=10, match_type="cdvae_coverage"
        )


class MPTSMetrics100(MPTSMetrics):
    def __init__(self, dummy=False, verbose=True):
        MPTSMetrics.__init__(
            self, dummy=dummy, verbose=verbose, num_gen=100, match_type="cdvae_coverage"
        )


class MPTSMetrics1000(MPTSMetrics):
    def __init__(self, dummy=False, verbose=True):
        MPTSMetrics.__init__(
            self,
            dummy=dummy,
            verbose=verbose,
            num_gen=1000,
            match_type="cdvae_coverage",
        )


class MPTSMetrics10000(MPTSMetrics):
    def __init__(self, dummy=False, verbose=True):
        MPTSMetrics.__init__(
            self,
            dummy=dummy,
            verbose=verbose,
            num_gen=10000,
            match_type="cdvae_coverage",
        )


# def get_rms_dist(gen_structures, test_structures):
#     rms_dist = np.zeros((len(gen_structures), len(test_structures)))
#     for i, gs in enumerate(tqdm(gen_structures)):
#         for j, ts in enumerate(tqdm(test_structures)):
#             rms_dist[i, j] = sm.get_rms_dist(gs, ts)[0]

#     return rms_dist

# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Just a Fibonacci demonstration")
    parser.add_argument(
        "--version",
        action="version",
        version=f"$matbench_genmetrics {__version__}",
    )
    parser.add_argument(dest="n", help="n-th Fibonacci number", type=int, metavar="INT")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    print("The {}-th Fibonacci number is {}".format(args.n, fib(args.n)))
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m matbench_genmetrics.skeleton 42
    #
    run()

# %% Code Graveyard
# def get_match_rate(gen_structures, test_structures):
#     match_rate = np.zeros(len(gen_structures), self.num_test)
#     for i, gs in enumerate(tqdm(gen_structures)):
#         for j, ts in enumerate(tqdm(test_structures)):
#             if i > j:
#                 match_rate[i, j] = sm.fit(gs, ts)
#             elif i == j:
#                 match_rate[i, j] = True

#     # add transpose https://stackoverflow.com/a/58806735/13697228
#     match_rate = (match_rate + match_rate.T) / 2 - np.diag(np.diag(match_rate))

#     return match_rate


# def get_match_rate(gen_structures, test_structures):
#     sm = StructureMatcher(stol=0.5, ltol=0.3, angle_tol=10.0)

#     match_rate = np.zeros(len(gen_structures), self.num_test)
#     for i, gs in enumerate(tqdm(gen_structures)):
#         for j, ts in enumerate(tqdm(test_structures)):
#             if i > j:
#                 match_rate[i, j] = sm.fit(gs, ts)
#             elif i == j:
#                 match_rate[i, j] = True

#     # add transpose https://stackoverflow.com/a/58806735/13697228
#     match_rate = (match_rate + match_rate.T) / 2 - np.diag(np.diag(match_rate))

#     return match_rate

# from CBFV.composition import generate_features

# self.test_formulas = [s.reduced_formula for s in test_structures]
# self.gen_formulas = [s.reduced_formula for s in gen_structures]

# self.test_cbfv, _, _, _ =
# generate_features(pd.DataFrame(dict(formula=self.test_formulas, target=0.0)))
# self.gen_cbfv, _, _, _ = generate_features(dict(formula=self.gen_formulas,
# target=0.0))

# try:
#     import google.colab  # type: ignore # noqa: F401

#     IN_COLAB = True
# except ImportError:
#     IN_COLAB = False

# elif test_comp_fingerprints and (gen_comp_fingerprints is None):
#     self.gen_comp_fingerprints = test_comp_fingerprints
#     self.gen_struct_fingerprints = test_struct_fingerprints
#     self.symmetric = True
# elif test_comp_fingerprints and gen_comp_fingerprints:
#     assert gen_comp_fingerprints is not None
#     self.gen_comp_fingerprints = gen_comp_fingerprints
#     self.gen_struct_fingerprints = gen_struct_fingerprints
#     self.symmetric = False
