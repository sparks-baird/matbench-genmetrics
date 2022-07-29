"""
This is a skeleton file that can serve as a starting point for a Python
console script. To run this script uncomment the following lines in the
``[options.entry_points]`` section in ``setup.cfg``::

    console_scripts =
         fibonacci = matbench_genmetrics.skeleton:run

Then run ``pip install .`` (or ``pip install -e .`` for editable mode)
which will install the command ``fibonacci`` inside your current environment.

Besides console scripts, the header (i.e. until ``_logger``...) of this file can
also be used as template for Python modules.

Note:
    This file can be renamed depending on your needs or safely removed if not needed.

References:
    - https://setuptools.pypa.io/en/latest/userguide/entry_point.html
    - https://pip.pypa.io/en/stable/reference/pip_install
"""

import argparse
import logging
import sys

import numpy as np
from mp_time_split.core import MPTimeSplit
from pymatgen.analysis.structure_matcher import StructureMatcher
from scipy.stats import wasserstein_distance
from tqdm import tqdm
from tqdm.notebook import tqdm as ipython_tqdm

from matbench_genmetrics import __version__

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


sm = StructureMatcher(stol=0.5, ltol=0.3, angle_tol=10.0)

try:
    import google.colab  # type: ignore # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


class GenMatcher(object):
    def __init__(self, test_structures, gen_structures, verbose=True) -> None:
        self.test_structures = test_structures
        self.gen_structures = gen_structures
        self.num_test = len(test_structures)
        self.num_gen = len(gen_structures)
        self.verbose = verbose

        if self.verbose:
            # https://stackoverflow.com/a/58102605/13697228
            is_notebook = hasattr(__builtins__, "__IPYTHON__")
            self.tqdm = ipython_tqdm if is_notebook else tqdm
            self.tqdm = tqdm
            self.tqdm_kwargs = dict(position=0, leave=True) if IN_COLAB else {}
        else:
            self.tqdm = lambda x: x
            self.tqdm_kwargs = {}

        self._match_matrix = None

    @property
    def match_matrix(self):
        if self._match_matrix is not None:
            return self._match_matrix

        match_matrix = np.zeros((self.num_test, self.num_gen))
        for i, ts in enumerate(self.tqdm(self.test_structures, **self.tqdm_kwargs)):
            for j, gs in enumerate(self.gen_structures):
                match_matrix[i, j] = sm.fit(ts, gs)

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
        if self.num_test != self.num_gen:
            raise ValueError("Test and gen sets should be identical.")
        # TODO: assert that test and gen sets are identical
        return np.clip(self.match_counts - 1, 0, None)

    @property
    def duplicity_count(self):
        return np.sum(self.duplicity_counts)

    @property
    def duplicity_rate(self):
        num_possible = self.num_test**2 - self.num_test
        return self.duplicity_count / num_possible


class GenMetrics(object):
    def __init__(
        self,
        train_structures,
        test_structures,
        gen_structures,
        test_pred_structures=None,
        verbose=True,
    ):
        self.train_structures = train_structures
        self.test_structures = test_structures
        self.gen_structures = gen_structures
        self.test_pred_structures = test_pred_structures
        self.verbose = verbose
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
        train_test_structures = self.train_structures + self.test_structures
        train_test_spg = [ts.get_space_group_info()[1] for ts in train_test_structures]
        gen_spg = [ts.get_space_group_info()[1] for ts in self.gen_structures]
        dummy_case = wasserstein_distance(train_test_spg, [1])
        return 1 - wasserstein_distance(train_test_spg, gen_spg) / dummy_case

    @property
    def coverage(self):
        """Match rate between test structures and generated structures."""
        coverage = GenMatcher(
            self.test_structures, self.gen_structures, verbose=self.verbose
        ).match_rate
        return coverage

    @property
    def novelty(self):
        """One minus match rate between train structures and generated structures."""
        similarity = GenMatcher(
            self.train_structures, self.gen_structures, verbose=self.verbose
        ).match_rate
        return 1.0 - similarity

    @property
    def uniqueness(self):
        """One minus duplicity rate within generated structures."""
        commonality = GenMatcher(
            self.gen_structures, self.gen_structures, verbose=self.verbose
        ).duplicity_rate
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


class MPTSMetrics(GenMetrics):
    def __init__(self, dummy=False, verbose=True):
        self.dummy = dummy
        self.verbose = verbose
        self.mpt = MPTimeSplit(target="energy_above_hull")
        self.folds = self.mpt.folds
        self.recorded_metrics = {}

    def get_train_and_val_data(self, fold, include_val=False):
        self.mpt.load(dummy=self.dummy)
        (
            self.train_inputs,
            self.val_inputs,
            self.train_outputs,
            self.val_outputs,
        ) = self.mpt.get_train_and_val_data(fold)

        if include_val:
            return self.train_inputs, self.val_inputs

        return self.train_inputs

    def evaluate_and_record(self, fold, gen_structures, test_pred_structures=None):
        GenMetrics.__init__(
            self,
            self.train_inputs.tolist(),
            self.val_inputs.tolist(),
            gen_structures,
            test_pred_structures=test_pred_structures,
            verbose=self.verbose,
        )

        self.recorded_metrics[fold] = self.metrics


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
