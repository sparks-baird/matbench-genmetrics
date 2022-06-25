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
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm

from matbench_genmetrics import __version__

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


class Match(object):
    def __init__(self, test_structures, gen_structures) -> None:
        self.test_structures = test_structures
        self.gen_structures = gen_structures
        self.num_test = len(test_structures)
        self.num_gen = len(gen_structures)

        self.match_counts = None

    def get_match_matrix(self):
        match_matrix = np.zeros((self.num_test, self.num_gen))
        for i, ts in enumerate(tqdm(self.test_structures)):
            for j, gs in enumerate(tqdm(self.gen_structures)):
                match_matrix[i, j] = sm.fit(ts, gs)

        return match_matrix

    def get_match_counts(self):
        if self.match_counts is not None:
            return self.match_counts
        self.match_matrix = self.get_match_matrix(
            self.test_structures, self.gen_structures
        )
        self.match_counts = np.sum(self.match_matrix, axis=0)
        return self.match_counts

    def get_match_rate(self):
        self.match_counts = self.get_match_counts(
            self.test_structures, self.gen_structures
        )
        self.match_count = np.sum(self.match_counts > 0)
        self.match_rate = self.match_count / self.num_test
        return self.match_rate

    def get_match_duplicity_rate(self):
        self.match_counts = self.get_match_counts(
            self.test_structures, self.gen_structures
        )
        self.match_duplicity = np.sum(self.match_counts > 1)
        self.match_duplicity_rate = self.match_duplicity / self.num_test
        return self.match_duplicity


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
