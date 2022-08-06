from itertools import product
from typing import Callable, List, Tuple

import numpy as np
import numpy.typing as npt
import pytest
from mp_time_split.utils.gen import DummyGenerator
from numpy.testing import assert_array_equal
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import (
    PerturbStructureTransformation,
)

from matbench_genmetrics.core import GenMatcher, GenMetrics, MPTSMetrics
from matbench_genmetrics.utils.featurize import cdvae_cov_struct_fingerprints

# from pytest_cases import fixture, parametrize, parametrize_with_cases

np.random.seed(10)

coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60)

# cases: exact same, different composition, different lattice
dummy_structures = [
    Structure(lattice, ["Si", "Si"], coords),
    Structure(lattice, ["Si", "Si"], coords),
    Structure(lattice, ["Ni", "Ni"], coords),
    Structure(
        Lattice.from_parameters(a=2.95, b=2.95, c=14.37, alpha=90, beta=90, gamma=120),
        ["Ag", "Au"],
        [[0, 0, 0], [1 / 3, 2 / 3, 1 / 6]],
    ),  # https://materialsproject.org/materials/mp-1229092
]


# @fixture
def dummy_gen_matcher():
    """Get GenMatcher instance with dummy_structures as both test and gen structures."""
    return GenMatcher(dummy_structures, dummy_structures, match_type="StructureMatcher")


# @fixture
def dummy_gen_metrics():
    """Get GenMetrics instance with dummy_structures as train/test/gen structures.

    Notes
    -----
    - No matches between train and test (except train is repeated)
    - One match between test and test_pred
    - One match between train and gen
    - One match between test and gen
    """
    train_structures = dummy_structures[0:2]
    test_structures = dummy_structures[2:4]
    test_pred_structures = [dummy_structures[2], dummy_structures[0]]

    gen_structures = [
        dummy_structures[0],
        dummy_structures[0],
        dummy_structures[2],
        PerturbStructureTransformation(
            distance=5.0, min_distance=2.5
        ).apply_transformation(dummy_structures[3]),
    ]

    return GenMetrics(
        train_structures,
        test_structures,
        gen_structures,
        test_pred_structures,
        match_type="StructureMatcher",
    )


# @fixture
def dummy_mpts_metrics():
    """Get MPTSMetrics() with dummy MPTS as train/test, dummy_structures as pred/gen."""
    mptm = MPTSMetrics(dummy=True, match_type="StructureMatcher")

    fold = 0
    mptm.get_train_and_val_data(fold)
    mptm.evaluate_and_record(fold, dummy_structures)

    return mptm


dummy_matcher_expected = {
    "match_matrix": [
        [1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    "match_counts": [2.0, 2.0, 1.0, 1.0],
    "match_count": 4.0,
    "match_rate": 1.0,
    "duplicity_counts": [1.0, 1.0, 0.0, 0.0],
    "duplicity_count": 2.0,
    "duplicity_rate": 1 / 6,  # 1 of 6 entries in upper triangle of match matrix
}

dummy_metrics_expected = {
    "validity": 0.8139255702280912,
    "coverage": 0.5,
    "novelty": 0.5,
    "uniqueness": 5 / 6,
}

dummy_mpts_expected = {
    "validity": 0.9207589285714286,
    "coverage": 0.0,
    "novelty": 1.0,
    "uniqueness": 5 / 6,
}

fixtures = [dummy_gen_matcher(), dummy_gen_metrics(), dummy_mpts_metrics()]
checkitems: List[List] = [
    list(dummy_matcher_expected.items()),
    list(dummy_metrics_expected.items()),
    list(dummy_mpts_expected.items()),
]

combs: List[Tuple[Callable, Tuple[str, npt.ArrayLike]]] = []
for f, ci in zip(fixtures, checkitems):
    combs = combs + list(product([f], ci))


@pytest.mark.parametrize("fixture,checkitem", combs)
def test_numerical_attributes(fixture: object, checkitem: Tuple[str, npt.ArrayLike]):
    """Verify that numerical attributes match the expected values.

    Parameters
    ----------
    fixture : Callable
        a pytest fixture that returns an instantiated class operable with getattr
    checkitem : Tuple[str, npt.ArrayLike]
        A tuple of the attribute name to test and the expected ArrayLike value.

    Examples
    --------
    >>> test_numerical_attributes(SomeClass(), ("class_attr", [1, 2, 3]))
    """
    attr, checkvalue = checkitem
    value = getattr(fixture, attr)

    assert_array_equal(
        np.asarray(value),
        np.asarray(checkvalue),
        err_msg=f"bad value for {fixture.__class__.__name__}.{attr}",
    )


def test_mpts_metrics():
    mptm = MPTSMetrics(dummy=True, verbose=False, match_type="StructureMatcher")
    for fold in mptm.folds:
        train_val_inputs = mptm.get_train_and_val_data(fold)

        np.random.seed(10)
        dg = DummyGenerator()
        dg.fit(train_val_inputs)
        gen_structures = dg.gen(n=3)

        mptm.evaluate_and_record(fold, gen_structures)

    # TODO: check that all metrics are between 0 and 1
    print(mptm.recorded_metrics)


def test_non_verbose():
    mptm = MPTSMetrics(dummy=True, verbose=False, match_type="StructureMatcher")
    fold = mptm.folds[0]
    train_val_inputs = mptm.get_train_and_val_data(fold)

    np.random.seed(10)
    dg = DummyGenerator()
    dg.fit(train_val_inputs)
    gen_structures = dg.gen(n=3)

    mptm.evaluate_and_record(fold, gen_structures)

    print(mptm.recorded_metrics)


def test_cdvae_coverage():
    mptm = MPTSMetrics(dummy=True, verbose=True, match_type="cdvae_coverage")
    fold = mptm.folds[0]
    train_val_inputs = mptm.get_train_and_val_data(fold)

    np.random.seed(10)
    dg = DummyGenerator()
    dg.fit(train_val_inputs)
    gen_structures = dg.gen(n=3)

    mptm.evaluate_and_record(fold, gen_structures)

    print(mptm.recorded_metrics)


def test_cdvae_cov_struct_fingerprints():
    cdvae_cov_struct_fingerprints(dummy_structures, verbose=True)


# %% Code Graveyard
# def flatten_params(
#     fixtures: List[Callable], expecteds: List[dict]
# ) -> List[Tuple[Callable, str, npt.ArrayLike]]:
#     """Match list of fixtures to list of dicts element-wise; create flat combinations.

#     Based on `pytest nested function parametrization
#     <https://stackoverflow.com/a/62033116/13697228>`_

#     Parameters
#     ----------
#     fixtures : List[Callable]
#         List of fixture functions.
#     expecteds : List[dict]
#         List of dictionaries mapping class variable names to expected values.

#     Returns
#     -------
#         flat_fixtures : List[Callable]
#             List of paired (repeated) fixture functions.
#         flat_attributes : List[str]
#             List of flattened class variable names.
#         flat_values : List[Sequence]
#             List of flattened expected values.

#     Examples
#     --------
#     >>> @fixture
#     >>> def dummy_gen_matcher():
#     >>>    return GenMatcher(dummy_structures, dummy_structures)

#     >>> @fixture
#     >>> def dummy_gen_metrics():
#     >>>    return GenMetrics(dummy_structures, dummy_structures)

#     >>> fixtures = [dummy_gen_matcher, dummy_gen_metrics]
#     >>> expecteds = [{"match_rate": [1.0]}, {"validity": [0.5], "novelty": [0.5], "uniqueness": [0.5]}] # noqa: E501
#     >>> flatten_params(fixtures, expecteds)
#     [(<function dummy_gen_matcher at 0x000001E1A89CED30>, 'match_rate', [1.0]),
#     (<function dummy_gen_metrics at 0x000001E1A89CEF70>, 'validity', [0.5]),
#     (<function dummy_gen_metrics at 0x000001E1A89CEF70>, 'novelty', [0.5]), (<function
#     dummy_gen_metrics at 0x000001E1A89CEF70>, 'uniqueness', [0.5])]

#     See also
#     --------
#     https://stackoverflow.com/a/42400786/13697228
#     """
#     combinations = zip(fixtures, expecteds)
#     combinations: List[Tuple[Callable, str, npt.ArrayLike]] = []
#     for fix, expected in zip(fixtures, expecteds):
#         for attribute, value in expected.items():
#             combinations.append((fix, attribute, value))
#     return combinations


# combinations = flatten_params(fixtures, expecteds)
