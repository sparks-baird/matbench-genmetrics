from typing import Callable, List, Sequence, Tuple

import numpy as np
from numpy.testing import ArrayLike, assert_array_equal
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pytest_cases import fixture, parametrize

from matbench_genmetrics.core import GenMatcher, GenMetrics

coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60)
dummy_structures = [
    Structure(lattice, ["Si", "Si"], coords),
    Structure(lattice, ["Ni", "Ni"], coords),
]


@fixture
def dummy_gen_matcher():
    """Get GenMatcher instance with dummy_structures as both test and gen structures."""
    return GenMatcher(dummy_structures, dummy_structures)


@fixture
def dummy_gen_metrics():
    """Get GenMetrics instance with dummy_structures as train/test/gen structures."""
    return GenMetrics(dummy_structures, dummy_structures)


dummy_matcher_expected = {
    "match_matrix": [[1.0, 0.0], [0.0, 1.0]],
    "match_counts": [1.0, 1.0],
    "match_count": [2.0],
    "match_rate": [1.0],
    "duplicity_counts": [0.0, 0.0],
    "duplicity_count": [0.0],
    "duplicity_rate": [0.0],
}

dummy_metrics_expected = {"validity": [0.5], "novelty": [0.5], "uniqueness": [0.5]}

fixtures = [dummy_gen_matcher, dummy_gen_metrics]
expecteds: List[dict] = [dummy_matcher_expected, dummy_metrics_expected]
attrs = [list(dummy_matcher_expected.keys()), list(dummy_metrics_expected.keys())]
values = [list(dummy_matcher_expected.values()), list(dummy_metrics_expected.values())]


def flatten_params(
    fixtures: List[Callable], expecteds: List[dict]
) -> Tuple[List[Callable], List[str], List[Sequence]]:
    """Match list of fixtures to list of dicts element-wise; create flat combinations.

    Based on `pytest nested function parametrization
    <https://stackoverflow.com/a/62033116/13697228>`_

    Parameters
    ----------
    fixtures : List[Callable]
        List of fixture functions.
    expecteds : List[dict]
        List of dictionaries mapping class variable names to expected values.

    Returns
    -------
        flat_fixtures : List[Callable]
            List of paired (repeated) fixture functions.
        flat_attributes : List[str]
            List of flattened class variable names.
        flat_values : List[Sequence]
            List of flattened expected values.

    Examples
    --------
    >>> @fixture
    >>> def dummy_gen_matcher():
    >>>    return GenMatcher(dummy_structures, dummy_structures)

    >>> @fixture
    >>> def dummy_gen_metrics():
    >>>    return GenMetrics(dummy_structures, dummy_structures)

    >>> fixtures = [dummy_gen_matcher, dummy_gen_metrics]
    >>> expecteds = [{"match_rate": [1.0]}, {"validity": [0.5], "novelty": [0.5], "uniqueness": [0.5]}] # noqa: E501
    >>> flatten_params(fixtures, expecteds)
    ((<function dummy_gen_matcher at 0x0000019FDEDC81F0>, <function dummy_gen_metrics at 0x0000019FDEDC8430>, <function dummy_gen_metrics at 0x0000019FDEDC8430>, <function dummy_gen_metrics at 0x0000019FDEDC8430>), ('match_rate', 'validity', 'novelty', 'uniqueness'), ([1.0], [0.5], [0.5], [0.5])) # noqa: E501

    See also
    --------
    https://stackoverflow.com/a/42400786/13697228
    """
    combinations = []
    for fix, expected in zip(fixtures, expecteds):
        for attribute, value in expected.items():
            combinations.append([fix, attribute, value])
    flat_fixtures, flat_attributes, flat_values = zip(*combinations)
    return flat_fixtures, flat_attributes, flat_values


flat_fixtures, flat_attributes, flat_values = flatten_params(fixtures, expecteds)


@parametrize(
    fixture=flat_fixtures, attr=flat_attributes, check_value=flat_values, indirect=True
)
def test_numerical_attributes(fixture: Callable, attr: str, check_value: ArrayLike):
    """Verify that numerical attributes match the expected values.

    Note that scalars are converted to numpy arrays before comparison.

    Parameters
    ----------
    fixture : Callable
        a pytest fixture that returns an instantiated class operable with getattr
    attr : str
        the attribute to test, e.g. "match_matrix"
    check_value : np.ndarray
        the expected value of the attribute checked via ``assert_array_equal``

    Examples
    --------
    >>> test_numerical_attributes(dummy_gen_metrics, "match_count", expected)
    """
    value = getattr(fixture, attr)
    value = np.array(value) if not isinstance(value, np.ndarray) else value

    assert_array_equal(
        value,
        np.array(check_value),
        err_msg=f"bad value for {dummy_gen_matcher.__class__.__name__}.{attr}",
    )
