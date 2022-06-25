from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from matbench_genmetrics.core import GenMatcher

coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60)
dummy_structures = [
    Structure(lattice, ["Si", "Si"], coords),
    Structure(lattice, ["Ni", "Ni"], coords),
]


@pytest.fixture
def dummy_gen_matcher():
    """Get GenMatcher instance with dummy_structures as both test and gen structures."""
    return GenMatcher(dummy_structures, dummy_structures)


@pytest.fixture
def dummy_gen_metrics():
    """Get GenMetrics instance with dummy_structures as train/test/gen structures."""
    return GenMatcher(dummy_structures, dummy_structures)


params = ["attr", "expected"]

dummy_matcher_expected = [
    ["match_matrix", [[1.0, 0.0], [0.0, 1.0]]],
    ["match_counts", [1.0, 1.0]],
    ["match_count", [2.0]],
    ["match_rate", [1.0]],
    ["duplicity_counts", [0.0, 0.0]],
    ["duplicity_count", [0.0]],
    ["duplicity_rate", [0.0]],
]

dummy_metrics_expected = [
    ["validity", [0.5]],
    ["novelty", [0.5]],
    ["uniqueness", [0.5]],
]

fixtures = [dummy_gen_matcher, dummy_gen_metrics]
expecteds = [dummy_matcher_expected, dummy_metrics_expected]

combs = list(zip(fixtures, [params] * len(fixtures), expecteds))
fixtures, attrs, expecteds = list(zip(*combs))


@pytest.mark.parametrize(["attr", "expected"], [fixtures, attrs, expecteds])
def test_numerical_attributes(fixture: Callable, attr: str, expected: np.ndarray):
    """Verify that numerical attributes match the expected values.

    Note that scalars are converted to numpy arrays before comparison.

    Parameters
    ----------
    fixture : Callable
        a pytest fixture that returns a GenMetrics object
    attr : str
        the attribute to test, e.g. "match_matrix"
    expected : np.ndarray
        the expected value of the attribute checked via ``assert_array_equal``

    Examples
    --------
    >>> test_numerical_attributes(dummy_gen_metrics, "match_count", expected)
    """
    value = getattr(fixture, attr)
    value = np.array(value) if not isinstance(value, np.ndarray) else value

    assert_array_equal(
        value,
        np.array(expected),
        err_msg=f"bad value for {fixture.__class__.__name__}.{attr}",
    )
