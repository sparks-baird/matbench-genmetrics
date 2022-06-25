from typing import Callable

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from matbench_genmetrics.core import GenMetrics

coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60)
dummy_structures = [
    Structure(lattice, ["Si", "Si"], coords),
    Structure(lattice, ["Ni", "Ni"], coords),
]


@pytest.fixture
def dummy_gen_metrics():
    return GenMetrics(dummy_structures, dummy_structures)


params = ["attr", "expected"]

expected = [
    ["match_matrix", np.array([[1.0, 0.0], [0.0, 1.0]])],
    ["match_counts", np.array([1.0, 1.0])],
    ["match_count", np.array([[1, 0], [0, 1]])],
    ["match_rate", np.array([1.0])],
    ["duplicity_count", np.array([0.0])],
    ["duplicity_rate", np.array([0.0])],
]


@pytest.parametrize(params, expected)
def test_numerical_attributes(
    dummy_gen_metrics: Callable, attr: str, expected: np.ndarray
):
    """Verify that numerical attributes match the expected values.

    Note that scalars are converted to numpy arrays before comparison.

    Parameters
    ----------
    dummy_gen_metrics : Callable
        a pytest fixture that returns a GenMetrics object
    attr : str
        the attribute to test, e.g. "match_matrix"
    expected : np.ndarray
        the expected value of the attribute checked via ``assert_array_equal``

    Examples
    --------
    >>> test_numerical_attributes(dummy_gen_metrics, "match_count", expected)
    OUTPUT
    """
    value = getattr(dummy_gen_metrics, attr)
    value = np.array(value) if not isinstance(value, np.ndarray) else value
    assert_array_equal(
        value,
        expected,
        err_msg=f"bad value for {dummy_gen_metrics.__name__}.{attr}",
    )


def test_instantiation(dummy_gen_metrics):
    gm = dummy_gen_metrics
    return gm
