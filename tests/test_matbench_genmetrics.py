import numpy as np
from numpy.testing import assert_array_equal
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from matbench_genmetrics.core import get_match_counts, get_match_matrix, get_match_rate

coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60)
dummy_structures = [
    Structure(lattice, ["Si", "Si"], coords),
    Structure(lattice, ["Ni", "Ni"], coords),
]


def test_get_match_matrix():
    match_matrix = get_match_matrix(dummy_structures, dummy_structures)
    match_matrix_check = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert_array_equal(match_matrix, match_matrix_check)
    return match_matrix


def test_get_match_counts():
    match_counts = get_match_counts(dummy_structures, dummy_structures)
    assert_array_equal(match_counts, np.array([1.0, 1.0]))
    return match_counts


def test_get_match_rate():
    match_rate = get_match_rate(dummy_structures, dummy_structures)
    assert_array_equal(np.array(match_rate), np.array([1.0]))
    return match_rate


# def test_get_rms_dist():
#     rms_dist = get_rms_dist(dummy_structures, dummy_structures)
#     return rms_dist


if __name__ == "__main__":
    test_get_match_rate()
    test_get_match_counts()
    test_get_match_matrix()
    # test_get_rms_dist()
    1 + 1

# # compare the metrics of the structures against themselves
# def test_metrics_against_self():
#     for structure in dummy_structures:
#         assert get_metrics(structure) == get_metrics(structure)


# # compare the metrics of one set of structures against another
# def test_metrics_against_others():
#     assert get_metrics(dummy_structures[0]) != get_metrics(dummy_structures[1])
