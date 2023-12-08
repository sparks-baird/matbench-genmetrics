# Helped me to resolve an error related to the creation of the fixtures
import numpy as np
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.transformations.standard_transformations import (
    PerturbStructureTransformation,
)

from matbench_genmetrics.core.metrics import GenMatcher, GenMetrics, MPTSMetrics

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

1 + 1


# @fixture
def dummy_gen_matcher():
    """Get GenMatcher instance with dummy_structures as both test and gen structures."""
    dummy_structures_2 = dummy_structures.copy()
    return GenMatcher(
        dummy_structures, dummy_structures_2, match_type="StructureMatcher"
    )


# @fixture
def test_dummy_gen_metrics():
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
def test_dummy_mpts_metrics():
    """Get MPTSMetrics() with dummy MPTS as train/test, dummy_structures as pred/gen."""
    mptm = MPTSMetrics(dummy=True, match_type="StructureMatcher")

    fold = 0
    mptm.get_train_and_val_data(fold)
    mptm.evaluate_and_record(fold, dummy_structures)

    return mptm


if __name__ == "__main__":
    test_dummy_mpts_metrics()
