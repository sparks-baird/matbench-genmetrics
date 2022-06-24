from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

coords = [[0, 0, 0], [0.75, 0.5, 0.75]]
lattice = Lattice.from_parameters(a=3.84, b=3.84, c=3.84, alpha=120, beta=90, gamma=60)
dummy_structures = [
    Structure(lattice, ["Si", "Si"], coords),
    Structure(lattice, ["Ni", "Ni"], coords),
]

# compare the metrics of the structures against themselves

# compare the metrics of one set of structures against another
