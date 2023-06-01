try:
    from pyxtal import pyxtal
except ImportError as e:
    print(e)
    print(
        "Failed to import pyxtal. Try `pip install mp_time_split[pyxtal]` or `pip install pyxtal`"  # noqa: E501
    )


class DummyGenerator:
    def __init__(self):
        pass

    def fit(self, inputs):
        inputs

    def gen(self, n=100):
        crystal = pyxtal()
        structures = []
        for _ in range(n):
            crystal.from_random(3, 99, ["Ba", "Ti", "O"], [1, 1, 3])
            structures.append(crystal.to_pymatgen())
        return structures
