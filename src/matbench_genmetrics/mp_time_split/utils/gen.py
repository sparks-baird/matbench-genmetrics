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
        """
        This function generates a list of pymatgen Structure objects by creating
        random crystals using the pyxtal library. Each crystal is composed of
        Ba, Ti, and O in a 1:1:3 ratio.

        Parameters
        ----------
        n : int, optional
            The number of structures to generate, by default 100.

        Returns
        -------
        List[Structure]
            A list of pymatgen Structure objects.

        Examples
        --------
        >>> structures = DummyGenerator().gen(n=100)
        """
        crystal = pyxtal()
        structures = []
        for _ in range(n):
            crystal.from_random(3, 99, ["Ba", "Ti", "O"], [1, 1, 3])
            structures.append(crystal.to_pymatgen())
        return structures
