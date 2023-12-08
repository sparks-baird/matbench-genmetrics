# from ase.visualize import view
# from pymatgen.io.ase import AseAtomsAdaptor
import matplotlib.pyplot as plt
import numpy as np
from pymatviz.structure_viz import plot_structure_2d

# def plot_structure_3d(structure):
#     view(AseAtomsAdaptor.get_atoms(structure))


def plot_structures_2d(structures, nrows, ncols, seed=10, formula_as_title=True):
    """
    This function plots 2D representations of a list of pymatgen Structure
    objects. If the number of structures is greater than the number of subplots,
    it randomly selects a subset of structures to plot.

    Parameters
    ----------
    structures : List[Structure]
        List of pymatgen Structure objects to be plotted.
    nrows : int
        Number of rows in the subplot grid.
    ncols : int
        Number of columns in the subplot grid.
    seed : int, optional
        Seed for the random number generator, by default 10.
    formula_as_title : bool, optional
        If True, the reduced formula of the structure is used as the title of
        the subplot, by default True.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        A tuple containing the matplotlib Figure object and an array of Axes
        objects.
    """
    if len(structures) > nrows * ncols:
        # get random structures
        plot_structures = np.random.RandomState(seed=seed).choice(
            structures, size=nrows * ncols, replace=False
        )
    else:
        plot_structures = structures

    fig, axes = plt.subplots(nrows, ncols)

    for s, ax in zip(plot_structures, axes.flatten()):
        plot_structure_2d(s, ax=ax)
        if formula_as_title:
            formula = s.composition.reduced_formula
            if len(formula) > 15:
                formula = formula[0:7] + ".." + formula[-7:]
            ax.set_title(formula)

    return fig, axes


def plot_images(images, nrows, ncols, seed=10, formula_as_title=True):
    """
    This function plots a list of images. If the number of images is greater
    than the number of subplots, it randomly selects a subset of images to plot.

    Parameters
    ----------
    images : List[np.ndarray]
        List of images to be plotted. Each image is a numpy array.
    nrows : int
        Number of rows in the subplot grid.
    ncols : int
        Number of columns in the subplot grid.
    seed : int, optional
        Seed for the random number generator, by default 10.
    formula_as_title : bool, optional
        If True, the reduced formula of the structure corresponding to the image
        is used as the title of the subplot, by default True.

    Returns
    -------
    Tuple[plt.Figure, np.ndarray]
        A tuple containing the matplotlib Figure object and an array of Axes
        objects.
    """
    if len(images) > nrows * ncols:
        # get random structures
        plot_images = np.random.RandomState(seed=seed).choice(
            images, size=nrows * ncols, replace=False
        )
    else:
        plot_images = images

    fig, axes = plt.subplots(nrows, ncols)

    for s, ax in zip(plot_images, axes.flatten()):
        ax.imshow(s)
        if formula_as_title:
            formula = s.composition.reduced_formula
            if len(formula) > 15:
                formula = formula[0:7] + ".." + formula[-7:]
            ax.set_title(formula)

    return fig, axes
