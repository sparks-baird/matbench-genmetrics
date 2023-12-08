import logging
import warnings
from typing import List

import numpy as np
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
from scipy.spatial.distance import cdist, pdist, squareform
from tqdm import tqdm
from tqdm.notebook import tqdm as ipython_tqdm

warnings.filterwarnings(
    "ignore",
    message="No oxidation states specified on sites! For better results, set the site oxidation states in the structure.",  # noqa: E501
)
warnings.filterwarnings(
    "ignore",
    message="CrystalNN: cannot locate an appropriate radius, covalent or atomic radii will be used, this can lead to non-optimal results.",  # noqa: E501
)

# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sm = StructureMatcher(stol=0.5, ltol=0.3, angle_tol=10.0)

# https://stackoverflow.com/a/58102605/13697228
is_notebook = hasattr(__builtins__, "__IPYTHON__")


def dummy_tqdm(x, **kwargs):  # noqa: E731
    """
    A dummy function that simply returns its input. Used as a placeholder for
    the tqdm progress bar function when verbose mode is not enabled.
    """
    return x


def get_tqdm(verbose):
    """
    Returns the appropriate tqdm function based on the environment and
    verbosity. If verbose is False, returns a dummy function that does nothing.

    Parameters
    ----------
    verbose : bool
        If True, returns the tqdm function. If False, returns a dummy function.
    """
    if verbose:
        return ipython_tqdm if is_notebook else tqdm
    else:
        return dummy_tqdm


def structure_matcher(s1: Structure, s2: Structure):
    """
    Checks if two pymatgen Structure objects match according to pymatgen's
    StructureMatcher criteria with the following settings:

    StructureMatcher(stol=0.5, ltol=0.3, angle_tol=10.0)

    Parameters
    ----------
    s1 : Structure
        The first structure to compare.
    s2 : Structure
        The second structure to compare.

    Returns
    -------
    bool
        True if the structures match, False otherwise.
    """
    return sm.fit(s1, s2)


pairwise_match_fn_dict = {"StructureMatcher": structure_matcher}


def structure_pairwise_match_matrix(
    test_structures: List[Structure],
    gen_structures: List[Structure],
    match_type: str = "StructureMatcher",
    verbose: bool = False,
    symmetric: bool = False,
):
    """
    This function computes a pairwise match matrix between two lists of pymatgen
    Structure objects using a specified match function. The match function is
    determined by the `match_type` parameter.

    Parameters
    ----------
    test_structures : List[Structure]
        List of pymatgen Structure objects to be compared.
    gen_structures : List[Structure]
        List of pymatgen Structure objects to be compared.
    match_type : str, optional
        The type of match function to use, by default "StructureMatcher".
    verbose : bool, optional
        If True, the function will provide a running progress bar, by default
        False.
    symmetric : bool, optional
        If True, the function will compute a symmetric match matrix else an
        array in the style of cdist, by default False.
    """
    # TODO: replace with group_structures to be faster
    pairwise_match_fn = pairwise_match_fn_dict[match_type]
    match_matrix = np.zeros((len(test_structures), len(gen_structures)))
    if verbose:
        #     logger.setLevel(logging.DEBUG)
        logger.info(f"Computing {match_type} match matrix pairwise")

    my_tqdm = get_tqdm(verbose)
    for i, ts in enumerate(my_tqdm(test_structures)):
        for j, gs in enumerate(gen_structures):
            if not symmetric or (symmetric and i < j):
                match_matrix[i, j] = pairwise_match_fn(ts, gs)
    if symmetric:
        match_matrix = match_matrix + match_matrix.T
    return match_matrix


CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
bva = BVAnalyzer()


def cdvae_cov_match_matrix(
    test_fingerprints: List[List[float]],
    gen_fingerprints: List[List[float]],
    symmetric: bool = False,
    cutoff: float = 10.0,
):
    """
    This function computes a match matrix between two sets of fingerprints (test
    and generated) based on a cutoff distance.

    Parameters
    ----------
    test_fingerprints : List[List[float]]
        List of test fingerprints to be compared. Each fingerprint is a list of
        floats.
    gen_fingerprints : List[List[float]]
        List of generated fingerprints to be compared. Each fingerprint is a
        list of floats.
    symmetric : bool, optional
        If True, the function will compute a symmetric match matrix, else an
        array in the style of cdist, by default False.
    cutoff : float, optional
        The cutoff distance for matching fingerprints, by default 10.0.

    Returns
    -------
    np.ndarray
        A numpy array representing the match matrix (either squareform(pdist) or
        cdist style depending on `symmetric` arg). Each entry is a boolean
        indicating whether the corresponding pair of fingerprints match (True)
        or not (False).
    """
    if symmetric:
        dm = squareform(pdist(test_fingerprints))
    else:
        dm = cdist(test_fingerprints, gen_fingerprints)
    return dm <= cutoff


def cdvae_cov_compstruct_match_matrix(
    test_comp_fingerprints: List[List[float]],
    gen_comp_fingerprints: List[List[float]],
    test_struct_fingerprints: List[List[float]],
    gen_struct_fingerprints: List[List[float]],
    symmetric: bool = False,
    comp_cutoff: float = 10.0,
    struct_cutoff: float = 0.4,
    verbose: bool = False,
):
    """
    This function computes a match matrix between two sets of composition and
    structure fingerprints (test and generated) based on specified cutoff
    distances.

    Parameters
    ----------
    test_comp_fingerprints : List[List[float]]
        List of test composition fingerprints to be compared. Each fingerprint
        is a list of floats.
    gen_comp_fingerprints : List[List[float]]
        List of generated composition fingerprints to be compared. Each
        fingerprint is a list of floats.
    test_struct_fingerprints : List[List[float]]
        List of test structure fingerprints to be compared. Each fingerprint is
        a list of floats.
    gen_struct_fingerprints : List[List[float]]
        List of generated structure fingerprints to be compared. Each
        fingerprint is a list of floats.
    symmetric : bool, optional
        If True, the function will compute a symmetric match matrix, otherwise a
        list in the style of cdist, by default False.
    comp_cutoff : float, optional
        The cutoff distance for matching composition fingerprints, by default
        10.0.
    struct_cutoff : float, optional
        The cutoff distance for matching structure fingerprints, by default 0.4.
    verbose : bool, optional
        If True, the function will provide a running progress bar, by default
        False.
    """
    if verbose:
        #     logger.setLevel(logging.DEBUG)
        logger.info("Computing composition match matrix")
    comp_match_matrix = cdvae_cov_match_matrix(
        test_comp_fingerprints,
        gen_comp_fingerprints,
        symmetric=symmetric,
        cutoff=comp_cutoff,
    )
    if verbose:
        logger.info("Computing structure match matrix")
    struct_match_matrix = cdvae_cov_match_matrix(
        test_struct_fingerprints,
        gen_struct_fingerprints,
        symmetric=symmetric,
        cutoff=struct_cutoff,
    )
    # multiply, since 0*0=0, 0*1=0, 1*0=0, 1*1=1
    return comp_match_matrix * struct_match_matrix


ALLOWED_MATCH_TYPES = ["StructureMatcher", "cdvae_coverage"]


def get_structure_match_matrix(
    test_structures: List[Structure],
    gen_structures: List[Structure],
    match_type: str = "cdvae_coverage",
    symmetric: bool = False,
    verbose: bool = False,
    **match_kwargs,
):
    """
    This function computes a match matrix between two lists of pymatgen
    Structure objects using a specified match function. The match function is
    determined by the `match_type` parameter.

    Parameters
    ----------
    test_structures : List[Structure]
        List of pymatgen Structure objects to be compared.
    gen_structures : List[Structure]
        List of pymatgen Structure objects to be compared.
    match_type : str, optional
        The type of match function to use, by default "cdvae_coverage".
    symmetric : bool, optional
        If True, the function will compute a symmetric match matrix, else an
        array in the style of cdist, by default False.
    verbose : bool, optional
        If True, the function will provide a running progress bar, by default
        False.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the match matrix.

    Raises
    ------
    ValueError
        If the `match_type` is not recognized.
    """
    if match_type == "StructureMatcher":
        return structure_pairwise_match_matrix(
            test_structures,
            gen_structures,
            match_type="StructureMatcher",
            symmetric=symmetric,
            verbose=verbose,
            **match_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown match type {match_type}. Must be one of {ALLOWED_MATCH_TYPES}"
        )  # noqa: E501


def get_fingerprint_match_matrix(
    test_comp_fingerprints: List[List[float]],
    gen_comp_fingerprints: List[List[float]],
    test_struct_fingerprints: List[List[float]],
    gen_struct_fingerprints: List[List[float]],
    match_type: str = "cdvae_coverage",
    symmetric: bool = False,
    verbose: bool = False,
    **match_kwargs,
):
    """
    This function computes a match matrix between two sets of composition and
    structure fingerprints (test and generated) using a specified match
    function. The match function is determined by the `match_type` parameter.

    Parameters
    ----------
    test_comp_fingerprints : List[List[float]]
        List of test composition fingerprints to be compared. Each fingerprint
        is a list of floats.
    gen_comp_fingerprints : List[List[float]]
        List of generated composition fingerprints to be compared. Each
        fingerprint is a list of floats.
    test_struct_fingerprints : List[List[float]]
        List of test structure fingerprints to be compared. Each fingerprint is
        a list of floats.
    gen_struct_fingerprints : List[List[float]]
        List of generated structure fingerprints to be compared. Each
        fingerprint is a list of floats.
    match_type : str, optional
        The type of match function to use, by default "cdvae_coverage".
    symmetric : bool, optional
        If True, the function will compute a symmetric match matrix else an
        array in the cdist style, by default False.
    verbose : bool, optional
        If True, the function will provide a running progress bar, by default
        False.

    Returns
    -------
    np.ndarray
        A 2D numpy array representing the match matrix.

    Raises
    ------
    ValueError
        If the `match_type` is not recognized.
    """
    if match_type == "cdvae_coverage":
        return cdvae_cov_compstruct_match_matrix(
            test_comp_fingerprints,
            gen_comp_fingerprints,
            test_struct_fingerprints,
            gen_struct_fingerprints,
            symmetric=symmetric,
            verbose=verbose,
            **match_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown match type {match_type}. Must be one of {ALLOWED_MATCH_TYPES}"
        )  # noqa: E501
