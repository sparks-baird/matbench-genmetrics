import logging
import warnings

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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

sm = StructureMatcher(stol=0.5, ltol=0.3, angle_tol=10.0)

# https://stackoverflow.com/a/58102605/13697228
is_notebook = hasattr(__builtins__, "__IPYTHON__")


def dummy_tqdm(x, **kwargs):  # noqa: E731
    return x


def get_tqdm(verbose):
    if verbose:
        return ipython_tqdm if is_notebook else tqdm
    else:
        return dummy_tqdm


def structure_matcher(s1: Structure, s2: Structure):
    return sm.fit(s1, s2)


pairwise_match_fn_dict = {"StructureMatcher": structure_matcher}


def structure_pairwise_match_matrix(
    test_structures,
    gen_structures,
    match_type="StructureMatcher",
    verbose=False,
    symmetric=False,
):
    # TODO: replace with group_structures to be faster
    pairwise_match_fn = pairwise_match_fn_dict[match_type]
    match_matrix = np.zeros((len(test_structures), len(gen_structures)))
    if verbose:
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
    test_fingerprints,
    gen_fingerprints,
    symmetric=False,
    cutoff=10.0,
):
    if symmetric:
        dm = squareform(pdist(test_fingerprints))
    else:
        dm = cdist(test_fingerprints, gen_fingerprints)
    return dm <= cutoff


def cdvae_cov_compstruct_match_matrix(
    test_comp_fingerprints,
    gen_comp_fingerprints,
    test_struct_fingerprints,
    gen_struct_fingerprints,
    symmetric=False,
    comp_cutoff=10.0,
    struct_cutoff=0.4,
    verbose=False,
):
    if verbose:
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
    test_structures,
    gen_structures,
    match_type="cdvae_coverage",
    symmetric=False,
    verbose=False,
    **match_kwargs,
):
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
    test_comp_fingerprints,
    gen_comp_fingerprints,
    test_struct_fingerprints,
    gen_struct_fingerprints,
    match_type="cdvae_coverage",
    symmetric=False,
    verbose=False,
    **match_kwargs,
):
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


# %% Code Graveyard

# if verbose:
#     logger.info("Decorating structures with oxidation states")
# oxi_structures = []
# for s in structures:
#     try:
#         oxi_struct = bva.get_oxi_state_decorated_structure(s)
#     except ValueError:
#         # TODO: track how many couldn't have valences assigned
#         oxi_struct = s
#     oxi_structures.append(oxi_struct)

# fingerprint_fn = (
#     cdvae_cov_comp_fingerprints
#     if composition_only
#     else cdvae_cov_struct_fingerprints
# )

# site_fps = [CrystalNNFP.featurize(s, i) for i in range(len(s))]

# base_10_check = [10 ** j for j in range(0, 20)]
# if i in base_10_check == 0:
#     logger.info(f"{time()} Struct fingerprint {i}/{len(structures)}")
