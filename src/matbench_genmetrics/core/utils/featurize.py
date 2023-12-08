import logging
import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
from element_coder import encode
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.structure.sites import SiteStatsFingerprint
from pymatgen.core.structure import Structure

from matbench_genmetrics.core.utils.match import get_tqdm

cnnf = CrystalNNFingerprint.from_preset("ops")
ep = ElementProperty.from_preset("magpie")
ssf = SiteStatsFingerprint(cnnf, stats=("mean"))


def featurize_comp_struct(
    structures: List[Structure],
    material_ids: Optional[List] = None,
    comp_name="composition",
    struct_name="structure",
    material_id_name="material_id",
    include_pmg_object=False,
    keep_as_df=False,
):
    """
    This function takes a list of structures and optional material IDs, and
    generates composition and structure fingerprints using different types of
    featurizers.

    The composition fingerprints are generated using the ElementProperty
    featurizer from the matminer library, which uses a preset of "magpie" to
    generate a set of element property features. The structure fingerprints are
    generated using matminer's SiteStatsFingerprint featurizer, which uses a
    CrystalNNFingerprint instance (with a preset of "ops") to generate site
    fingerprints, and then calculates the mean of these fingerprints.

    The function also allows for customization of the names of the composition,
    structure, and material ID columns. It can optionally include the pymatgen
    object in the output and can return the data as a dataframe or as a numpy
    array.

    Parameters
    ----------
    structures : List[Structure]
        List of pymatgen Structure objects to be featurized.
    material_ids : Optional[List], optional
        List of material IDs corresponding to the structures, by default None.
    comp_name : str, optional
        Name to use for the composition column, by default "composition".
    struct_name : str, optional
        Name to use for the structure column, by default "structure".
    material_id_name : str, optional
        Name to use for the material ID column, by default "material_id".
    include_pmg_object : bool, optional
        Whether to include the pymatgen object in the output, by default False.
    keep_as_df : bool, optional
        Whether to keep the output as a dataframe, by default False. If False,
        the output will be a numpy array.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple of two dataframes: the first contains the composition
        fingerprints, and the second contains the structure fingerprints.

    Examples
    --------
    >>> comp_fingerprints, struct_fingerprints = featurize_comp_struct(structures, material_ids=None, comp_name="composition", struct_name="structure", material_id_name="material_id", include_pmg_object=False, keep_as_df=False) # noqa: E501
    """
    S = pd.Series(structures)
    compositions = S.apply(lambda s: s.composition)

    S = S.to_frame(struct_name)
    compositions = compositions.to_frame(comp_name)

    if material_ids:
        S.index = pd.Index(material_ids)
        compositions.index = pd.Index(material_ids)
        S.index.name = material_id_name
        compositions.index.name = material_id_name

    comp_fingerprints = ep.featurize_dataframe(compositions, comp_name)

    struct_fingerprints = ssf.featurize_dataframe(S, struct_name, ignore_errors=True)

    if not include_pmg_object:
        comp_fingerprints.drop(comp_name, axis=1, inplace=True)
        struct_fingerprints.drop(struct_name, axis=1, inplace=True)

    if not keep_as_df:
        comp_fingerprints = comp_fingerprints.values
        struct_fingerprints = struct_fingerprints.values

    return comp_fingerprints, struct_fingerprints


def mod_petti_contributions(structures: List[Structure]):
    """
    This function takes a list of pymatgen Structure objects and calculates the
    modified Pettifor number contributions for each element in the structures.

    The modified Pettifor number is a measure of the electronegativity of an
    element in a specific structure. The function returns a dataframe sorted by
    the modified Pettifor number.

    Parameters
    ----------
    structures : List[Structure]
        List of pymatgen Structure objects for which to calculate the modified
        Pettifor number contributions.

    Returns
    -------
    pd.DataFrame
        A dataframe with two columns: 'mod_petti', which contains the modified
        Pettifor numbers, and 'contribution', which contains the corresponding
        contributions of each element in the structures. The dataframe is sorted
        by the 'mod_petti' column.

    Examples
    --------
    >>> mod_petti_df = mod_petti_contributions(structures)
    """
    compositions = pd.Series(structures).apply(
        lambda s: s.composition.fractional_composition
    )
    # NOTE: be aware of amount_tolerance=1e-8
    summed_comp = np.sum(compositions).fractional_composition
    _data = summed_comp._data
    mod_petti = [encode(k, "mod_pettifor") for k in _data.keys()]
    mod_petti_comp = dict(zip(mod_petti, _data.values()))
    mod_petti_df = pd.DataFrame(
        dict(mod_petti=mod_petti_comp.keys(), contribution=mod_petti_comp.values()),
    ).sort_values("mod_petti")
    return mod_petti_df


def cdvae_cov_comp_fingerprints(structures: List[Structure], verbose: bool = False):
    """
    This function takes a list of pymatgen Structure objects and generates
    composition fingerprints for each structure using the ElementProperty
    featurizer from the matminer library.

    The featurizer uses a preset of "magpie" to generate a set of element
    property features. The function also has a verbose mode that, when enabled,
    provides a running progress bar.

    Parameters
    ----------
    structures : List[Structure]
        List of pymatgen Structure objects to be featurized.
    verbose : bool, optional
        If True, the function will provide more detailed output, by default
        False.

    Returns
    -------
    List[List[float]]
        A list of lists, where each inner list contains the composition
        fingerprints for a structure.

    Examples
    --------
    >>> fingerprints = cdvae_cov_comp_fingerprints(structures, verbose=False)
    """
    my_tqdm = get_tqdm(verbose)
    CompFP = ElementProperty.from_preset("magpie")
    return [CompFP.featurize(s.composition) for s in my_tqdm(structures)]


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


def cdvae_cov_struct_fingerprints(structures: List[Structure], verbose: bool = False):
    """
    This function takes a list of pymatgen Structure objects and generates
    structure fingerprints for each structure using the CrystalNNFingerprint
    featurizer from the matminer library.

    The featurizer uses a preset of "ops" to generate a set of site
    fingerprints, and then calculates the mean of these fingerprints. The
    function also has a verbose mode that, when enabled, provides more detailed
    output. If a structure fails to featurize, it is replaced with NaN values.


    The function is based on an implementation in CDVAE:
    https://github.com/txie-93/cdvae.

    Parameters
    ----------
    structures : List[Structure]
        List of pymatgen Structure objects to be featurized.
    verbose : bool, optional
        If True, the function will provide more detailed output, by default
        False.

    Returns
    -------
    List[List[float]]
        A list of lists, where each inner list contains the structure
        fingerprints for a structure.

    Examples
    --------
    >>> fingerprints = cdvae_cov_struct_fingerprints(structures, verbose=False)
    """
    # Aside: Use SiteStatsFingerprint if OK with NaN rows upon partial site
    # failures.
    CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
    my_tqdm = get_tqdm(verbose)
    struct_fps = []
    num_sites = []
    num_failed_sites = []
    failed_structures = []
    for s in my_tqdm(structures):
        site_fps = []
        exceptions = []
        ns = len(s)
        num_sites.append(ns)
        for i in range(ns):
            try:
                site_fp = CrystalNNFP.featurize(s, i)
                site_fps.append(site_fp)
            except Exception as e:
                exceptions.append(f"site {i}: {str(e)}")

        num_failed_sites.append(len(exceptions))
        if exceptions:
            exception_strs = "\n".join(exceptions)
            logger.warning(
                f"{len(exception_strs)} exceptions encountered for structure {i}:\n{s}\n The exceptions are:\n{exception_strs}"  # noqa: E501
            )
        if site_fps:
            struct_fp = np.array(site_fps).mean(axis=0)
        else:
            failed_structures.append(s)
            # NaN vector https://stackoverflow.com/a/1704853/13697228
            num_features = len(CrystalNNFP.feature_labels())
            struct_fp = np.empty(num_features)
            struct_fp[:] = np.nan
        struct_fps.append(struct_fp)
    if num_failed_sites:
        fail_rate = np.array(num_failed_sites) / np.array(num_sites)
        avg_fail_rate = np.mean(fail_rate[fail_rate > 0])
        logger.warning(
            f"{len(num_failed_sites)} structures partially failed to featurize, with on average {avg_fail_rate:.2f} site failure rate per failed structure, and where failed sites were ignored during averaging."  # noqa: E501
        )
    if failed_structures:
        failed_structure_strs = [str(fs) for fs in failed_structures]
        if len(failed_structure_strs) < 10:
            failed_structure_str = "\n".join(failed_structure_strs)
        else:
            failed_structure_str = "\n".join(failed_structure_strs[:10])
        logger.warning(
            f"{len(failed_structures)} structures totally failed to featurize. These were replaced with NaN values. Up to the first 10 structures are displayed here: {failed_structure_str}"  # noqa: E501
        )

    return struct_fps
