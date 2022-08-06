import logging
import warnings

import numpy as np
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.structure.sites import SiteStatsFingerprint
from mp_time_split.core import MPTimeSplit

from matbench_genmetrics.utils.match import get_tqdm


def featurize_comp_struct(
    dummy=False,
    comp_name="composition",
    struct_name="structure",
    material_id_name="material_id",
    include_pmg_object=False,
    keep_as_df=False,
):
    cnnf = CrystalNNFingerprint.from_preset("ops")
    ep = ElementProperty.from_preset("magpie")
    ssf = SiteStatsFingerprint(cnnf, stats=("mean"))
    mpt = MPTimeSplit(target="energy_above_hull")

    mpt.load(dummy=dummy)

    structures = mpt.data.structure
    compositions = structures.apply(lambda s: s.composition)
    material_ids = mpt.data.material_id

    structures = structures.to_frame(struct_name)
    compositions = compositions.to_frame(comp_name)

    structures.index = material_ids
    compositions.index = material_ids
    structures.index.name = material_id_name
    compositions.index.name = material_id_name

    comp_fingerprints = ep.featurize_dataframe(compositions, comp_name)

    struct_fingerprints = ssf.featurize_dataframe(
        structures, struct_name, ignore_errors=True
    )

    if not include_pmg_object:
        comp_fingerprints.drop(comp_name, axis=1, inplace=True)
        struct_fingerprints.drop(struct_name, axis=1, inplace=True)

    if not keep_as_df:
        comp_fingerprints = comp_fingerprints.values
        struct_fingerprints = struct_fingerprints.values

    return comp_fingerprints, struct_fingerprints


def cdvae_cov_comp_fingerprints(structures, verbose=False):
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

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def cdvae_cov_struct_fingerprints(structures, verbose=False):
    CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
    """Use SiteStatsFingerprint if OK with NaN rows upon partial site failures."""
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
