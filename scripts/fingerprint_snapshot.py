from os import path
from pathlib import Path

from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.structure.sites import SiteStatsFingerprint
from mp_time_split.core import MPTimeSplit

comp_name = "composition"
struct_name = "structure"
material_id_name = "material_id"

dummy = False


def featurize_comp_struct():
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

    return comp_fingerprints, struct_fingerprints


if __name__ == "__main__":
    comp_fingerprints, struct_fingerprints = featurize_comp_struct()
    data_dir = path.join("data", "interim")
    if dummy:
        data_dir = path.join(data_dir, "dummy")
    Path(data_dir).mkdir(exist_ok=True, parents=True)
    comp_fingerprints.drop(comp_name, axis=1).to_csv(
        path.join(data_dir, "comp_fingerprints.csv")
    )
    struct_fingerprints.drop(struct_name, axis=1).to_csv(
        path.join(data_dir, "struct_fingerprints.csv")
    )

# comp_fingerprints = np.array(cdvae_cov_comp_fingerprints(mpt.inputs, verbose=True))
# struct_fingerprints = np.array(cdvae_cov_struct_fingerprints(mpt.inputs,
# verbose=True))

# save the fingerprints and upload to figshare

# %% Code Graveyard
# (
#     train_struct_fingerprints,
#     val_struct_fingerprints,
#     train_outputs,
#     val_outputs,
# ) = mpt.get_train_and_val_data(fold)

# mpt.inputs = cdvae_cov_struct_fingerprints(original_inputs)

# (
#     train_struct_fingerprints,
#     val_struct_fingerprints,
#     train_outputs,
#     val_outputs,
# ) = mpt.get_train_and_val_data(fold)

# from matbench_genmetrics.utils.match import (
#     cdvae_cov_comp_fingerprints,
#     cdvae_cov_struct_fingerprints,
# )
