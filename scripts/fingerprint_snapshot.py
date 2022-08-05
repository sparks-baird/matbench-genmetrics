from mp_time_split.core import MPTimeSplit

from matbench_genmetrics.utils.match import (
    cdvae_cov_comp_fingerprints,
    cdvae_cov_struct_fingerprints,
)

mpt = MPTimeSplit(target="energy_above_hull")

mpt.load(dummy=True)

original_inputs = mpt.inputs

comp_fingerprints = cdvae_cov_comp_fingerprints(mpt.inputs)
struct_fingerprints = cdvae_cov_struct_fingerprints(mpt.inputs)

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
