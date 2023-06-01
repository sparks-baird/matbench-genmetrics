from os import path

from matminer.utils.io import load_dataframe_from_json, store_dataframe_as_json

from matbench_genmetrics.mp_time_split.splitter import MPTimeSplit, get_data_home
from matbench_genmetrics.mp_time_split.utils.data import (
    DUMMY_SNAPSHOT_NAME,
    SNAPSHOT_NAME,
)

# %% dummy data
mpt = MPTimeSplit(num_sites=(1, 2), elements=["V"])
dummy_expt_df = mpt.fetch_data(one_by_one=True)
dummy_data_path = path.join(get_data_home(), DUMMY_SNAPSHOT_NAME)

store_dataframe_as_json(dummy_expt_df, dummy_data_path, compression=None)
store_dataframe_as_json(dummy_expt_df, dummy_data_path + ".gz", compression="gz")

dummy_expt_df_check = load_dataframe_from_json(dummy_data_path)

dummy_match = dummy_expt_df.compare(dummy_expt_df_check)
if not dummy_match.empty:
    raise ValueError(f"dummy_expt_df and dummy_expt_df_check unmatched: {dummy_match}")

# %% full data
mpt = MPTimeSplit(num_sites=(1, 52))
expt_df = mpt.fetch_data()
data_path = path.join(get_data_home(), SNAPSHOT_NAME)
store_dataframe_as_json(expt_df, data_path, compression=None)
store_dataframe_as_json(expt_df, data_path + ".gz", compression="gz")
expt_df_check = load_dataframe_from_json(dummy_data_path)

match = dummy_expt_df.compare(dummy_expt_df_check)
if not match.empty:
    raise ValueError(f"expt_df and expt_df_check unmatched: {match}")

1 + 1


# %% Code Graveyard
# expt_df.to_json()
# store_dataframe_as_json(expt_df, data_path, compression="gz")
# with zopen(data_path, "wb") as f:
#     data = json.dumps(expt_df.to_json()).encode()
#     f.write(data)

# with open(dummy_data_path, "w") as f:
# json.dumps(dummy_expt_df, cls=MontyEncoder)
# dummy_expt_df.structure = dummy_expt_df.structure.apply(lambda s: s.as_dict())
# f.write(jsonpickle.encode(dummy_expt_df))

# with open(dummy_data_path, "r") as f:
# json_string = f.read()
# dummy_expt_df_check = json.loads(json_string, cls=MontyDecoder)
# dummy_expt_df_check = jsonpickle.decode(json_string)
# dummy_expt_df_check.structure =
# dummy_expt_df_check.structure.apply(Structure.from_dict)

# with open(dummy_data_path, "w") as f:
#     f.write(jsonpickle.encode(dummy_expt_df))

# import jsonpickle
# import jsonpickle.ext.pandas as jsonpickle_pandas
# jsonpickle_pandas.register_handlers()

# https://stackoverflow.com/a/4359298/13697228
