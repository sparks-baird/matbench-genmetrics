import sys
from os import path

import pytest
from matminer.utils.io import load_dataframe_from_json

from matbench_genmetrics.mp_time_split.splitter import MPTimeSplit, get_data_home
from matbench_genmetrics.mp_time_split.utils.data import DUMMY_SNAPSHOT_NAME
from matbench_genmetrics.mp_time_split.utils.split import mp_time_splitter

dummy_data_path = path.join(get_data_home(), DUMMY_SNAPSHOT_NAME)

num_sites = (1, 2)
elements = ["V"]


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python 3.8+")
def test_data_snapshot_one_by_one():
    dummy_expt_df_check = load_dataframe_from_json(dummy_data_path)
    mpt = MPTimeSplit(num_sites=num_sites, elements=elements)
    dummy_expt_df = mpt.fetch_data(one_by_one=True)
    dummy_match = dummy_expt_df.compare(dummy_expt_df_check)
    if not dummy_match.empty:
        raise ValueError(
            f"dummy_expt_df and dummy_expt_df_check unmatched: {dummy_match}"
        )


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python 3.8+")
def test_get_train_and_val_data():
    mpt = MPTimeSplit(num_sites=num_sites, elements=elements)
    mpt.fetch_data(one_by_one=True)
    train_inputs = []
    val_inputs = []
    train_outputs = []
    val_outputs = []
    for fold in mpt.folds:
        train_input, val_input, train_output, val_output = mpt.get_train_and_val_data(
            fold
        )
        train_inputs.append(train_input)
        val_inputs.append(val_input)
        train_outputs.append(train_output)
        val_outputs.append(val_output)
    return train_inputs, val_inputs, train_outputs, val_outputs


@pytest.mark.skipif(sys.version_info < (3, 8), reason="requires Python 3.8+")
def test_get_test_data():
    mpt = MPTimeSplit(num_sites=num_sites, elements=elements)
    mpt.fetch_data(one_by_one=True)
    train_inputs, test_inputs, train_outputs, test_outputs = mpt.get_test_data()
    return train_inputs, test_inputs, train_outputs, test_outputs


def test_load():
    mpt = MPTimeSplit(num_sites=num_sites, elements=elements)
    data = mpt.load(dummy=True)
    return data


def test_mp_time_splitter():
    """ """
    mpt = MPTimeSplit(num_sites=num_sites, elements=elements)
    data = mpt.load(dummy=True)
    trainval_splits, test_split = mp_time_splitter(data, use_trainval_test=True)
    trainval_splits = mp_time_splitter(data, use_trainval_test=False)
    trainval_splits
    test_split


if __name__ == "__main__":
    # test_data_snapshot()
    test_data_snapshot_one_by_one()
    data = test_load()
    train_inputs, val_inputs, train_outputs, val_outputs = test_get_train_and_val_data()
    train_inputs, test_inputs, train_outputs, test_outputs = test_get_test_data()
    data

# %% Code Graveyard
# def test_data_snapshot():
#     dummy_expt_df_check = load_dataframe_from_json(dummy_data_path)
#     mpt = MPTimeSplit(num_sites=(1, 2), elements=["V"])
#     dummy_expt_df = mpt.fetch_data()
#     dummy_match = dummy_expt_df.compare(dummy_expt_df_check)
#     if not dummy_match.empty:
#         raise ValueError(
#             f"dummy_expt_df and dummy_expt_df_check unmatched: {dummy_match}"
#         )
