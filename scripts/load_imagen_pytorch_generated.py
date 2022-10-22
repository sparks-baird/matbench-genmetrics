def main():
    import pickle
    from os import path

    from xtal2png import XtalConverter

    from matbench_genmetrics.core import MPTSMetrics10, MPTSMetrics1000

    fold = 0
    dummy = False
    checkpoint_epoch = 1999

    data_dir = path.join("data", "processed")
    fpath = path.join(data_dir, f"gen_images_epoch={checkpoint_epoch}.pkl")

    with open(fpath, "rb") as f:
        gen_images = pickle.load(f)

    xc = XtalConverter(encode_cell_type=None, decode_cell_type=None, verbose=False)

    if dummy:
        gen_images = gen_images[0:10]
        mptm = MPTSMetrics10(dummy=dummy, verbose=False)
    else:
        mptm = MPTSMetrics1000(dummy=dummy, verbose=False)

    gen_structures = xc.png2xtal(gen_images)

    mptm.get_train_and_val_data(fold)
    mptm.evaluate_and_record(fold, gen_structures)

    print(mptm.recorded_metrics)

    return mptm


if __name__ == "__main__":
    mptm = main()


# %% Code Graveyard
# fpath = path.join(data_dir, "gen_structures_epoch=700.pkl")
# with open(fpath, "rb") as f:
#     gen_structures = pickle.load(f)
