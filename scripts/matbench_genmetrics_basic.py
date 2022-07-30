from tqdm import tqdm

from matbench_genmetrics.core import MPTSMetrics

mptm = MPTSMetrics(dummy=False, verbose=False)
for fold in tqdm(mptm.folds):
    train_val_inputs = mptm.get_train_and_val_data(fold)

    # dg = DummyGenerator()
    # dg.fit(train_val_inputs)
    # gen_structures = dg.gen(n=50)

    gen_structures = train_val_inputs.head(10)

    mptm.evaluate_and_record(fold, gen_structures)

print(mptm.recorded_metrics)
