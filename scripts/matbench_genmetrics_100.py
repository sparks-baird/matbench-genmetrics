from mp_time_split.utils.gen import DummyGenerator
from tqdm import tqdm

from matbench_genmetrics.core import MPTSMetrics100

mptm = MPTSMetrics100(dummy=False, verbose=True)
for fold in tqdm(mptm.folds):
    train_val_inputs = mptm.get_train_and_val_data(fold)

    dg = DummyGenerator()
    dg.fit(train_val_inputs)
    gen_structures = dg.gen(n=mptm.num_gen)

    mptm.evaluate_and_record(fold, gen_structures)

print(mptm.recorded_metrics)
1 + 1
