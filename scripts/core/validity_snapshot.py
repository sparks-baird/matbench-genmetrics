from os import path
from pathlib import Path
from typing import List

import pandas as pd

from matbench_genmetrics.core.utils.featurize import mod_petti_contributions
from matbench_genmetrics.mp_time_split.splitter import MPTimeSplit

mpt = MPTimeSplit(target="energy_above_hull")

dummy = True
mpt.load(dummy=dummy)
material_ids: List[str] = mpt.data.material_id.tolist()
structures = mpt.data.structure
material_id_name = "material_id"

if __name__ == "__main__":
    space_group_numbers = structures.apply(lambda s: s.get_space_group_info()[1])

    space_group_numbers.name = "space_group_number"
    space_group_numbers.index = pd.Index(material_ids)
    space_group_numbers.index.name = material_id_name

    mod_petti_df = mod_petti_contributions(structures)

    data_dir = path.join("data", "interim")
    if dummy:
        data_dir = path.join(data_dir, "dummy")
    Path(data_dir).mkdir(exist_ok=True, parents=True)

    space_group_numbers.to_csv(path.join(data_dir, "space_group_numbers.csv"))
    mod_petti_df.to_csv(path.join(data_dir, "mod_petti_contributions.csv"), index=False)


# %% Code Graveyard
# space_group_number = [s.get_space_group_info()[1] for s in structures]
