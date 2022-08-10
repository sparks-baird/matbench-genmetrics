from os import path
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from element_coder import encode
from mp_time_split.core import MPTimeSplit

mpt = MPTimeSplit(target="energy_above_hull")

dummy = True
mpt.load(dummy=dummy)
material_ids: List[str] = mpt.data.material_id.tolist()
structures = mpt.data.structure
material_id_name = "material_id"

if __name__ == "__main__":
    space_group_numbers = structures.apply(lambda s: s.get_space_group_info()[1])
    compositions = structures.apply(lambda s: s.composition.fractional_composition)

    space_group_numbers.name = "space_group_number"
    space_group_numbers.index = pd.Index(material_ids)
    space_group_numbers.index.name = material_id_name

    # NOTE: be aware of amount_tolerance=1e-8
    summed_comp = np.sum(compositions).fractional_composition
    _data = summed_comp._data
    mod_petti = [encode(k, "mod_pettifor") for k in _data.keys()]
    mod_petti_comp = dict(zip(mod_petti, _data.values()))

    mod_petti_df = pd.DataFrame(
        dict(
            symbol=list(_data.keys()),
            mod_petti=mod_petti_comp.keys(),
            contribution=mod_petti_comp.values(),
        )
    ).sort_values("mod_petti")

    data_dir = path.join("data", "interim")
    if dummy:
        data_dir = path.join(data_dir, "dummy")
    Path(data_dir).mkdir(exist_ok=True, parents=True)

    space_group_numbers.to_csv(path.join(data_dir, "space_group_numbers.csv"))
    mod_petti_df.to_csv(path.join(data_dir, "mod_petti_contributions.csv"), index=False)


# %% Code Graveyard
# space_group_number = [s.get_space_group_info()[1] for s in structures]
