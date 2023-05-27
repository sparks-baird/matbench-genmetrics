from os import path
from pathlib import Path
from typing import List

from matbench_genmetrics.core.utils.featurize import featurize_comp_struct
from matbench_genmetrics.mp_time_split.splitter import MPTimeSplit

mpt = MPTimeSplit(target="energy_above_hull")

dummy = False
mpt.load(dummy=dummy)
material_ids: List[str] = mpt.data.material_id.tolist()
structures = mpt.data.structure

if __name__ == "__main__":
    comp_name = "composition"
    struct_name = "structure"
    material_id_name = "material_id"

    comp_fingerprints, struct_fingerprints = featurize_comp_struct(
        structures,
        material_ids=material_ids,
        comp_name=comp_name,
        struct_name=struct_name,
        material_id_name=material_id_name,
        keep_as_df=True,
    )
    data_dir = path.join("data", "interim")
    if dummy:
        data_dir = path.join(data_dir, "dummy")
    Path(data_dir).mkdir(exist_ok=True, parents=True)
    comp_fingerprints.to_csv(path.join(data_dir, "comp_fingerprints.csv"))
    struct_fingerprints.to_csv(path.join(data_dir, "struct_fingerprints.csv"))
