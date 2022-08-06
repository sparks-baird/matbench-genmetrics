from os import path
from pathlib import Path

from matbench_genmetrics.utils.featurize import featurize_comp_struct

if __name__ == "__main__":
    comp_name = "composition"
    struct_name = "structure"
    material_id_name = "material_id"

    dummy = False
    comp_fingerprints, struct_fingerprints = featurize_comp_struct(
        dummy=dummy,
        comp_name=comp_name,
        struct_name=struct_name,
        material_id_name=material_id_name,
    )
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
