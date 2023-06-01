"""Touch up the conda recipe from grayskull using conda-souschef."""
import os
from copy import copy
from os import getcwd
from os.path import basename, dirname, join, normpath
from pathlib import Path
from shutil import copyfile
from warnings import warn

import mp_time_split as module
import numpy as np
from souschef.recipe import Recipe

# from packaging.version import VERSION_PATTERN


name, version = module.__name__, module.__version__

replace_underscores_with_hyphens = True

if replace_underscores_with_hyphens:
    name = name.replace("_", "-")

src_dirname = "src"
if basename(normpath(getcwd())) != src_dirname:
    warn(
        f"`meta.yaml` will be saved to {join(getcwd(), name)} instead of {join(src_dirname, name)}. If this is not the desired behavior, delete {join(getcwd(), name)}, `cd` to {src_dirname}, and rerun."  # noqa: E501
    )

# Regex to match PEP440 compliant version strings
# https://stackoverflow.com/a/38020327/13697228
# _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)

# if bool(_regex.match(version)):

version = os.popen("git describe --abbrev=0 --tags").read().replace("\n", "")

# warn("version is 'unknown', falling back to {version} via git tag")

os.system(f"grayskull pypi {name}=={version}")

# Whether to save meta.yaml and LICENSE.txt file to a "scratch" folder for `conda build`
personal_conda_channel = False

fpath = join(name, "meta.yaml")

if personal_conda_channel:
    fpath2 = join(name, "scratch", "meta.yaml")
    Path(dirname(fpath2)).mkdir(exist_ok=True)

my_recipe = Recipe(load_file=fpath)

# my_recipe.add_section("outputs")
# my_recipe["outputs"] = {"name": name}
# my_recipe["outputs"].add_section("name")({"name": name.replace("-", "_")})

# ensure proper order for conda-forge
keys = list(my_recipe.keys())

# https://docs.conda.io/projects/conda-build/en/latest/resources/define-metadata.html
key_order = [
    "package",
    "source",
    "build",
    "requirements",
    "test",
    "outputs",
    "about",
    "app",
    "extra",
]
unshared_keys = np.setdiff1d(key_order, keys)

ordered_keys = copy(key_order)
for key in unshared_keys:
    ordered_keys.remove(key)

for key in ordered_keys:
    my_recipe.yaml.move_to_end(key)

my_recipe["build"].add_section({"noarch": "python"})

try:
    del my_recipe["build"]["skip"]
except Exception as e:
    print(e)
    warn("Could not delete build: skip section (probably because it didn't exist)")

try:
    del my_recipe["requirements"]["build"]
except Exception as e:
    print(e)
    warn("Could not delete build section (probably because it didn't exist)")

min_py_ver = "3.6"
my_recipe["requirements"]["host"].remove("python")
my_recipe["requirements"]["host"].append(f"python >={min_py_ver}")

my_recipe["requirements"]["run"].remove("python")
my_recipe["requirements"]["run"].append(f"python >={min_py_ver}")

# remove the `# [py<38]` selector comment
run_section = my_recipe["requirements"]["run"]
idx = run_section.index("importlib-metadata")
my_recipe["requirements"]["run"].remove("importlib-metadata")
my_recipe["requirements"]["run"].append("importlib-metadata")


my_recipe["about"]["doc_url"] = "mp-time-split.readthedocs.io"

# # sometimes package names differ between PyPI and Anaconda but haven't been registered
# # yet on grayskull (e.g. `kaleido`)
# my_recipe["requirements"]["run"].replace("kaleido", "python-kaleido")

# # It's better to install some packages either exclusively via Anaconda or
# # via custom PyPI installation instructions (see e.g. the selectable table from:
# # https://pytorch.org/get-started/locally/)
# my_recipe["requirements"]["run"].append("pytorch >=1.9.0")
# my_recipe["requirements"]["run"].append("cudatoolkit <11.4")

my_recipe.save(fpath)

if personal_conda_channel:
    my_recipe.save(fpath2)
    copyfile("LICENSE.txt", join(dirname(fpath2), "LICENSE.txt"))
