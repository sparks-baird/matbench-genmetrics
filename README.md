[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![ReadTheDocs](https://readthedocs.org/projects/matbench-genmetrics/badge/?version=latest)](https://matbench-genmetrics.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/matbench-genmetrics/main.svg)](https://coveralls.io/r/<USER>/matbench-genmetrics)
[![PyPI-Server](https://img.shields.io/pypi/v/matbench-genmetrics.svg)](https://pypi.org/project/matbench-genmetrics/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/matbench-genmetrics.svg)](https://anaconda.org/conda-forge/matbench-genmetrics)
![PyPI - Downloads](https://img.shields.io/pypi/dm/matbench-genmetrics)
<!-- These are examples of badges you might also want to add to your README. Update the URLs accordingly.
[![Built Status](https://api.cirrus-ci.com/github/<USER>/matbench-genmetrics.svg?branch=main)](https://cirrus-ci.com/github/<USER>/matbench-genmetrics)
[![Monthly Downloads](https://pepy.tech/badge/matbench-genmetrics/month)](https://pepy.tech/project/matbench-genmetrics)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/matbench-genmetrics)
-->
> **NOTE: This is a WIP repository (as of 2022-06-21) being developed in parallel with [`xtal2png`](https://github.com/sparks-baird/xtal2png). Feedback and contributions welcome!**
# matbench-genmetrics

> Generative materials benchmarking metrics, inspired by CDVAE.

This repository provides standardized benchmarks for benchmarking generative models for
crystal structure. Each benchmark has a fixed dataset, a predefined split, and a notion
of best (i.e. metric) associated with it.

## Getting Started

### Installation

Create a conda environment with the `matbench-genmetrics` package installed from the
`conda-forge` channel. Then activate the environment.

```bash
conda create --name matbench-genmetrics --channel conda-forge python==3.9.* matbench-genmetrics
conda activate matbench-genmetrics
```

> NOTE: It doesn't have to be Python 3.9; you can remove `python==3.9.*` altogether or
change this to e.g. `python==3.8.*`. See [Advanced Installation](##Advanced-Installation)

### Basic Usage

```python
from mp_time_split.utils.gen import DummyGenerator
from matbench_genmetrics.core import MPTSMetrics

mptm = MPTSMetrics(dummy=False)
for fold in mptm.folds:
    train_val_inputs = mptm.get_train_and_val_data(fold)

    dg = DummyGenerator()
    dg.fit(train_val_inputs)
    gen_structures = dg.gen(n=10000)

    mptm.record(fold, gen_structures)

print(mptm.recorded_metrics)
```

> ```python
>
> ```

## Advanced Installation

In order to set up the necessary environment:

1. review and uncomment what you need in `environment.yml` and create an environment `matbench-genmetrics` with the help of [conda]:
   ```
   conda env create -f environment.yml
   ```
2. activate the new environment with:
   ```
   conda activate matbench-genmetrics
   ```

> **_NOTE:_**  The conda environment will have matbench-genmetrics installed in editable mode.
> Some changes, e.g. in `setup.cfg`, might require you to run `pip install -e .` again.


Optional and needed only once after `git clone`:

3. install several [pre-commit] git hooks with:
   ```bash
   pre-commit install
   # You might also want to run `pre-commit autoupdate`
   ```
   and checkout the configuration under `.pre-commit-config.yaml`.
   The `-n, --no-verify` flag of `git commit` can be used to deactivate pre-commit hooks temporarily.

4. install [nbstripout] git hooks to remove the output cells of committed notebooks with:
   ```bash
   nbstripout --install --attributes notebooks/.gitattributes
   ```
   This is useful to avoid large diffs due to plots in your notebooks.
   A simple `nbstripout --uninstall` will revert these changes.


Then take a look into the `scripts` and `notebooks` folders.

## Dependency Management & Reproducibility

1. Always keep your abstract (unpinned) dependencies updated in `environment.yml` and eventually
   in `setup.cfg` if you want to ship and install your package via `pip` later on.
2. Create concrete dependencies as `environment.lock.yml` for the exact reproduction of your
   environment with:
   ```bash
   conda env export -n matbench-genmetrics -f environment.lock.yml
   ```
   For multi-OS development, consider using `--no-builds` during the export.
3. Update your current environment with respect to a new `environment.lock.yml` using:
   ```bash
   conda env update -f environment.lock.yml --prune
   ```
## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers.
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes.
├── CONTRIBUTING.md         <- Guidelines for contributing to this project.
├── Dockerfile              <- Build a docker container with `docker build .`.
├── LICENSE.txt             <- License as chosen on the command-line.
├── README.md               <- The top-level README for developers.
├── configs                 <- Directory for configurations of model & application.
├── data
│   ├── external            <- Data from third party sources.
│   ├── interim             <- Intermediate data that has been transformed.
│   ├── processed           <- The final, canonical data sets for modeling.
│   └── raw                 <- The original, immutable data dump.
├── docs                    <- Directory for Sphinx documentation in rst or md.
├── environment.yml         <- The conda environment file for reproducibility.
├── models                  <- Trained and serialized models, model predictions,
│                              or model summaries.
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for
│                              ordering), the creator's initials and a description,
│                              e.g. `1.0-fw-initial-data-exploration`.
├── pyproject.toml          <- Build configuration. Don't change! Use `pip install -e .`
│                              to install for development or to build `tox -e build`.
├── references              <- Data dictionaries, manuals, and all other materials.
├── reports                 <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures             <- Generated plots and figures for reports.
├── scripts                 <- Analysis and production scripts which import the
│                              actual PYTHON_PKG, e.g. train_model.
├── setup.cfg               <- Declarative configuration of your project.
├── setup.py                <- [DEPRECATED] Use `python setup.py develop` to install for
│                              development or `python setup.py bdist_wheel` to build.
├── src
│   └── matbench_genmetrics <- Actual Python package where the main functionality goes.
├── tests                   <- Unit tests which can be run with `pytest`.
├── .coveragerc             <- Configuration for coverage reports of unit tests.
├── .isort.cfg              <- Configuration for git hook that sorts imports.
└── .pre-commit-config.yaml <- Configuration of pre-commit git hooks.
```

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.2.2.post1.dev2+ge50b5e1 and the [dsproject extension] 0.7.2.post1.dev2+geb5d6b6.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
