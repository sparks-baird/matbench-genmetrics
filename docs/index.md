# matbench-genmetrics

[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![ReadTheDocs](https://readthedocs.org/projects/matbench-genmetrics/badge/?version=latest)](https://matbench-genmetrics.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/sparks-baird/matbench-genmetrics/main.svg)](https://coveralls.io/r/sparks-baird/matbench-genmetrics)
[![PyPI-Server](https://img.shields.io/pypi/v/matbench-genmetrics.svg)](https://pypi.org/project/matbench-genmetrics/)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/matbench-genmetrics.svg)](https://anaconda.org/conda-forge/matbench-genmetrics)
![PyPI - Downloads](https://img.shields.io/pypi/dm/matbench-genmetrics)

Generative materials benchmarking metrics, inspired by [guacamol](https://www.benevolent.com/guacamol) and [CDVAE](https://github.com/txie-93/cdvae).

<a class="github-button" href="https://github.com/sparks-baird/matbench-genmetrics"
data-icon="octicon-star" data-size="large" data-show-count="true" aria-label="Star
sparks-baird/matbench-genmetrics on GitHub">Star</a>
<a class="github-button"
href="https://github.com/sgbaird" data-size="large" data-show-count="true"
aria-label="Follow @sgbaird on GitHub">Follow @sgbaird</a>
<a class="github-button" href="https://github.com/sparks-baird/matbench-genmetrics/issues"
data-icon="octicon-issue-opened" data-size="large" data-show-count="true"
aria-label="Issue sparks-baird/matbench-genmetrics on GitHub">Issue</a>
<a class="github-button" href="https://github.com/sparks-baird/matbench-genmetrics/discussions" data-icon="octicon-comment-discussion" data-size="large" aria-label="Discuss sparks-baird/matbench-genmetrics on GitHub">Discuss</a>
<br><br>

Many generative models for crystal structure have been developed; however, few
standardized benchmarks exist, and none exist in a format as easy-to-use as Matbench.
Here, we introduce  matbench-genmetrics, an open-source Python library for benchmarking
generative models for crystal structures. We incorporate benchmark datasets, splits, and
metrics inspired by [Crystal Diffusion Variational AutoEncoder (CDVAE)](https://github.com/txie-93/cdvae), which has been
demonstrated as state-of-the-art in generative crystal structure tasks. We
provide our own benchmarks using time-series style cross-validation splits from
Materials Project via our [mp-time-split package](https://mp-time-split.readthedocs.io/en/latest/) and focus on four metrics: validity,
coverage, novelty, and uniqueness. Finally, we provide a simple example of preparation
and submission to the  matbench-genmetrics leaderboard using [pyxtal](https://pyxtal.readthedocs.io/en/latest/) for generation.

## Contents

```{toctree}
:maxdepth: 2

Overview <readme>
Contributions & Help <contributing>
License <license>
Authors <authors>
Changelog <changelog>
Module Reference <api/modules>
GitHub Source <https://github.com/sparks-baird/matbench-genmetrics>
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`

[Sphinx]: http://www.sphinx-doc.org/
[Markdown]: https://daringfireball.net/projects/markdown/
[reStructuredText]: http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html
[MyST]: https://myst-parser.readthedocs.io/en/latest/

<script async defer src="https://buttons.github.io/buttons.js"></script>
