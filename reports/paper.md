---
title: 'matbench-genmetrics: A Python library for benchmarking crystal structure generative models using time-based splits of Materials Project structures'
tags:
  - Python
  - materials informatics
  - crystal structure
  - generative modeling
  - TimeSeriesSplit
  - benchmarking
authors:
  - name: Sterling G. Baird
    orcid: 0000-0002-4491-6876
    equal-contrib: false
    corresponding: true
    affiliation: "1,3" # (Multiple affiliations must be quoted)
  - name: Hasan M. Sayeed
    orcid: 0000-0002-6583-7755
    equal-contrib: false
    corresponding: false
    affiliation: "1"
  - name: Joseph Montoya
    orcid: 0000-0001-5760-2860
    affiliation: "2"
    # Kevin Jablonka? element-coder was a great contribution here, though it exists in another repository
  - name: Taylor D. Sparks
    orcid: 0000-0001-8020-7711
    equal-contrib: false
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations:
 - name: Materials Science & Engineering, University of Utah, USA
   index: 1
 - name: Toyota Research Institute, Los Altos, CA, USA
   index: 2
 - name: Acceleration Consortium, University of Toronto. 80 St George St, Toronto, ON M5S 3H6
   index: 3
date: 19 March 2024
bibliography: paper.bib

# # Optional fields if submitting to a AAS journal too, see this blog post:
# # https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The progress of a machine learning field is both tracked and propelled through the development of robust benchmarks. While significant progress has been made to create standardized, easy-to-use benchmarks for molecular discovery (e.g., [@brown_guacamol_2019]), this remains a challenge for solid-state material discovery [@xie_crystal_2022; @zhao_physics_2022; @alverson_generative_2022]. To address this limitation, we propose `matbench-genmetrics`, an open-source Python library for benchmarking generative models for crystal structures. We use four evaluation metrics inspired by Guacamol [@brown_guacamol_2019] and Crystal Diffusion Variational AutoEncoder (CDVAE) [@xie_crystal_2022]---validity, coverage, novelty, and uniqueness---to assess performance on Materials Project data splits using timeline-based cross-validation. We believe that `matbench-genmetrics` will provide the standardization and convenience required for rigorous benchmarking of crystal structure generative models. A visual overview of the `matbench-genmetrics` library is provided in \autoref{fig:summary}.

![Summary visualization of `matbench-genmetrics` to evaluate crystal generative model performance using validity, coverage, novelty, and uniqueness metrics based on calendar-time splits of experimentally determined Materials Project database entries. Validity is the comparison of distribution characteristics (space group number) between the generated materials and the training and test sets. Coverage is the number of matches between the generated structures and a held-out test set. Novelty is a comparison between the generated and training structures. Finally, uniqueness is a measure of the number of repeats within the generated structures (i.e., comparing the set of generated structures to itself). For in-depth descriptions and equations for the four metrics described above, see [https://matbench-genmetrics.readthedocs.io/en/latest/readme.html](https://matbench-genmetrics.readthedocs.io/en/latest/readme.html) and [https://matbench-genmetrics.readthedocs.io/en/latest/metrics.html](https://matbench-genmetrics.readthedocs.io/en/latest/metrics.html).\label{fig:summary}](figures/matbench-genmetrics.png)

<!-- Maybe move the emojis beneath the name and horizontal line -->

# Statement of need

In the field of materials informatics, where materials science intersects with machine learning, benchmarks play a crucial role in assessing model performance and enabling fair comparisons among various tools and models. Typically, these benchmarks focus on evaluating the accuracy of predictive models for materials properties, utilizing well-established metrics such as mean absolute error (MAE) and root-mean-square error (RMSE) to measure performance against actual measurements. A standard practice involves splitting the data into two parts, with one serving as training data for model development and the other as test data for assessing performance [@dunn_benchmarking_2020].

However, benchmarking generative models, which aim to create entirely new data rather than focusing solely on predictive accuracy, presents unique challenges. While significant progress has been made in standardizing benchmarks for tasks like image generation and molecule synthesis, the field of crystal structure generative modeling lacks this level of standardization (this is separate from machine learning interatomic potentials, which have the robust and comprehensive [`matbench-discovery`](https://matbench-discovery.materialsproject.org/) [@riebesell_matbench_2024] and [Jarvis Leaderboard](https://pages.nist.gov/jarvis_leaderboard/) benchmarking frameworks [@choudhary_large_2023]). Molecular generative modeling benefits from widely adopted benchmark platforms such as Guacamol [@brown_guacamol_2019] and Moses [@polykovskiy_molecular_2020], which offer easy installation, usage guidelines, and leaderboards for tracking progress. In contrast, existing evaluations in crystal structure generative modeling, as seen in CDVAE [@xie_crystal_2022], FTCP [@ren_invertible_2022], PGCGM [@zhao_physics_2022], CubicGAN [@zhao_high-throughput_2021], and CrysTens [@alverson_generative_2022], lack standardization, pose challenges in terms of installation and application to new models and datasets, and lack publicly accessible leaderboards. While these evaluations are valuable within their respective scopes, there is a clear need for a dedicated benchmarking platform to promote standardization and facilitate robust comparisons.

In this work, we introduce `matbench-genmetrics`, a materials benchmarking platform for crystal structure generative models. We use concepts from molecular generative modeling benchmarking to create a set of evaluation metrics---validity, coverage, novelty, and uniqueness---which are broadly defined as follows:

- **Validity**: a measure of how well the generated materials match the distribution of the training dataset
- **Coverage**: the ability to successfully predict known materials which have been held out
- **Novelty**: generating structures which are close matches to examples in the training set are penalized
- **Uniqueness**: the number of repeats within the generated structures

`matbench-genmetrics` is comprised of two namespace packages. The first is `matbench_genmetrics.core`, which provides the following features:

- `GenMatcher`: A class for calculating matches between two sets of structures
- `GenMetrics`: A class for calculating validity, coverage, novelty, and uniqueness metrics
- `MPTSMetrics`: class for loading `mp_time_split` data, calculating time-series cross-validation metrics, and saving results
- Fixed benchmark classes for 10, 100, 1000, and 10000 generated structures

Additionally, we introduce the `matbench_genmetrics.mp_time_split` namespace package as a complement to `matbench_genmetrics.core`. It provides a standardized dataset and cross-validation splits for evaluating the mentioned four metrics. Time-based splits have been utilized in materials informatics model validation, such as predicting future thermoelectric materials via word embeddings [@tshitoyan_unsupervised_2019], searching for efficient solar photoabsorption materials through multi-fidelity optimization [@palizhati_agents_2022], and predicting future materials stability trends via network models [@aykol_network_2019]. Recently, Hu et al. [@zhao_physics_2022] used what they call a rediscovery metric, referred to here as a coverage metric in line with molecular benchmarking terminology, to evaluate crystal structure generative models. While time-series splitting wasn't used, they showed that after generating millions of structures, only a small percentage of held-out structures had matches. These results highlight the difficulty (and robustness) of coverage tasks. By leveraging timeline metadata from the Materials Project database [@jain_commentary_2013] and creating a standard time-series splitting of data, `matbench_genmetrics.mp_time_split` enables rigorous evaluation of future discovery performance.

The `matbench_genmetrics.mp_time_split` namespace package provides the following features:

- downloading and storing snapshots of Materials Project crystal structures via `pymatgen` [@ong_python_2013]
- modification of `pymatgen` search criteria to fetch custom datasets
- utilities for post-processing Materials Project entries
- convenience methods to access the snapshot dataset
- predefined scikit-learn `TimeSeriesSplit` cross-validation splits [@ong_python_2013]

In future work, metrics will serve as multi-criteria filters to prevent manipulation. Standalone metrics can be "hacked" by generating nonsensical structures for novelty or including training structures to inflate validity scores. To address this, multiple criteria are considered simultaneously for each generated structure, such as novelty, uniqueness, and filtering rules like non-overlapping atoms, stoichiometry, or checkCIF criteria [@spek_checkcif_2020]. Additional filters based on machine learning models can be applied for properties like negative formation energy, energy above hull, ICSD classification, and coordination number. Applying machine-learning-based structural relaxation using M3GNet [@chen_universal_2022] (e.g., as in CrysTens [@alverson_generative_2022]) before filtering is also of interest. Contributions related to multi-criteria filtering, enhanced validity filters, and implementing a benchmark submission system and public leaderboard are welcome.

We believe that the `matbench-genmetrics` ecosystem is a robust and easy-to-use benchmarking platform that will help propel novel materials discovery and targeted crystal structure inverse design. We hope that practioners of crystal structure generative modeling will adopt `matbench-genmetrics`, contribute improvements and ideas, and submit their results to the planned public leaderboard.

# Acknowledgements

We acknowledge contributions from Kevin M. Jablonka ([\@kjappelbaum](https://github.com/kjappelbaum)), Matthew K. Horton ([\@mkhorton](https://github.com/mkhorton)), Kyle D. Miller ([\@kyledmiller](https://github.com/kyledmiller)), and Janosh Riebesell ([\@janosh](https://github.com/janosh)). S.G.B. and T.D.S. acknowledge support by the National Science Foundation, USA under Grant No. DMR-1651668. We acknowledge OpenAI for the use of ChatGPT for basic proofreading and editing.

# References
