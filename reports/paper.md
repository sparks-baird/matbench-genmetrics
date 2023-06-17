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
    affiliation: "1" # (Multiple affiliations must be quoted)
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
date: 17 June 2023
bibliography: paper.bib

# # Optional fields if submitting to a AAS journal too, see this blog post:
# # https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

The progress of a machine learning field is both tracked and propelled through the development of robust benchmarks. While significant progress has been made to create standardized, easy-to-use benchmarks for molecular discovery (e.g., [@brown_guacamol_2019]), this remains a challenge for solid-state material discovery [@xie_crystal_2022; @zhao_physics_2022; @alverson_generative_2022]. To address this limitation, we propose `matbench_genmetrics`, an open-source Python library for benchmarking generative models for crystal structures. We incorporate benchmark datasets, splits, and metrics. The four evaluation metrics we use---validity, coverage, novelty, and uniqueness---are inspired by Guacamol [@brown_guacamol_2019] and Crystal Diffusion Variational AutoEncoder (CDVAE) [@xie_crystal_2022]. The evaluation metrics and benchmark classes are implemented in the `matbench-genmetrics.core` namespace package. The datasets and splitting are handled via the `matbench_genmetrics.mp_time_split` namespace package using Materials Project [@jain_commentary_2013] crystal structures and time-series cross-validation splits [@pedregosa_scikit-learn_2011] based on date first reported in literature, respectively. Future work may involve an automated leaderboard, an automated submission system, and easy-to-use examples for preparation and submission. We believe that `matbench-genmetrics` will provide the standardization and convenience required for rigorous benchmarking of crystal structure generative models. A visual overview of the `matbench_genmetrics` library is provided in \autoref{fig:summary}.

![Summary visualization of `matbench_genmetrics` to evaluate crystal generative model performance using validity, coverage, novelty, and uniqueness metrics based on calendar-time splits of experimentally determined Materials Project database entries. Validity is the comparison of distribution characteristics (space group number) between the generated materials and the training and test sets. Coverage is the number of matches between the generated structures and a held-out test set. Novelty is a comparison between the generated and training structures. Finally, uniqueness is a measure of the number of repeats within the generated structures (i.e., comparing the set of generated structures to itself). For in-depth descriptions and equations for the four metrics described above, see https://matbench-genmetrics.readthedocs.io/en/latest/readme.html and https://github.com/sparks-baird/matbench-genmetrics/blob/main/reports/matbench-genmetrics.tex.\label{fig:summary}](figures/matbench-genmetrics.png)

<!-- Maybe move the emojis beneath the name and horizontal line -->

# Statement of need

In the field of materials informatics, benchmarks are often geared towards property prediction with standard metrics such as mean absolute error (MAE) and root mean squared error (RMSE) and random train-test splits [@dunn_benchmarking_2020]; however, generative modeling requires domain-specific evaluation metrics. While the field of molecular generative modeling is converging on a set of benchmark datasets and metrics, crystal structure generative modeling currently exhibits little standardization. For example, in the field of molecular generative modeling, there are two widely popular benchmark platforms --- Guacamol [@brown_guacamol_2019] and Moses [@polykovskiy_molecular_2020] --- each with a large following, easy installation and usage instructions, and some form of a leaderboard. By contrast, existing crystal structure generative modeling performance evaluations (e.g., CDVAE [@xie_crystal_2022], FTCP [@ren_invertible_2022], PGCGM [@zhao_physics_2022], CubicGAN [@zhao_high-throughput_2021], CrysTens [@alverson_generative_2022]) haven't converged on a common set of standards, are difficult to install and apply to new models and datasets, and lack public leaderboards. While these one-off assessments have been valuable in assessing a model's performance against a subset of related models, a benchmarking platform is needed to promote standardization and robust comparisons.

In this work, we introduce `matbench-genmetrics`, a materials benchmarking platform for crystal structure generative models. We use concepts from molecular generative modeling benchmarking to create a set of evaluation metrics --- validity, coverage, novelty, and uniqueness --- which are broadly defined as follows:

- Validity: a measure of how well the generated materials match the distribution of the training dataset
- Coverage: the ability to successfully predict known materials which have been held out
- Novelty: generating structures which are close matches to examples in the training set are penalized
- Uniqueness: the number of repeats within the generated structures

The `matbench_genmetrics.core` namespace package provides the following features:
- GenMatcher: A class for calculating matches between two sets of structures
- GenMetrics: A class for calculating validity, coverage, novelty, and uniqueness metrics
- MPTSMetrics: class for loading `mp_time_split` data, calculating cross-validation metrics, and saving results
- Fixed benchmark classes for 10, 100, 1000, and 10000 generated structures

As a complement to `matbench_genmetrics.core`, we introduce an additional namespace package, `matbench_genmetrics.mp_time_split`, which provides a standardized dataset and cross-validation splits to assess the four evaluation metrics mentioned above. Time-based splits have been used in the past for validating materials informatics models. For example, Jain et al. [@tshitoyan_unsupervised_2019] "tested whether [the] model -- if trained at various points in the past -- would have correctly predicted thermoelectric materials reported later in the literature." Likewise, Montoya et al. [@palizhati_agents_2022] "seeded [multi-fidelity agents with the] first 500 experimentally discovered compositions (based on ICSD58 timeline of their first publication) and their corresponding DFT data." Hummelshøj et al. [@aykol_network_2019] describe the difficulties associated with predicting future trends of materials discovery in the time-evolution of a materials stability network. We note that each of these examples used bespoke splitting of the data. Recently, Hu et al. [@zhao_physics_2022] used what they call a rediscovery metric (we refer to this as a coverage metric in line with molecular benchmarking terminology) to evaluate the results of their crystal structure generative model, though this was not using a time-based split. The need to generate millions of structures to replicate small portions of the heldout dataset highlights the difficulty of the task. When used with other benchmarking metrics, time-based coverage can provide the rigor required to effectively evaluate the performance of generative materials discovery models. In the Materials Project database [@jain_commentary_2013], there are records of when experimentally validated compounds were first reported in the literature. By using this metadata in the `matbench_genmetrics.mp_time_split` namespace package, we are able to ask: "how well can we predict what will be discovered in the future?" `matbench_genmetrics.mp_time_split` acts as a convenient, standardized backend for coverage benchmarking metrics.

The `matbench_genmetrics.mp_time_split` namespace package provides the following features:
- downloading and storing snapshots of Materials Project crystal structures via pymatgen [REF] (experimentally verified, theoretical, or both)
- modification of search criteria to fetch custom datasets
- utilities for post-processing the Materials Project entries
- convenient access to a snapshot dataset
- predefined scikit-learn TimeSeriesSplit cross-validation splits [REF]

In future work, the metrics can be used as multi-criteria filtering. This is important is because, standalone, the metrics can be "hacked". For example, the novelty metric may be made perfect simply by generating a diverse set of nonsensical crystal structures. Likewise, the validity score may be bloated simply by passing in the list of training structures as the generated structures. To combat this, multiple criteria may be considered simultaneously: for example, requiring that a structure must be simultaneously novel, unique, and passing certain filtering criteria such as non-overlapping atoms, stoichiometry rules, or `checkCIF` criteria [@spek_checkcif_2020]. Additional filters based on maching learning prediction models can be used for properties such as formation energy (i.e., must be negative), energy above hull, ICSD classification, and coordination number. Of particular interest is applying machine-learning based structural relaxation through universal interatomic potential models such as `M3GNet` [@chen_universal_2022] prior to filtering (e.g., `M3GNet` was used for performance evaluation in CrysTens [@alverson_generative_2022]). Additional future work involves the creation of a benchmark submission system and public leaderboard. Contributions related to multi-criteria filtering, enhanced validity filters, benchmark submission systems, and the public leaderboard are welcome.

We believe that the `matbench_genmetrics` ecosystem is a robust and easy-to-use benchmarking platform that will help propel novel materials discovery and targeted crystal structure inverse design. We hope that practioners of crystal structure generative modeling will adopt `matbench_genmetrics`, contribute improvements and ideas, and submit their results to the planned public leaderboard.

# Acknowledgements

We acknowledge contributions from Kevin M. Jablonka (<a href="https://github.com/kjappelbaum">@kjappelbaum</a>), Matthew K. Horton (<a href="https://github.com/mkhorton">@mkhorton</a>), and Kyle D. Miller (<a href="https://github.com/kyledmiller">@kyledmiller</a>). S.G.B. and T.D.S. acknowledge support by the National Science Foundation, USA under Grant No. DMR-1651668.

# References


<!-- Here, we highlight the coverage metric (or rediscovery metric) which involves
the ability to successfully predict known materials which have been held out. For
experimental materials discovery, a robust measure of performance is whether or not we
can predict materials of the future based
only on training data from the past. In other words: "how well can we predict what will
be discovered in the future?" In the Materials Project database [@jain_commentary_2013],
there are records of when experimentally validated compounds were first reported in the
literature. As a robust validation setup, we formalize the time-series splits of
Materials Project crystal structures for use in generative modeling benchmarking via the
`mp_time_split` Python namespace of the `matbench_genmetrics` ecosystem (see \autoref{fig:summary}). `mp_time_split` provides
convenience functions for downloading and processing snapshots of experimentally
verified Materials Project entries and creating random time-series splits of the data. -->


<!-- We believe `mp-time-split` provides the convenience and standardization required of
rigorous benchmarking of generative materials discovery models. `mp-time-split` serves
as the basis for a set of benchmarking metrics hosted in the [`matbench-genmetrics`](https://github.com/sparks-baird/matbench-genmetrics) suite
which has recently been applied to `xtal2png` [@baird_xtal2png_2022], a generative model
for crystal structure. -->


<!--- Mention similar options in molecular discovery benchmarking, e.g. guacamol which I believe has something similar in terms of rediscovery, though maybe not time-based. Mention legacy materials informatics (CrabNet, CGCNN, etc.) and the shift towards inverse design via generative modeling (CDVAE, FTCP, PGCGM, CubicGAN, etc.). --->
