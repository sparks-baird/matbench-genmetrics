# Metrics

The four metrics are validity, coverage, novelty, and uniqueness. We describe each in detail below.

## Validity

We define validity as one minus the Wasserstein distance between distribution of space group numbers for train and generated structures divided by the distance of the dummy case between train and the space group number 1:

$$
    1-\frac{w\left(\mathrm{SG}_{\mathrm{train}},\mathrm{SG}_{\mathrm{test}}\right)}{w\left(\mathrm{SG}_{\mathrm{train}},1\right)}
$$
where $w$, $\mathrm{SG}_{\mathrm{train}}$, and $\mathrm{SG}_{\mathrm{test}}$ represent Wasserstein distance, vector of space group numbers for the training data, and vector of space group numbers for the test data, respectively.

## Coverage

Coverage ("predict the future") is given by the match counts between the held-out test structures and the generated structures divided by the number of test structures:

$$
    \frac{\sum _{i=1}^{n_{\text{test}}} \sum _{j=1}^{n_{\text{gen}}} \left(
        \left\{
        \begin{array}{cc}
            1 & d\left(s_{\text{test},i},s_{\text{gen},j}\right)\leq \text{tol} \\
            0 & d\left(s_{\text{test},i},s_{\text{gen},j}\right)>\text{tol} \\
        \end{array}
        \\
        \right.
        \right)}{n_{\text{test}}}
$$
where $n_{\text{test}}$, $n_{\text{gen}}$, $d$, $s_{\text{test},i}$, $s_{\text{gen},j}$, and $\text{tol}$ represent number of structures in the test set, number of structures in the generated set, crystallographic distance according to \texttt{StructureMatcher} from \texttt{pymatgen.analysis.structure\_matcher}, $i$-th structure of the test set, $j$-th structure of the generated set, and a tolerance threshold, respectively.

## Novelty

Novelty is given by one minus the match counts between train structures and generated structures divided by number of generated structures:

$$
    1-\frac{\sum _{i=1}^{n_{\text{train}}} \sum _{j=1}^{n_{\text{gen}}} \left(
        \left\{
        \begin{array}{cc}
            1 & d\left(s_{\text{train},i},s_{\text{gen},j}\right)\leq \text{tol} \\
            0 & d\left(s_{\text{train},i},s_{\text{gen},j}\right)>\text{tol} \\
        \end{array}
        \\
        \right.
        \right)}{n_{\text{gen}}}
$$
where $n_{\text{train}}$, $n_{\text{gen}}$, $d$, $s_{\text{train},i}$, $s_{\text{gen},j}$, and $\text{tol}$ represent number of structures in the training set, number of structures in the generated set, crystallographic distance according to \texttt{StructureMatcher} from \texttt{pymatgen.analysis.structure\_matcher}, $i$-th structure of the training set, $j$-th structure of the generated set, and a tolerance threshold, respectively.

## Uniqueness

Uniqueness is given by one minus the non-self-comparing match counts within generated structures divided by total possible number of non-self-comparing matches:

$$
    1-\frac{\sum _{i=1}^{n_{\text{gen}}} \sum _{j=1}^{n_{\text{gen}}} \left(
        \left\{
        \begin{array}{cc}
            0 & i=j \\
            1 & d\left(s_{\text{gen},i},s_{\text{gen},j}\right)\leq \text{tol}\land i\neq j \\
            0 & d\left(s_{\text{gen},i},s_{\text{gen},j}\right)>\text{tol}\land i\neq j \\
        \end{array}
        \\
        \right.
        \right)}{n_{\text{gen}}^2-n_{\text{gen}}}
$$
where $n_{\text{gen}}$, $d$, $s_{\text{gen},i}$, $s_{\text{gen},j}$, and $\text{tol}$
represent number of structures in the generated set, crystallographic distance according
to `StructureMatcher` from `pymatgen.analysis.structure_matcher`, $i$-th
structure of the generated set, $j$-th structure of the generated set, and a tolerance
threshold, respectively.
