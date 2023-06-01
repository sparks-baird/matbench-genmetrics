import re
from typing import List

import pybtex.errors
from pybtex.database.input import bibtex
from tqdm import tqdm

pybtex.errors.set_strict_mode(False)

SNAPSHOT_NAME = "mp_time_summary.json"
DUMMY_SNAPSHOT_NAME = "mp_dummy_time_summary.json"

noble = ["He", "Ar", "Ne", "Kr", "Xe", "Og", "Rn"]
# fmt: off
radioactive = ["U", "Th", "Ra", "Pu", "Po", "Rn", "Cm", "At", "Bk", "Fr", "Ac", "Am", "Bh", "Cf", "Np", "Ts", "Tc", "Md", "Lr", "Fm", "Hs", "Mt", "No", "Pm", "Rf", "Sg", "Ds", "Cn", "Rg", "Lv", "Og", "Fl", "Nh", "Db", "Es", "Mc", "Pa", "Bi", "Cs"]  # noqa: E501
# fmt: on


def get_discovery_dict(references: List[dict]) -> List[dict]:
    """Get a dictionary containing earliest bib info for each MP entry.

    Modified from source:
    "How do I do a time-split of Materials Project entries? e.g. pre-2018 vs. post-2018"
    https://matsci.org/t/42584/4?u=sgbaird, answer by @Joseph_Montoya, Materials Project Alumni

    Parameters
    ----------
    provenance_results : List[dict]
        List of references results, e.g. taken from from the ``ProvenanceRester`` API
        results (:func:`mp_api.provenance`)

    Returns
    -------
    discovery, List[dict]
        Dictionary containing earliest bib info for each MP entry with keys: ``["year",
        "authors", "num_authors"]``

    Examples
    --------
    >>> with MPRester(api_key) as mpr:
    ...     provenance_results = mpr.provenance.search(num_sites=(1, 4), elements=["V"])
    >>> discovery = get_discovery_dict(provenance_results)
    [{'year': 1963, 'authors': ['Raub, E.', 'Fritzsche, W.'], 'num_authors': 2}, {'year': 1925, 'authors': ['Becker, K.', 'Ebert, F.'], 'num_authors': 2}, {'year': 1965, 'authors': ['Giessen, B.C.', 'Grant, N.J.'], 'num_authors': 2}, {'year': 1957, 'authors': ['Philip, T.V.', 'Beck, P.A.'], 'num_authors': 2}, {'year': 1963, 'authors': ['Darby, J.B.jr.'], 'num_authors': 1}, {'year': 1977, 'authors': ['Aksenova, T.V.', 'Kuprina, V.V.', 'Bernard, V.B.', 'Skolozdra, R.V.'], 'num_authors': 4}, {'year': 1964, 'authors': ['Maldonado, A.', 'Schubert, K.'], 'num_authors': 2}, {'year': 1962, 'authors': ['Darby, J.B.jr.', 'Lam, D.J.', 'Norton, L.J.', 'Downey, J.W.'], 'num_authors': 4}, {'year': 1925, 'authors': ['Becker, K.', 'Ebert, F.'], 'num_authors': 2}, {'year': 1959, 'authors': ['Dwight, A.E.'], 'num_authors': 1}] # noqa: E501
    """
    discovery = []
    for refs in tqdm(references):
        parser = bibtex.Parser()
        refs = "".join(refs)
        refs = parser.parse_string(refs)
        entries = refs.entries
        entries_by_year = [
            (int(entry.fields["year"]), entry)
            for _, entry in entries.items()
            if "year" in entry.fields and re.match(r"\d{4}", entry.fields["year"])
        ]
        if entries_by_year:
            entries_by_year = sorted(entries_by_year, key=lambda x: x[0])
            first_report = {
                "year": entries_by_year[0][0],
                "authors": entries_by_year[0][1].persons["author"],
            }
            first_report["authors"] = [str(auth) for auth in first_report["authors"]]
            first_report["num_authors"] = len(first_report["authors"])
            discovery.append(first_report)
        else:
            discovery.append(dict(year=None, authors=None, num_authors=None))
    return discovery


# def encode_dataframe(df):
#     jsonpickle_pandas.register_handlers()
#     return jsonpickle.encode(df)


# def decode_dataframe_from_string(string):
#     jsonpickle_pandas.register_handlers()
#     return jsonpickle.decode(string, classes=[Structure])


# %% Code graveyard
