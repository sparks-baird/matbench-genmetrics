from typing import List, Optional, Tuple, Union

import pandas as pd

try:
    from mp_api.client import MPRester
except Exception as e:
    print(e)
    print("Falling back to from mp_api import MPRester")
    from mp_api import MPRester


from mp_api.client.core.client import DEFAULT_API_KEY
from tqdm import tqdm
from typing_extensions import Literal

from matbench_genmetrics.mp_time_split.utils.data import (
    get_discovery_dict,
    noble,
    radioactive,
)

# ensure match between following and `Literal` type hint for `exclude_elements`
AVAILABLE_EXCLUDE_STRS = ["noble", "radioactive", "noble+radioactive"]


def fetch_data(
    api_key: Union[str, DEFAULT_API_KEY] = DEFAULT_API_KEY,
    fields: Optional[List[str]] = [
        "structure",
        "material_id",
        "theoretical",
        "energy_above_hull",
        "formation_energy_per_atom",
    ],
    num_sites: Optional[Tuple[int, int]] = None,
    elements: Optional[List[str]] = None,
    exclude_elements: Optional[
        Union[List[str], Literal["noble", "radioactive", "noble+radioactive"]]
    ] = None,
    use_theoretical: bool = False,
    return_both_if_experimental: bool = False,
    one_by_one: bool = False,
    **search_kwargs,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Retrieve MP data sorted by MPID (theoretical+exptl) or pub year (exptl).

    See `*How do I do a time-split of Materials Project entries? e.g. pre-2018 vs.
    post-2018* <https://matsci.org/t/42584>`_

    Output ``DataFrame``-s will contain all specified `fields` unless ``fields is
    None``, in which case all :func:`MPRester().summary.available_fields` will be
    returned. If return experimental data, the additional fields of ``provenance``,
    ``discovery`` and ``year`` corresponding to
    :func:`emmet.core.provenance.ProvenanceDoc`, a dictionary containing earliest year
    and author information, and the earliest year, respectively, will also be returned.

    Parameters
    ----------
    api_key : Union[str, DEFAULT_API_KEY]
        :func:`mp_api` API Key. On Windows, can set as an environment variable via:
        ``setx MP_API_KEY="abc123def456"``. By default:
        :func:`mp_api.core.client.DEFAULT_API_KEY`
        See also:
        https://github.com/materialsproject/api/issues/566#issuecomment-1087941474
    fields : Optional[List[str]]
        fields (List[str]): List of fields to project. When searching, it is better to
        only ask for the specific fields of interest to reduce the time taken to
        retrieve the documents. See the :func:`MPRester().summary.available_fields`
        property to see a list of fields to choose from. By default:
        ``["structure", "material_id", "theoretical"]``.
    num_sites : Tuple[int, int]
        Tuple of min and max number of sites used as filtering criteria, e.g. ``(1,
        52)`` meaning at least ``1`` and no more than ``52`` sites. If ``None`` then no
        compounds with any number of sites are allowed. By default None.
    elements : List[str]
        List of element symbols, e.g. ``["Ni", "Fe"]``. If ``None`` then all elements
        are allowed. By default None.
    exclude_elements : Optional[
                            Union[List[str], Literal["noble", "radioactive",
                            "noble+radioactive"]]
                        ]
        List of element symbols to _exclude_, e.g. ``["Ar", "Ne"]``. If ``None`` then
        all elements are allowed. If a supported string value ("noble", "radioactive",
        or "noble+radioactive"), then filters out the appropriate elements. By default
        None.
    use_theoretical : bool, optional
        Whether to include both theoretical and experimental compounds or to filter down
        to only experimentally-verified compounds, by default False
    return_both_if_experimental : bool, optional
        Whether to return both the full DataFrame containing theoretical+experimental
        (`df`) and the experimental-only DataFrame (`expt_df`) or only `expt_df`, by
        default False. This is only applicable if `use_theoretical` is False.
    one_by_one: bool, optional
        Whether to retrieve data one-by-one instead of in bulk. This is useful for
        testing with a small number or in case the mp-api search is malfunctioning
        (since need provenance attributes). By default False.
    search_kwargs : dict, optional
        kwargs: Supported search terms, e.g. nelements_max=3 for the "materials" search
        API. Consult the specific API route for valid search terms,
        i.e. :func:`MPRester().summary.available_fields`

    Returns
    -------
    df : pd.DataFrame
        if `use_theoretical` then returns a DataFrame containing both theoretical and
        experimental compounds.
    expt_df, df : Tuple[pd.DataFrame, pd.DataFrame]
        if not `use_theoretical` and `return_both_if_experimental, then returns two
        :func:`pd.DataFrame`-s containing theoretical+experimental and
        experimental-only.
    expt_df : pd.DataFrame
        if not `use_theoretical` and not `return_both_if_experimental`, then returns a
        :func:`pd.DataFrame` containing the experimental-only compounds.

    Examples
    --------
    >>> api_key = "abc123def456"
    >>> num_sites = (1, 52)
    >>> elements = ["V"]
    >>> expt_df = fetch_data(api_key, num_sites=num_sites, elements=elements)

    >>> df = fetch_data(
            api_key,
            num_sites=num_sites,
            elements=elements,
            use_theoretical=True
        )

    >>> expt_df, df = fetch_data(
            api_key,
            num_sites=num_sites,
            elements=elements,
            use_theoretical=False,
            return_both_if_experimental
        )
    """
    if fields is not None:
        if "material_id" not in fields:
            fields.append("material_id")
        if not use_theoretical and "theoretical" not in fields:
            fields.append("theoretical")

    if exclude_elements is None:
        excl_elems = None
    elif isinstance(exclude_elements, str):
        if exclude_elements not in AVAILABLE_EXCLUDE_STRS:
            raise NotImplementedError(
                f"Because str passed to `exclude_elements` instead of list of str, expected one of {AVAILABLE_EXCLUDE_STRS}"  # noqa: E501
            )
        if exclude_elements == "noble":
            excl_elems = noble
        elif exclude_elements == "radioactive":
            excl_elems = radioactive
        elif exclude_elements == "noble+radioactive":
            excl_elems = noble + radioactive
    else:
        excl_elems = exclude_elements

    with MPRester(api_key) as mpr:
        results = mpr.summary.search(
            num_sites=num_sites,
            elements=elements,
            exclude_elements=excl_elems,
            fields=fields,
            **search_kwargs,
        )

        if fields is not None:
            field_data = []
            for r in results:
                field_data.append({field: getattr(r, field) for field in fields})
        else:
            field_data = results

        material_id = [str(fd["material_id"]) for fd in field_data]

        # mvc values get distinguished by a negative sign
        index = [
            int(mid.replace("mp-", "").replace("mvc-", "-")) for mid in material_id
        ]
        df = pd.DataFrame(field_data, index=index)
        df = df.sort_index()

        if not use_theoretical:
            # REVIEW: whether to use MPID class or str of MPIDs?
            # if latter, `expt_df.material_id.apply(str).tolist()`
            expt_df = df.query("theoretical == False")
            expt_material_id = expt_df.material_id.tolist()

            if not one_by_one:
                # https://github.com/materialsproject/api/issues/613
                provenance_results = mpr.provenance.search(
                    fields=["references", "material_id"]
                )
                provenance_ids = [fpr.material_id for fpr in provenance_results]
                prov_df = pd.Series(
                    name="provenance", data=provenance_results, index=provenance_ids
                )
                expt_provenance_results = prov_df.loc[expt_material_id]
            else:
                # slow version
                expt_provenance_results = [
                    mpr.provenance.get_data_by_id(mid) for mid in tqdm(expt_material_id)
                ]
            # CrystalSystem not JSON serializable, see
            # https://github.com/materialsproject/api/issues/615
            # expt_df["provenance"] = expt_provenance_results

            # extract earliest ICSD year
            references = [pr.references for pr in expt_provenance_results]
            discovery = get_discovery_dict(references)
            year = [disc["year"] for disc in discovery]
            # https://stackoverflow.com/a/35387129/13697228
            expt_df = expt_df.assign(
                references=references, discovery=discovery, year=year
            )

            expt_df = expt_df.sort_values(by=["year"])

    if use_theoretical:
        return df
    elif return_both_if_experimental:
        return expt_df, df
    else:
        return expt_df
