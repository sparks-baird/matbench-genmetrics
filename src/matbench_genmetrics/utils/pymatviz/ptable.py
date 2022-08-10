"""https://github.com/janosh/pymatviz/blob/main/pymatviz/ptable.py"""
from __future__ import annotations

from os.path import abspath, dirname
from typing import TYPE_CHECKING, Literal, Sequence

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from pymatgen.core import Composition

ROOT = dirname(dirname(abspath(__file__)))

df_ptable = pd.read_csv(f"{ROOT}/pymatviz/elements.csv", comment="#").set_index(
    "symbol"
)

if TYPE_CHECKING:
    from typing import TypeAlias

    ElemValues: TypeAlias = dict[str | int, int | float] | pd.Series | Sequence[str]

    CountMode = Literal[
        "element_composition", "fractional_composition", "reduced_composition"
    ]


def count_elements(
    elem_values: ElemValues,
    count_mode: CountMode = "element_composition",
    exclude_elements: Sequence[str] = (),
) -> pd.Series:
    """Processes elemental heatmap data. If passed a list of strings, assume they are
    compositions and count the occurrences of each chemical element. Else ensure the
    data is a pd.Series filled with zero values for missing element symbols.

    Source:

    Args:
        elem_values (dict[str, int | float] | pd.Series | list[str]): Iterable of
            composition strings/objects or map from element symbols to heatmap values.
        count_mode ('{element|fractional|reduced}_composition'):
            Only used when elem_values is a list of composition strings/objects.
            - element_composition (default): Count elements in each composition as is,
                i.e. without reduction or normalization.
            - fractional_composition: Convert to normalized compositions in which the
                amounts of each species sum to before counting.
                Example: Fe2 O3 -> Fe0.4 O0.6
            - reduced_composition: Convert to reduced compositions (i.e. amounts
                normalized by greatest common denominator) before counting.
                Example: Fe4 P4 O16 -> Fe P O4.
        exclude_elements (Sequence[str]): Elements to exclude from the count. Defaults
            to ().
    Returns:
        pd.Series: Map element symbols to heatmap values.
    """
    # ensure elem_values is Series if we got dict/list/tuple
    srs = pd.Series(elem_values)

    if is_numeric_dtype(srs):
        pass
    elif is_string_dtype(srs):
        # assume all items in elem_values are composition strings
        srs = pd.DataFrame(
            getattr(Composition(comp_str), count_mode).as_dict() for comp_str in srs
        ).sum()  # sum up element occurrences
    else:
        raise ValueError(
            "Expected elem_values to be map from element symbols to heatmap values or "
            f"list of compositions (strings or Pymatgen objects), got {elem_values}"
        )

    try:
        # if index consists entirely of strings representing integers, convert to ints
        srs.index = srs.index.astype(int)
    except (ValueError, TypeError):
        pass

    if pd.api.types.is_integer_dtype(srs.index):
        # if index is all integers, assume they represent atomic
        # numbers and map them to element symbols (H: 1, He: 2, ...)
        if srs.index.max() > 118 or srs.index.min() < 1:
            raise ValueError(
                "element value keys were found to be integers and assumed to represent "
                "atomic numbers, but values are outside expected range [1, 118]."
            )
        map_atomic_num_to_elem_symbol = (
            df_ptable.reset_index().set_index("atomic_number").symbol
        )
        srs.index = srs.index.map(map_atomic_num_to_elem_symbol)

    # ensure all elements are present in returned Series (with value zero if they
    # weren't in elem_values before)
    srs = srs.reindex(df_ptable.index, fill_value=0).rename("count")

    if len(exclude_elements) > 0:
        try:
            srs = srs.drop(exclude_elements)
        except KeyError as exc:
            bad_symbols = ", ".join(x for x in exclude_elements if x not in srs)
            raise ValueError(
                f"Unexpected symbol(s) {bad_symbols} in {exclude_elements=}"
            ) from exc

    return srs
