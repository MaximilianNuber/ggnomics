"""Intersection-matrix stripe specification."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd
from plotnine import aes, geom_segment

from ._types import UpSetStripes


def upset_stripes(
    *,
    mapping=None,
    geom=None,
    colors: Sequence[str] | Mapping[Any, str] | None = ("white", "#F2F2F2"),
    data: pd.DataFrame | None = None,
) -> UpSetStripes:
    """Construct background stripes for the intersection matrix."""

    if data is not None:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas.DataFrame or None")
        if "set" not in data.columns:
            raise KeyError("data must contain a 'set' column")
    if isinstance(colors, Mapping):
        stored_colors = dict(colors)
    elif colors is None:
        stored_colors = None
    else:
        stored_colors = tuple(colors)
        if not stored_colors:
            raise ValueError("colors must contain at least one color")
    return UpSetStripes(
        mapping=aes() if mapping is None else mapping,
        geom=geom if geom is not None else geom_segment(size=7),
        colors=stored_colors,
        data=None if data is None else data.copy(deep=False),
    )
