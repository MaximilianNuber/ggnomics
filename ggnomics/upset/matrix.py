"""Intersection matrix specification."""

from __future__ import annotations

from typing import Mapping

from plotnine import geom_point, geom_segment

from ._types import IntersectionMatrixSpec


def intersection_matrix(
    *,
    geom=None,
    segment=None,
    outline_color: Mapping[str, str] | None = None,
) -> IntersectionMatrixSpec:
    """Construct the layers used for an intersection-membership matrix."""

    colors = dict(outline_color or {"active": "black", "inactive": "#B3B3B3"})
    missing = {"active", "inactive"}.difference(colors)
    if missing:
        raise ValueError(f"outline_color is missing required keys: {sorted(missing)}")
    return IntersectionMatrixSpec(
        geom=geom if geom is not None else geom_point(size=3),
        segment=segment if segment is not None else geom_segment(),
        outline_color=colors,
    )
