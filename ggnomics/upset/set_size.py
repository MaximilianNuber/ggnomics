"""Marginal set-size panel specification."""

from __future__ import annotations

from plotnine import aes, geom_col

from ._types import SetSizeSpec


def upset_set_size(
    *,
    mapping=None,
    geom=None,
    position: str = "left",
    filter_intersections: bool = False,
) -> SetSizeSpec:
    """Construct a marginal set-size panel specification."""

    if position not in {"left", "right"}:
        raise ValueError(f"position must be 'left' or 'right'; got {position!r}")
    return SetSizeSpec(
        mapping=aes() if mapping is None else mapping,
        geom=geom if geom is not None else geom_col(width=0.6),
        position=position,
        filter_intersections=bool(filter_intersections),
    )
