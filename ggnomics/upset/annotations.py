"""Composable annotation specifications for UpSet plots."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

from plotnine import aes, geom_bar, geom_col, ggplot

from ._types import UpSetAnnotation
from .modes import Mode, normalize_mode


def _mapping_or_empty(mapping):
    return aes() if mapping is None else mapping


def intersection_size(
    *,
    mapping=None,
    counts: bool = True,
    bar_number_threshold: float = 0.85,
    text_colors: Mapping[str, str] | None = None,
    text: Mapping[str, Any] | None = None,
    text_mapping=None,
    mode: Mode = "distinct",
    position: Any = "stack",
    width: float = 0.9,
    **geom_kwargs: Any,
) -> UpSetAnnotation:
    """Construct an intersection-size bar annotation specification."""

    if not 0 <= bar_number_threshold <= 1:
        raise ValueError("bar_number_threshold must be between 0 and 1")
    if width <= 0:
        raise ValueError("width must be positive")
    colors = dict(text_colors or {"on_background": "black", "on_bar": "white"})
    missing = {"on_background", "on_bar"}.difference(colors)
    if missing:
        raise ValueError(f"text_colors is missing required keys: {sorted(missing)}")
    options = {
        "mapping": _mapping_or_empty(mapping),
        "counts": bool(counts),
        "bar_number_threshold": float(bar_number_threshold),
        "text_colors": colors,
        "text": dict(text or {}),
        "text_mapping": _mapping_or_empty(text_mapping),
        "position": position,
        "width": float(width),
        "geom_kwargs": dict(geom_kwargs),
    }
    plot = ggplot(mapping=aes(x="intersection")) + geom_bar(
        mapping=_mapping_or_empty(mapping),
        position=position,
        width=width,
        **geom_kwargs,
    )
    return UpSetAnnotation(
        plot=plot,
        kind="intersection_size",
        mode=normalize_mode(mode),
        options=MappingProxyType(options),
    )


def intersection_ratio(
    *,
    mapping=None,
    counts: bool = True,
    bar_number_threshold: float = 0.75,
    text_colors: Mapping[str, str] | None = None,
    text: Mapping[str, Any] | None = None,
    text_mapping=None,
    mode: Mode = "distinct",
    denominator_mode: Mode = "union",
    width: float = 0.9,
    **geom_kwargs: Any,
) -> UpSetAnnotation:
    """Construct an intersection-size ratio annotation specification."""

    if not 0 <= bar_number_threshold <= 1:
        raise ValueError("bar_number_threshold must be between 0 and 1")
    if width <= 0:
        raise ValueError("width must be positive")
    colors = dict(text_colors or {"on_background": "black", "on_bar": "white"})
    missing = {"on_background", "on_bar"}.difference(colors)
    if missing:
        raise ValueError(f"text_colors is missing required keys: {sorted(missing)}")
    options = {
        "mapping": _mapping_or_empty(mapping),
        "counts": bool(counts),
        "bar_number_threshold": float(bar_number_threshold),
        "text_colors": colors,
        "text": dict(text or {}),
        "text_mapping": _mapping_or_empty(text_mapping),
        "denominator_mode": normalize_mode(denominator_mode),
        "width": float(width),
        "geom_kwargs": dict(geom_kwargs),
    }
    plot = ggplot(mapping=aes(x="intersection", y="ratio")) + geom_col(
        mapping=_mapping_or_empty(mapping),
        width=width,
        **geom_kwargs,
    )
    return UpSetAnnotation(
        plot=plot,
        kind="intersection_ratio",
        mode=normalize_mode(mode),
        options=MappingProxyType(options),
    )


def upset_annotate(y: str, geom: Any | Sequence[Any]) -> UpSetAnnotation:
    """Construct a custom annotation with fixed intersection and y mappings."""

    if not isinstance(y, str) or not y:
        raise ValueError("y must be a non-empty column name")
    geoms = list(geom) if isinstance(geom, (list, tuple)) else [geom]
    if not geoms:
        raise ValueError("geom must contain at least one Plotnine layer")
    plot = ggplot(mapping=aes(x="intersection", y=y))
    for layer in geoms:
        plot = plot + layer
    return UpSetAnnotation(
        plot=plot,
        kind="custom",
        mode=normalize_mode("distinct"),
        default_y=y,
    )
