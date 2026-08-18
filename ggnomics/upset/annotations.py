"""Composable annotation specifications and their panel builders."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    geom_bar,
    geom_col,
    geom_point,
    geom_text,
    ggplot,
    labs,
    scale_color_manual,
)

from ._types import ThemeCollection, UpSetAnnotation, UpSetData, UpSetQuery, copy_plot
from ._utils_query import find_query_intersection, query_applies
from ._utils_theme import apply_component_theme
from .intersections import select_mode_observations
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


def _annotation_summary(data: UpSetData, mode: str) -> pd.DataFrame:
    canonical = normalize_mode(mode)
    table = data.sizes.copy().reset_index()
    table["intersection"] = table["intersection"].astype(pd.CategoricalDtype(data.sorted_intersections, ordered=True))
    table["size"] = table[canonical].astype(float)
    return table


def build_size_annotation(
    data: UpSetData,
    annotation: UpSetAnnotation,
    *,
    name: str,
    queries: Sequence[UpSetQuery],
    themes: ThemeCollection,
):
    """Render an ``intersection_size`` annotation into a Plotnine plot.

    ``data`` must be prepared by :func:`~ggnomics.upset.upset_data`.
    ``annotation`` must have ``kind == "intersection_size"``, as constructed
    by :func:`intersection_size`. Returns a Plotnine plot bound to
    ``annotation.plot``'s geoms plus count labels and query highlighting,
    with ``name`` as its y-axis label and ``themes[name]`` (falling back to
    ``themes["default"]``) applied last. Consumed by
    :func:`~ggnomics.upset.compose_upset` via :func:`build_annotation`.
    """

    frame = select_mode_observations(data, mode=annotation.mode)
    plot = copy_plot(annotation.plot, data=frame)
    summary = _annotation_summary(data, annotation.mode)
    options = annotation.options
    if options.get("counts", True):
        maximum = float(summary["size"].max()) if len(summary) else 0.0
        threshold = float(options.get("bar_number_threshold", 0.85))
        summary["label_y"] = np.where(
            summary["size"] <= threshold * maximum,
            summary["size"],
            threshold * summary["size"],
        )
        summary["label_color"] = np.where(
            summary["size"] <= threshold * maximum,
            options["text_colors"]["on_background"],
            options["text_colors"]["on_bar"],
        )
        summary["size_label"] = summary["size"].map(lambda value: f"{value:g}")
        text_kwargs = {"va": "bottom", **dict(options.get("text", {}))}
        text_mapping = {
            "x": "intersection",
            "y": "label_y",
            "label": "size_label",
            "color": "label_color",
            **dict(options.get("text_mapping", {})),
        }
        plot = (
            plot
            + geom_text(
                data=summary,
                mapping=aes(**text_mapping),
                inherit_aes=False,
                show_legend=False,
                **text_kwargs,
            )
            + scale_color_manual(
                values={color: color for color in summary["label_color"].unique()},
                guide=None,
            )
        )

    for query in queries:
        if not query_applies(query, name):
            continue
        selected: list[str] = []
        identifier = find_query_intersection(data, query)
        if identifier is not None:
            selected = [identifier]
        elif query.set is not None:
            selected = [item for item in data.sorted_intersections if query.set in data.intersection_members[item]]
        if not selected:
            continue
        highlighted = summary.loc[summary["intersection"].isin(selected)]
        aesthetics = dict(query.aesthetics)
        plot = plot + geom_col(
            data=highlighted,
            mapping=aes(x="intersection", y="size"),
            inherit_aes=False,
            width=float(options.get("width", 0.9)),
            **aesthetics,
        )
    return apply_component_theme(plot + labs(y=name), component=name, themes=themes)


def build_ratio_annotation(
    data: UpSetData,
    annotation: UpSetAnnotation,
    *,
    name: str,
    queries: Sequence[UpSetQuery],
    themes: ThemeCollection,
):
    """Render an ``intersection_ratio`` annotation into a Plotnine plot.

    ``data`` must be prepared by :func:`~ggnomics.upset.upset_data`.
    ``annotation`` must have ``kind == "intersection_ratio"``, as constructed
    by :func:`intersection_ratio`. Returns a Plotnine plot bound to
    ``annotation.plot``'s geoms plus ratio labels and query highlighting,
    with ``name`` as its y-axis label and ``themes[name]`` (falling back to
    ``themes["default"]``) applied last. Consumed by
    :func:`~ggnomics.upset.compose_upset` via :func:`build_annotation`.
    """

    numerator = normalize_mode(annotation.mode)
    denominator = normalize_mode(annotation.options["denominator_mode"])
    summary = data.sizes.reset_index().copy()
    summary["intersection"] = summary["intersection"].astype(
        pd.CategoricalDtype(data.sorted_intersections, ordered=True)
    )
    summary["ratio"] = np.divide(
        summary[numerator],
        summary[denominator],
        out=np.zeros(len(summary), dtype=float),
        where=summary[denominator].to_numpy() != 0,
    )
    summary["ratio_label"] = summary[numerator].astype(str) + "/" + summary[denominator].astype(str)
    plot = copy_plot(annotation.plot, data=summary)
    if annotation.options.get("counts", True):
        maximum = float(summary["ratio"].max()) if len(summary) else 0.0
        threshold = float(annotation.options.get("bar_number_threshold", 0.75))
        summary["label_y"] = np.where(
            summary["ratio"] <= threshold * maximum,
            summary["ratio"],
            threshold * summary["ratio"],
        )
        summary["label_color"] = np.where(
            summary["ratio"] <= threshold * maximum,
            annotation.options["text_colors"]["on_background"],
            annotation.options["text_colors"]["on_bar"],
        )
        plot = (
            plot
            + geom_text(
                data=summary,
                mapping=aes(
                    x="intersection",
                    y="label_y",
                    label="ratio_label",
                    color="label_color",
                ),
                inherit_aes=False,
                show_legend=False,
                **dict(annotation.options.get("text", {})),
            )
            + scale_color_manual(
                values={color: color for color in summary["label_color"].unique()},
                guide=None,
            )
        )
    for query in queries:
        if not query_applies(query, name):
            continue
        identifier = find_query_intersection(data, query)
        if identifier is None:
            continue
        highlighted = summary.loc[summary["intersection"] == identifier]
        plot = plot + geom_col(
            data=highlighted,
            mapping=aes(x="intersection", y="ratio"),
            inherit_aes=False,
            width=float(annotation.options.get("width", 0.9)),
            **dict(query.aesthetics),
        )
    return apply_component_theme(plot + labs(y=name), component=name, themes=themes)


def build_custom_annotation(
    data: UpSetData,
    annotation: UpSetAnnotation | Any,
    *,
    name: str,
    queries: Sequence[UpSetQuery],
    themes: ThemeCollection,
):
    """Render a user-supplied annotation (:func:`upset_annotate` or a raw geom).

    ``data`` must be prepared by :func:`~ggnomics.upset.upset_data``.
    ``annotation`` is either a :class:`UpSetAnnotation` built by
    :func:`upset_annotate`, or any Plotnine layer/sequence of layers used
    directly as a ``base_annotations`` or ``annotations`` value. Returns a
    Plotnine plot over the ``"exclusive_intersection"`` observation rows (or
    ``annotation.mode``'s rows, for an ``UpSetAnnotation``) plus query
    highlighting, with ``name`` as its y-axis label and ``themes[name]``
    (falling back to ``themes["default"]``) applied last. Consumed by
    :func:`~ggnomics.upset.compose_upset` via :func:`build_annotation`.
    """

    if isinstance(annotation, UpSetAnnotation):
        frame = select_mode_observations(data, mode=annotation.mode)
        plot = copy_plot(annotation.plot, data=frame)
    else:
        frame = select_mode_observations(data, mode="exclusive_intersection")
        plot = copy_plot(annotation, data=frame)
    for query in queries:
        if not query_applies(query, name):
            continue
        identifier = find_query_intersection(data, query)
        if identifier is None:
            continue
        highlighted = frame.loc[frame["intersection"] == identifier]
        y = annotation.default_y if isinstance(annotation, UpSetAnnotation) else None
        if y is not None:
            plot = plot + geom_point(
                data=highlighted,
                mapping=aes(x="intersection", y=y),
                inherit_aes=False,
                **dict(query.aesthetics),
            )
    return apply_component_theme(plot + labs(y=name), component=name, themes=themes)


def build_annotation(
    data: UpSetData,
    annotation: UpSetAnnotation | Any,
    *,
    name: str,
    queries: Sequence[UpSetQuery],
    themes: ThemeCollection,
):
    """Route one annotation to its panel builder by kind.

    ``annotation.kind`` is a closed, package-internal tag (not a public
    extension point), so this dispatches on the tag directly rather than via
    :func:`functools.singledispatch`; a non-``UpSetAnnotation`` value (a raw
    Plotnine layer) is routed by the one structural ``isinstance`` check.
    Returns whatever :func:`build_size_annotation`,
    :func:`build_ratio_annotation`, or :func:`build_custom_annotation`
    returns. Consumed by :func:`~ggnomics.upset.compose_upset`.
    """

    if isinstance(annotation, UpSetAnnotation):
        if annotation.kind == "intersection_size":
            return build_size_annotation(data, annotation, name=name, queries=queries, themes=themes)
        if annotation.kind == "intersection_ratio":
            return build_ratio_annotation(data, annotation, name=name, queries=queries, themes=themes)
    return build_custom_annotation(data, annotation, name=name, queries=queries, themes=themes)
