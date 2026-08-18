"""Intersection matrix specification and its panel builder."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
from plotnine import (
    aes,
    geom_point,
    geom_segment,
    ggplot,
    labs,
    scale_color_identity,
    scale_color_manual,
    scale_y_continuous,
)

from ._types import (
    IntersectionMatrixSpec,
    ThemeCollection,
    UpSetData,
    UpSetQuery,
    UpSetStripes,
)
from ._utils_query import find_query_intersection, query_applies
from ._utils_theme import apply_component_theme


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


def _configured_geom(
    geom,
    *,
    data: pd.DataFrame,
    mapping: Mapping[str, str],
    aesthetics: Mapping[str, Any] | None = None,
):
    """Copy and bind a user-supplied geom without mutating its specification."""

    configured = deepcopy(geom)
    configured.data = data
    configured.mapping = aes(**{**dict(configured.mapping), **dict(mapping)})
    if aesthetics:
        configured.aes_params = {**configured.aes_params, **dict(aesthetics)}
    return configured


def build_matrix(
    data: UpSetData,
    *,
    name: str,
    matrix: IntersectionMatrixSpec,
    stripes: UpSetStripes,
    queries: Sequence[UpSetQuery],
    themes: ThemeCollection,
    labeller: Callable[[str], str],
):
    """Render the intersection-membership matrix into a Plotnine plot.

    ``data`` must be prepared by :func:`~ggnomics.upset.upset_data`. ``matrix``
    and ``stripes`` come from :func:`intersection_matrix` and
    :func:`~ggnomics.upset.upset_stripes`. Returns a Plotnine plot with one
    row per set (``data.sorted_sets``) and one column per intersection
    (``data.sorted_intersections``), background stripes, active/inactive
    points, connecting segments, ``labeller``-formatted y-axis labels (shown
    unless ``themes["intersections_matrix"]`` blanks them), ``name`` as its
    x-axis label, and query highlighting, with ``themes["intersections_matrix"]``
    (falling back to ``themes["default"]``) applied last. Consumed by
    :func:`~ggnomics.upset.compose_upset`.
    """

    frame = data.matrix_frame.copy()
    row_colors: list[str] = []
    if isinstance(stripes.colors, Mapping):
        metadata = stripes.data
        if metadata is None:
            raise ValueError("mapping-valued stripe colors require stripes.data")
        lookup = metadata.set_index("set")
        color_column = next((column for column in metadata.columns if column != "set"), None)
        if color_column is None:
            raise ValueError("stripes.data needs a metadata column in addition to 'set'")
        row_colors = [
            stripes.colors.get(lookup.at[group, color_column], "white") if group in lookup.index else "white"
            for group in data.sorted_sets
        ]
    elif stripes.colors is None:
        row_colors = ["white"] * len(data.sorted_sets)
    else:
        palette = tuple(stripes.colors)
        row_colors = [palette[position % len(palette)] for position in range(len(data.sorted_sets))]
    color_by_position = dict(enumerate(row_colors))
    frame["stripe_fill"] = frame["group_position"].map(color_by_position)

    stripe_frame = frame[["group", "group_position", "stripe_fill"]].drop_duplicates("group_position").copy()
    stripe_frame["stripe_x"] = data.sorted_intersections[0]
    stripe_frame["stripe_xend"] = data.sorted_intersections[-1]
    if stripes.data is not None:
        stripe_frame = stripe_frame.merge(
            stripes.data,
            how="left",
            left_on="group",
            right_on="set",
        )
    stripe_mapping = {
        "x": "stripe_x",
        "xend": "stripe_xend",
        "y": "group_position",
        "yend": "group_position",
        "color": "stripe_fill",
        **dict(stripes.mapping),
    }
    stripe_geom = _configured_geom(
        stripes.geom,
        data=stripe_frame,
        mapping=stripe_mapping,
    )
    plot = ggplot(frame, aes(x="intersection", y="group_position")) + stripe_geom
    if "color" in dict(stripes.mapping):
        if isinstance(stripes.colors, Mapping):
            plot = plot + scale_color_manual(values=dict(stripes.colors), guide=None, na_value="white")
    else:
        plot = plot + scale_color_identity()
    active = frame.loc[frame["value"]].copy()
    segments = active.groupby("intersection", observed=True)["group_position"].agg(y="min", yend="max").reset_index()
    if len(segments):
        plot = plot + _configured_geom(
            matrix.segment,
            data=segments,
            mapping={
                "x": "intersection",
                "xend": "intersection",
                "y": "y",
                "yend": "yend",
            },
            aesthetics={"color": matrix.outline_color["active"]},
        )
    plot = (
        plot
        + _configured_geom(
            matrix.geom,
            data=frame,
            mapping={"x": "intersection", "y": "group_position"},
            aesthetics={"color": matrix.outline_color["inactive"]},
        )
        + _configured_geom(
            matrix.geom,
            data=active,
            mapping={"x": "intersection", "y": "group_position"},
            aesthetics={"color": matrix.outline_color["active"]},
        )
    )
    for query in queries:
        if not query_applies(query, "intersections_matrix"):
            continue
        highlighted = frame.iloc[0:0]
        identifier = find_query_intersection(data, query)
        if identifier is not None:
            highlighted = active.loc[active["intersection"] == identifier]
        elif query.set is not None and query.set in data.sorted_sets:
            highlighted = frame.loc[frame["group"] == query.set]
        if len(highlighted):
            plot = plot + geom_point(
                data=highlighted,
                inherit_aes=True,
                size=3.6,
                **dict(query.aesthetics),
            )
    labels = [labeller(str(name_)) for name_ in data.sorted_sets]
    plot = plot + scale_y_continuous(breaks=list(range(len(labels))), labels=labels, expand=(0, 0.6)) + labs(x=name)
    return apply_component_theme(plot, component="intersections_matrix", themes=themes)
