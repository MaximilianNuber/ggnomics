"""Marginal set-size panel specification and its panel builder."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Sequence

from plotnine import (
    aes,
    coord_flip,
    geom_col,
    ggplot,
    labs,
    scale_x_discrete,
    scale_y_reverse,
)

from .._utils import display_dtype
from ._types import SetSizeSpec, ThemeCollection, UpSetData, UpSetQuery
from ._utils_query import query_applies
from ._utils_theme import apply_component_theme


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


def build_set_sizes(
    data: UpSetData,
    *,
    spec: SetSizeSpec,
    queries: Sequence[UpSetQuery],
    themes: ThemeCollection,
    labeller: Callable[[str], str],
):
    """Render the marginal set-size panel into a Plotnine plot.

    ``data`` must be prepared by :func:`~ggnomics.upset.upset_data`. ``spec``
    comes from :func:`upset_set_size`. Returns a flipped bar plot with one
    bar per set (``data.sorted_sets``), ``labeller``-formatted labels on the
    axis shared with the intersection matrix (shown unless
    ``themes["overall_sizes"]`` blanks them), reversed when
    ``spec.position == "left"``, plus query highlighting, with
    ``themes["overall_sizes"]`` (falling back to ``themes["default"]``)
    applied last. Consumed by :func:`~ggnomics.upset.compose_upset`.
    """

    if spec.filter_intersections:
        selected_rows = data.statistics.element_intersections.isin(data.sorted_intersections)
        sizes = data.statistics.memberships.loc[selected_rows, list(data.sorted_sets)].sum(axis=0)
    else:
        sizes = data.statistics.set_sizes.loc[list(data.sorted_sets)]
    frame = sizes.rename("size").rename_axis("group").reset_index()
    frame["group"] = frame["group"].astype(display_dtype(data.sorted_sets))
    mapping = {**dict(spec.mapping), "x": "group", "y": "size"}
    plot = ggplot(frame, aes(**mapping)) + deepcopy(spec.geom)
    for query in queries:
        if not query_applies(query, "overall_sizes") or query.set is None:
            continue
        highlighted = frame.loc[frame["group"] == query.set]
        if len(highlighted):
            plot = plot + geom_col(
                data=highlighted,
                mapping=aes(x="group", y="size"),
                inherit_aes=False,
                width=0.6,
                **dict(query.aesthetics),
            )
    labels = [labeller(str(name_)) for name_ in data.sorted_sets]
    plot = plot + scale_x_discrete(labels=labels) + coord_flip() + labs(y="Set size")
    if spec.position == "left":
        plot = plot + scale_y_reverse()
    return apply_component_theme(plot, component="overall_sizes", themes=themes)
