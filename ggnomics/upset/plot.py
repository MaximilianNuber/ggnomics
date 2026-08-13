"""Native Plotnine composition of prepared UpSet components."""

from __future__ import annotations

from copy import deepcopy
from math import inf
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    geom_col,
    geom_point,
    geom_text,
    ggplot,
    labs,
    scale_color_manual,
    scale_color_identity,
    scale_y_continuous,
    scale_y_reverse,
    theme,
    element_blank,
)
from plotnine.composition import Compose, plot_layout, plot_spacer

from ._types import (
    IntersectionMatrixSpec,
    SetSizeSpec,
    ThemeCollection,
    UpSetAnnotation,
    UpSetData,
    UpSetQuery,
    UpSetStripes,
    copy_plot,
)
from .annotations import intersection_size
from .intersections import upset_data
from .matrix import intersection_matrix
from .modes import Mode, normalize_mode
from .set_size import upset_set_size
from .stripes import upset_stripes
from .themes import upset_themes


def _identity(value: str) -> str:
    return value


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


def _mode_mask(
    observed: np.ndarray,
    target: np.ndarray,
    mode: str,
) -> np.ndarray:
    overlap = observed @ target
    observed_degree = observed.sum(axis=1)
    target_degree = int(target.sum())
    if target_degree == 0:
        return observed_degree == 0
    if mode == "exclusive_intersection":
        return (overlap == observed_degree) & (overlap == target_degree)
    if mode == "inclusive_intersection":
        return overlap == target_degree
    if mode == "inclusive_union":
        return overlap > 0
    if mode == "exclusive_union":
        return (overlap == observed_degree) & (observed_degree > 0)
    raise ValueError(f"unsupported canonical mode {mode!r}")


def _mode_frame(data: UpSetData, mode: str) -> pd.DataFrame:
    canonical = normalize_mode(mode)
    memberships = data.statistics.memberships.loc[
        :, list(data.statistics.sets)
    ].to_numpy(dtype=np.int8, copy=False)
    frames: list[pd.DataFrame] = []
    for identifier in data.sorted_intersections:
        members = data.intersection_members[identifier]
        target = np.asarray(
            [name in members for name in data.statistics.sets], dtype=np.int8
        )
        mask = _mode_mask(memberships, target, canonical)
        if not mask.any():
            continue
        frame = data.statistics.data.loc[mask].copy()
        frame["intersection"] = identifier
        frames.append(frame)
    if frames:
        result = pd.concat(frames, axis=0, ignore_index=False)
    else:
        result = data.statistics.data.iloc[0:0].copy()
        result["intersection"] = pd.Series(dtype="object")
    result["intersection"] = result["intersection"].astype(
        pd.CategoricalDtype(data.sorted_intersections, ordered=True)
    )
    return result


def _validate_queries(data: UpSetData, queries: Sequence[UpSetQuery]) -> None:
    """Reject group queries and unknown set/intersection names explicitly.

    Group-query highlighting is not implemented: no component currently
    reads ``UpSetQuery.group``. Rather than silently accepting and ignoring
    it, this raises so callers learn immediately rather than seeing an
    unhighlighted plot.
    """

    known_sets = set(data.statistics.sets)
    for query in queries:
        if query.group is not None:
            raise NotImplementedError(
                "upset_query(group=...) highlighting is not implemented in "
                "ggnomics.upset; only set=... and intersect=[...] queries "
                "currently affect plot components."
            )
        if query.set is not None and query.set not in known_sets:
            raise KeyError(
                f"query set {query.set!r} is not among the prepared sets "
                f"{sorted(known_sets)}"
            )
        if query.intersect is not None:
            unknown = [name for name in query.intersect if name not in known_sets]
            if unknown:
                raise KeyError(
                    f"query intersect contains unknown set names: {unknown}"
                )


def _query_intersection_id(data: UpSetData, query: UpSetQuery) -> str | None:
    if query.intersect is None:
        return None
    requested = set(query.intersect)
    canonical = tuple(name for name in data.statistics.sets if name in requested)
    for identifier, members in data.intersection_members.items():
        if members == canonical:
            return identifier
    return None


def _query_applies(query: UpSetQuery, component: str) -> bool:
    return query.only_components is None or component in query.only_components


def _apply_component_theme(
    plot,
    *,
    component: str,
    themes: ThemeCollection,
):
    components = themes.get(component, themes.get("default", ()))
    for addition in components:
        plot = plot + addition
    return plot


def _annotation_summary(data: UpSetData, mode: str) -> pd.DataFrame:
    canonical = normalize_mode(mode)
    table = data.sizes.copy().reset_index()
    table["intersection"] = table["intersection"].astype(
        pd.CategoricalDtype(data.sorted_intersections, ordered=True)
    )
    table["size"] = table[canonical].astype(float)
    return table


def _build_size_annotation(
    data: UpSetData,
    annotation: UpSetAnnotation,
    *,
    name: str,
    queries: Sequence[UpSetQuery],
    themes: ThemeCollection,
):
    frame = _mode_frame(data, annotation.mode)
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
        plot = plot + geom_text(
            data=summary,
            mapping=aes(**text_mapping),
            inherit_aes=False,
            show_legend=False,
            **text_kwargs,
        ) + scale_color_manual(
            values={color: color for color in summary["label_color"].unique()},
            guide=None,
        )

    for query in queries:
        if not _query_applies(query, name):
            continue
        selected: list[str] = []
        identifier = _query_intersection_id(data, query)
        if identifier is not None:
            selected = [identifier]
        elif query.set is not None:
            selected = [
                item
                for item in data.sorted_intersections
                if query.set in data.intersection_members[item]
            ]
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
    return _apply_component_theme(plot + labs(y=name), component=name, themes=themes)


def _build_ratio_annotation(
    data: UpSetData,
    annotation: UpSetAnnotation,
    *,
    name: str,
    queries: Sequence[UpSetQuery],
    themes: ThemeCollection,
):
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
    summary["ratio_label"] = (
        summary[numerator].astype(str) + "/" + summary[denominator].astype(str)
    )
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
        plot = plot + geom_text(
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
        ) + scale_color_manual(
            values={color: color for color in summary["label_color"].unique()},
            guide=None,
        )
    for query in queries:
        if not _query_applies(query, name):
            continue
        identifier = _query_intersection_id(data, query)
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
    return _apply_component_theme(plot + labs(y=name), component=name, themes=themes)


def _build_custom_annotation(
    data: UpSetData,
    annotation: UpSetAnnotation | Any,
    *,
    name: str,
    queries: Sequence[UpSetQuery],
    themes: ThemeCollection,
):
    if isinstance(annotation, UpSetAnnotation):
        frame = _mode_frame(data, annotation.mode)
        plot = copy_plot(annotation.plot, data=frame)
    else:
        frame = _mode_frame(data, "exclusive_intersection")
        plot = copy_plot(annotation, data=frame)
    for query in queries:
        if not _query_applies(query, name):
            continue
        identifier = _query_intersection_id(data, query)
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
    return _apply_component_theme(plot + labs(y=name), component=name, themes=themes)


def _build_annotation(
    data: UpSetData,
    annotation: UpSetAnnotation | Any,
    *,
    name: str,
    queries: Sequence[UpSetQuery],
    themes: ThemeCollection,
):
    if isinstance(annotation, UpSetAnnotation):
        if annotation.kind == "intersection_size":
            return _build_size_annotation(
                data, annotation, name=name, queries=queries, themes=themes
            )
        if annotation.kind == "intersection_ratio":
            return _build_ratio_annotation(
                data, annotation, name=name, queries=queries, themes=themes
            )
    return _build_custom_annotation(
        data, annotation, name=name, queries=queries, themes=themes
    )


def _build_matrix(
    data: UpSetData,
    *,
    name: str,
    matrix: IntersectionMatrixSpec,
    stripes: UpSetStripes,
    queries: Sequence[UpSetQuery],
    themes: ThemeCollection,
    labeller: Callable[[str], str],
):
    frame = data.matrix_frame.copy()
    row_colors: list[str] = []
    if isinstance(stripes.colors, Mapping):
        metadata = stripes.data
        if metadata is None:
            raise ValueError("mapping-valued stripe colors require stripes.data")
        lookup = metadata.set_index("set")
        color_column = next(
            (column for column in metadata.columns if column != "set"), None
        )
        if color_column is None:
            raise ValueError("stripes.data needs a metadata column in addition to 'set'")
        row_colors = [
            stripes.colors.get(lookup.at[group, color_column], "white")
            if group in lookup.index
            else "white"
            for group in data.sorted_sets
        ]
    elif stripes.colors is None:
        row_colors = ["white"] * len(data.sorted_sets)
    else:
        palette = tuple(stripes.colors)
        row_colors = [palette[position % len(palette)] for position in range(len(data.sorted_sets))]
    color_by_position = dict(enumerate(row_colors))
    frame["stripe_fill"] = frame["group_position"].map(color_by_position)

    stripe_frame = (
        frame[["group", "group_position", "stripe_fill"]]
        .drop_duplicates("group_position")
        .copy()
    )
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
            plot = plot + scale_color_manual(
                values=dict(stripes.colors), guide=None, na_value="white"
            )
    else:
        plot = plot + scale_color_identity()
    active = frame.loc[frame["value"]].copy()
    segments = (
        active.groupby("intersection", observed=True)["group_position"]
        .agg(y="min", yend="max")
        .reset_index()
    )
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
    plot = plot + _configured_geom(
        matrix.geom,
        data=frame,
        mapping={"x": "intersection", "y": "group_position"},
        aesthetics={"color": matrix.outline_color["inactive"]},
    ) + _configured_geom(
        matrix.geom,
        data=active,
        mapping={"x": "intersection", "y": "group_position"},
        aesthetics={"color": matrix.outline_color["active"]},
    )
    for query in queries:
        if not _query_applies(query, "intersections_matrix"):
            continue
        highlighted = frame.iloc[0:0]
        identifier = _query_intersection_id(data, query)
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
    plot = plot + scale_y_continuous(
        breaks=list(range(len(labels))), labels=labels, expand=(0, 0.6)
    ) + labs(x=name)
    return _apply_component_theme(
        plot, component="intersections_matrix", themes=themes
    )


def _build_set_sizes(
    data: UpSetData,
    *,
    spec: SetSizeSpec,
    queries: Sequence[UpSetQuery],
    themes: ThemeCollection,
):
    if spec.filter_intersections:
        selected_rows = data.statistics.element_intersections.isin(
            data.sorted_intersections
        )
        sizes = data.statistics.memberships.loc[
            selected_rows, list(data.sorted_sets)
        ].sum(axis=0)
    else:
        sizes = data.statistics.set_sizes.loc[list(data.sorted_sets)]
    frame = sizes.rename("size").rename_axis("group").reset_index()
    frame["group"] = frame["group"].astype(
        pd.CategoricalDtype(data.sorted_sets, ordered=True)
    )
    mapping = {**dict(spec.mapping), "x": "group", "y": "size"}
    plot = ggplot(frame, aes(**mapping)) + deepcopy(spec.geom)
    for query in queries:
        if not _query_applies(query, "overall_sizes") or query.set is None:
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
    plot = plot + coord_flip() + labs(y="Set size")
    if spec.position == "left":
        plot = plot + scale_y_reverse()
    plot = plot + theme(axis_text_y=element_blank(), axis_ticks_major_y=element_blank())
    return _apply_component_theme(plot, component="overall_sizes", themes=themes)


def _stack_plots(plots: Sequence[Any]):
    result = plots[0]
    for plot in plots[1:]:
        result = result / plot
    if len(plots) > 1:
        result = result + plot_layout(heights=[1.0] * len(plots))
    return result


def compose_upset(
    data: UpSetData,
    *,
    base_annotations: Literal["auto"] | Mapping[str, UpSetAnnotation] = "auto",
    name: str = "group",
    annotations: Mapping[str, UpSetAnnotation | Any] | None = None,
    themes: ThemeCollection | None = None,
    stripes: UpSetStripes | None = None,
    labeller: Callable[[str], str] = _identity,
    height_ratio: float = 0.5,
    width_ratio: float = 0.3,
    wrap: bool = False,
    set_sizes: SetSizeSpec | Literal[False] | None = None,
    mode: Mode = "distinct",
    queries: Sequence[UpSetQuery] = (),
    guides: Literal["keep", "collect", "over"] | None = None,
    encode_sets: bool = True,
    matrix: IntersectionMatrixSpec | None = None,
) -> Compose:
    """Compose prepared data into a native Plotnine UpSet plot."""

    del wrap, encode_sets
    if not isinstance(data, UpSetData):
        raise TypeError("data must be prepared by upset_data()")
    if height_ratio <= 0 or width_ratio <= 0:
        raise ValueError("height_ratio and width_ratio must be positive")
    _validate_queries(data, queries)
    selected_themes = upset_themes if themes is None else themes
    selected_stripes = upset_stripes() if stripes is None else stripes
    selected_matrix = intersection_matrix() if matrix is None else matrix
    selected_set_sizes = upset_set_size() if set_sizes is None else set_sizes

    if base_annotations == "auto":
        panels: dict[str, Any] = {
            "Intersection size": intersection_size(mode=mode)
        }
    elif isinstance(base_annotations, Mapping):
        panels = dict(base_annotations)
    else:
        raise TypeError("base_annotations must be 'auto' or a mapping")
    if annotations:
        overlap = set(panels).intersection(annotations)
        if overlap:
            raise ValueError(f"duplicate annotation names: {sorted(overlap)}")
        panels.update(annotations)
    if not panels:
        raise ValueError("at least one annotation panel is required")

    annotation_plots = [
        _build_annotation(
            data,
            annotation,
            name=panel_name,
            queries=queries,
            themes=selected_themes,
        )
        for panel_name, annotation in panels.items()
    ]
    top_right = _stack_plots(annotation_plots)
    matrix_plot = _build_matrix(
        data,
        name=name,
        matrix=selected_matrix,
        stripes=selected_stripes,
        queries=queries,
        themes=selected_themes,
        labeller=labeller,
    )

    guide_mode = "collect" if guides in {"collect", "over"} else guides
    if selected_set_sizes is False:
        composition = top_right / matrix_plot
        composition = composition + plot_layout(
            heights=[1.0, height_ratio], guides=guide_mode
        )
    else:
        set_plot = _build_set_sizes(
            data,
            spec=selected_set_sizes,
            queries=queries,
            themes=selected_themes,
        )
        top = plot_spacer() | top_right
        top = top + plot_layout(widths=[width_ratio, 1.0])
        bottom = set_plot | matrix_plot
        bottom = bottom + plot_layout(widths=[width_ratio, 1.0])
        composition = top / bottom
        composition = composition + plot_layout(
            heights=[1.0, height_ratio], guides=guide_mode
        )
    return composition


def upset(
    data,
    intersect: Sequence[str] | None = None,
    *,
    base_annotations: Literal["auto"] | Mapping[str, UpSetAnnotation] = "auto",
    name: str = "group",
    annotations: Mapping[str, UpSetAnnotation | Any] | None = None,
    themes: ThemeCollection | None = None,
    stripes: UpSetStripes | None = None,
    labeller: Callable[[str], str] = _identity,
    height_ratio: float = 0.5,
    width_ratio: float = 0.3,
    wrap: bool = False,
    set_sizes: SetSizeSpec | Literal[False] | None = None,
    mode: Mode = "distinct",
    queries: Sequence[UpSetQuery] = (),
    guides: Literal["keep", "collect", "over"] | None = None,
    encode_sets: bool = True,
    matrix: IntersectionMatrixSpec | None = None,
    min_size: int = 0,
    max_size: float = inf,
    min_degree: int = 0,
    max_degree: float = inf,
    n_intersections: int | None = None,
    keep_empty_groups: bool = False,
    warn_when_dropping_groups: bool = False,
    warn_when_converting: bool | Literal["auto"] = "auto",
    sort_sets: Literal["ascending", "descending", False] = "descending",
    sort_intersections: Literal["ascending", "descending", False] = "descending",
    sort_intersections_by: Sequence[Literal["cardinality", "degree", "ratio"]] = (
        "cardinality",
    ),
    sort_ratio_numerator: Mode = "exclusive_intersection",
    sort_ratio_denominator: Mode = "inclusive_union",
    group_by: Literal["degree", "sets"] = "degree",
    size_columns_suffix: str = "_size",
    max_combinations_datapoints_n: int = 10_000_000_000,
    intersections: Literal["observed", "all"] | Sequence[Sequence[str]] = "observed",
) -> Compose:
    """Compose an UpSet plot from raw data using ComplexUpset-style arguments."""

    if isinstance(data, UpSetData):
        if intersect is not None:
            raise TypeError("intersect must be omitted when data is already UpSetData")
        prepared = data
    else:
        if intersect is None:
            raise TypeError("intersect must be provided for a raw pandas.DataFrame")
        prepared = upset_data(
            data,
            intersect,
            min_size=min_size,
            max_size=max_size,
            min_degree=min_degree,
            max_degree=max_degree,
            n_intersections=n_intersections,
            keep_empty_groups=keep_empty_groups,
            warn_when_dropping_groups=warn_when_dropping_groups,
            warn_when_converting=warn_when_converting,
            sort_sets=sort_sets,
            sort_intersections=sort_intersections,
            sort_intersections_by=sort_intersections_by,
            sort_ratio_numerator=sort_ratio_numerator,
            sort_ratio_denominator=sort_ratio_denominator,
            group_by=group_by,
            mode=mode,
            size_columns_suffix=size_columns_suffix,
            encode_sets=encode_sets,
            max_combinations_datapoints_n=max_combinations_datapoints_n,
            intersections=intersections,
        )
    return compose_upset(
        prepared,
        base_annotations=base_annotations,
        name=name,
        annotations=annotations,
        themes=themes,
        stripes=stripes,
        labeller=labeller,
        height_ratio=height_ratio,
        width_ratio=width_ratio,
        wrap=wrap,
        set_sizes=set_sizes,
        mode=mode,
        queries=queries,
        guides=guides,
        encode_sets=encode_sets,
        matrix=matrix,
    )
