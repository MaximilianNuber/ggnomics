"""Native Plotnine composition of prepared UpSet components."""

from __future__ import annotations

from math import inf
from typing import Any, Callable, Literal, Mapping, Sequence

from plotnine.composition import Compose, plot_layout, plot_spacer

from ._types import (
    IntersectionMatrixSpec,
    SetSizeSpec,
    ThemeCollection,
    UpSetAnnotation,
    UpSetData,
    UpSetQuery,
    UpSetStripes,
)
from ._utils_query import validate_queries
from .alignment import (
    apply_left_margin_corrections,
    compute_margin_sensitivity,
    compute_panel_edges,
    suggest_left_margin_corrections,
    tag_panel_role,
)
from .annotations import build_annotation, intersection_size
from .intersections import upset_data
from .matrix import build_matrix, intersection_matrix
from .modes import Mode
from .set_size import build_set_sizes, upset_set_size
from .stripes import upset_stripes
from .themes import upset_themes

_ALIGNED_COLUMNS = (("spacer", "set_size"), ("annotation", "matrix"))


def _align_composition(composition: Compose) -> Compose:
    """Correct the cross-row panel-margin gap tagged by `compose_upset`.

    Renders `composition` twice (once to measure, once probed) to calibrate
    an exact ``plot_margin_left`` correction per :mod:`.alignment`, then
    returns a corrected composition with the same tagged panels. This is a
    workaround for a plotnine layout limitation (see :mod:`.alignment`), not
    a cheap operation; skip it via ``align_panels=False`` where the extra
    render cost matters more than sub-pixel panel alignment.
    """

    edges = compute_panel_edges(composition)
    sensitivity = compute_margin_sensitivity(composition, baseline=edges, columns=_ALIGNED_COLUMNS)
    corrections = suggest_left_margin_corrections(edges, sensitivity, columns=_ALIGNED_COLUMNS)
    return apply_left_margin_corrections(composition, corrections)


def _identity(value: str) -> str:
    return value


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
    align_panels: bool = True,
) -> Compose:
    """Compose prepared data into a native Plotnine UpSet plot.

    ``align_panels`` corrects a plotnine layout limitation where the
    annotation panel(s) and the intersection matrix -- stacked in the same
    column but different rows -- can end up with mismatched left edges when
    their y axes need different amounts of tick-label space; likewise for
    the blank corner and the set-size panel. When true (the default), this
    renders the composition twice internally (see :mod:`.alignment`) to
    measure and correct the gap for the composition's current figure size;
    pass false to skip that extra rendering cost.
    """

    del wrap, encode_sets
    if not isinstance(data, UpSetData):
        raise TypeError("data must be prepared by upset_data()")
    if height_ratio <= 0 or width_ratio <= 0:
        raise ValueError("height_ratio and width_ratio must be positive")
    validate_queries(data, queries)
    selected_themes = upset_themes if themes is None else themes
    selected_stripes = upset_stripes() if stripes is None else stripes
    selected_matrix = intersection_matrix() if matrix is None else matrix
    selected_set_sizes = upset_set_size() if set_sizes is None else set_sizes

    if base_annotations == "auto":
        panels: dict[str, Any] = {"Intersection size": intersection_size(mode=mode)}
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
        tag_panel_role(
            build_annotation(
                data,
                annotation,
                name=panel_name,
                queries=queries,
                themes=selected_themes,
            ),
            role="annotation",
        )
        for panel_name, annotation in panels.items()
    ]
    top_right = _stack_plots(annotation_plots)
    matrix_plot = tag_panel_role(
        build_matrix(
            data,
            name=name,
            matrix=selected_matrix,
            stripes=selected_stripes,
            queries=queries,
            themes=selected_themes,
            labeller=labeller,
        ),
        role="matrix",
    )

    guide_mode = "collect" if guides in {"collect", "over"} else guides
    if selected_set_sizes is False:
        composition = top_right / matrix_plot
        composition = composition + plot_layout(heights=[1.0, height_ratio], guides=guide_mode)
    else:
        set_plot = tag_panel_role(
            build_set_sizes(
                data,
                spec=selected_set_sizes,
                queries=queries,
                themes=selected_themes,
                labeller=labeller,
            ),
            role="set_size",
        )
        spacer = tag_panel_role(plot_spacer(), role="spacer")
        top = spacer | top_right
        top = top + plot_layout(widths=[width_ratio, 1.0])
        bottom = set_plot | matrix_plot
        bottom = bottom + plot_layout(widths=[width_ratio, 1.0])
        composition = top / bottom
        composition = composition + plot_layout(heights=[1.0, height_ratio], guides=guide_mode)
    if align_panels:
        composition = _align_composition(composition)
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
    align_panels: bool = True,
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
    sort_intersections_by: Sequence[Literal["cardinality", "degree", "ratio"]] = ("cardinality",),
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
        align_panels=align_panels,
    )
