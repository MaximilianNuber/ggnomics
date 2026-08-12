"""Statistical plot functions: violin/box with significance bars, marginal scatter, embedding panel."""

from __future__ import annotations

from functools import singledispatch
from itertools import combinations
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_violin,
    geom_boxplot,
    geom_jitter,
    theme_classic,
    theme,
    element_text,
    ggtitle,
    labs,
    scale_fill_manual,
    scale_fill_brewer,
    scale_y_continuous,
)

from ._utils import adaptive_size
from .signif._geom import _DeferredSignif, geom_signif

if TYPE_CHECKING:
    from plotnine.composition import Compose

_MARGINAL_MODES = ("density", "histogram", "boxplot")


def _unsupported_type(function_name: str, data: object) -> TypeError:
    return TypeError(
        f"{function_name} does not support {type(data).__module__}."
        f"{type(data).__qualname__}. Pass a pandas.DataFrame or install the "
        "optional dependency for a supported genomics container."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_annotation_mode(annotation):
    """Translate plot_violin_stats annotation string into map_signif_level value."""
    if annotation == "stars":
        return True
    if annotation == "pvalue":
        return lambda p: f"p={p:.3f}"
    if annotation == "padj":
        return lambda p: f"padj={p:.3f}"
    if callable(annotation):
        return annotation
    raise ValueError(f"Unknown annotation mode: {annotation!r}")


def _build_violin_data(
    feature_series: pd.Series,
    group_series: pd.Series,
    order: Optional[List],
) -> Tuple[pd.DataFrame, List]:
    """Combine already-resolved feature/group Series into a long DataFrame."""

    df = pd.DataFrame({
        "__feature__": feature_series.to_numpy(),
        "__group__": group_series.to_numpy(),
    })

    unique_groups = sorted(df["__group__"].unique().astype(str).tolist())
    if order is not None:
        groups_order = [str(g) for g in order if str(g) in unique_groups]
        groups_order += [g for g in unique_groups if g not in groups_order]
    else:
        groups_order = unique_groups

    df["__group__"] = df["__group__"].astype(str)
    df["__group__"] = pd.Categorical(df["__group__"], categories=groups_order, ordered=True)

    return df, groups_order


def _build_stat_annotated_plot(
    df: pd.DataFrame,
    groups_order: List,
    group_by: str,
    feature: str,
    geom_layer,
    comparisons: Optional[List[Tuple]],
    test: str,
    p_adjust: str,
    sig_only: bool,
    annotation,
    palette: Optional[Dict],
    x_label: Optional[str],
    y_label: Optional[str],
    title: Optional[str],
) -> ggplot:
    """Shared base-plot + significance-bracket construction for violin/box."""

    p = (
        ggplot(df)
        + aes(x="__group__", y="__feature__", fill="__group__")
        + geom_layer
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + labs(x=x_label or group_by, y=y_label or feature, fill=group_by)
    )

    if palette is not None:
        p = p + scale_fill_manual(breaks=list(palette.keys()), values=list(palette.values()))
    else:
        p = p + scale_fill_brewer(type="qual", palette="Set2")

    if title is not None:
        p = p + ggtitle(title)

    actual_comps: List[Tuple] = []
    if comparisons is not None:
        actual_comps = [c for c in comparisons if len(c) == 2]
    elif len(groups_order) <= 5:
        actual_comps = list(combinations(groups_order, 2))

    if actual_comps:
        deferred = geom_signif(
            comparisons=actual_comps,
            test=test,
            map_signif_level=_resolve_annotation_mode(annotation),
            p_adjust=p_adjust,
            sig_only=sig_only,
            margin_top=0.05,
            step_increase=0.12,
            tip_length=0.03,
        )
        signif_layers = deferred.resolve(df, x_col="__group__", y_col="__feature__")
        for layer_ in signif_layers:
            p = p + layer_

        if deferred._last_brackets:
            y_data_min = float(df["__feature__"].min())
            y_data_max = float(df["__feature__"].max())
            y_data_range = y_data_max - y_data_min or 1.0
            y_upper = max(b.y_bracket for b in deferred._last_brackets) + y_data_range * 0.08
            y_lower = y_data_min - y_data_range * 0.02
            p = p + scale_y_continuous(limits=(y_lower, y_upper))

    return p


# ---------------------------------------------------------------------------
# Public: plot_violin_stats
# ---------------------------------------------------------------------------


@singledispatch
def plot_violin_stats(
    data: pd.DataFrame,
    feature: str,
    group_by: str,
    layer: Optional[str] = None,
    comparisons: Optional[List[Tuple]] = None,
    test: str = "mannwhitney",
    p_adjust: str = "bonferroni",
    sig_only: bool = True,
    annotation: str = "stars",
    add_boxplot: bool = True,
    add_points: bool = False,
    point_size: Optional[float] = None,
    palette: Optional[Dict] = None,
    order: Optional[List] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    dodge: bool = False,
) -> ggplot:
    """Violin plot with automatic statistical significance annotations.

    Runs pairwise tests between groups defined by ``comparisons`` (or all
    pairs when the number of groups is ``<= 5`` and ``comparisons`` is
    ``None``). Draws significance brackets as ``geom_segment``/``geom_text``
    layers via :func:`ggnomics.signif.geom_signif`, using proportional
    spacing (fractions of the y-range) so brackets look correct regardless
    of data scale.

    Args:
        data: DataFrame whose rows are cells/samples. ``feature`` and
            ``group_by`` are columns.
        feature: Column holding the numeric value to plot. For a plain
            DataFrame this is simply a column; container adapters may also
            resolve it against an expression feature.
        group_by: Column whose categories define the x-axis groups.
        layer: Present for cross-container signature consistency. Has no
            effect for a plain DataFrame, which has no concept of layers.
        comparisons: Explicit list of ``(group_a, group_b)`` tuples to test.
            ``None`` auto-generates all pairs when there are ``<= 5`` groups.
        test: One of the tests supported by
            :func:`ggnomics.signif.run_comparisons`.
        p_adjust: Multiple-testing correction method.
        sig_only: Only draw brackets for significant comparisons.
        annotation: ``"stars"``, ``"pvalue"``, ``"padj"``, or a callable
            mapping a p-value to a label.
        add_boxplot: Overlay a narrow boxplot on each violin.
        add_points: Overlay jittered points.
        point_size: Jittered point size (``None`` → adaptive).
        palette: ``{category: hex}`` fill color mapping.
        order: Explicit group ordering (unlisted groups appended at the end).
        x_label: Override x-axis label (defaults to ``group_by``).
        y_label: Override y-axis label (defaults to ``feature``).
        title: Plot title.
        dodge: Reserved for future grouped-violin support.

    Returns:
        A ``plotnine.ggplot`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        KeyError: If a requested column is absent.
    """
    raise _unsupported_type("plot_violin_stats", data)


@plot_violin_stats.register(pd.DataFrame)
def _plot_violin_stats_dataframe(
    data: pd.DataFrame,
    feature: str,
    group_by: str,
    layer: Optional[str] = None,
    comparisons: Optional[List[Tuple]] = None,
    test: str = "mannwhitney",
    p_adjust: str = "bonferroni",
    sig_only: bool = True,
    annotation: str = "stars",
    add_boxplot: bool = True,
    add_points: bool = False,
    point_size: Optional[float] = None,
    palette: Optional[Dict] = None,
    order: Optional[List] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    dodge: bool = False,
) -> ggplot:
    for column in (feature, group_by):
        if column not in data.columns:
            raise KeyError(
                f"Column {column!r} not found in the DataFrame. "
                f"Available: {list(data.columns)[:20]}"
            )

    df, groups_order = _build_violin_data(data[feature], data[group_by], order)

    p = _build_stat_annotated_plot(
        df, groups_order, group_by, feature,
        geom_violin(scale="width", trim=True),
        comparisons, test, p_adjust, sig_only, annotation, palette,
        x_label, y_label, title,
    )

    if add_boxplot:
        p = p + geom_boxplot(width=0.1, fill="white", outlier_alpha=0.3)
    if add_points:
        ps = point_size or adaptive_size(len(df), size_max=0.8)
        p = p + geom_jitter(width=0.2, height=0.0, size=ps, alpha=0.4)

    return p


# ---------------------------------------------------------------------------
# Public: plot_box_stats
# ---------------------------------------------------------------------------


@singledispatch
def plot_box_stats(
    data: pd.DataFrame,
    feature: str,
    group_by: str,
    layer: Optional[str] = None,
    comparisons: Optional[List[Tuple]] = None,
    test: str = "mannwhitney",
    p_adjust: str = "bonferroni",
    sig_only: bool = True,
    annotation: str = "stars",
    add_points: bool = False,
    point_size: Optional[float] = None,
    palette: Optional[Dict] = None,
    order: Optional[List] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
) -> ggplot:
    """Box plot with automatic statistical significance annotations.

    Same semantics as :func:`plot_violin_stats` but uses ``geom_boxplot`` as
    the base layer, sharing all stat-annotation logic.

    Args:
        data: DataFrame whose rows are cells/samples. ``feature`` and
            ``group_by`` are columns.
        feature: Column holding the numeric value to plot. For a plain
            DataFrame this is simply a column; container adapters may also
            resolve it against an expression feature.
        group_by: Column whose categories define the x-axis groups.
        layer: Present for cross-container signature consistency. Has no
            effect for a plain DataFrame, which has no concept of layers.
        comparisons: Explicit list of ``(group_a, group_b)`` tuples to test.
            ``None`` auto-generates all pairs when there are ``<= 5`` groups.
        test: One of the tests supported by
            :func:`ggnomics.signif.run_comparisons`.
        p_adjust: Multiple-testing correction method.
        sig_only: Only draw brackets for significant comparisons.
        annotation: ``"stars"``, ``"pvalue"``, ``"padj"``, or a callable
            mapping a p-value to a label.
        add_points: Overlay jittered points.
        point_size: Jittered point size (``None`` → adaptive).
        palette: ``{category: hex}`` fill color mapping.
        order: Explicit group ordering (unlisted groups appended at the end).
        x_label: Override x-axis label (defaults to ``group_by``).
        y_label: Override y-axis label (defaults to ``feature``).
        title: Plot title.

    Returns:
        A ``plotnine.ggplot`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        KeyError: If a requested column is absent.
    """
    raise _unsupported_type("plot_box_stats", data)


@plot_box_stats.register(pd.DataFrame)
def _plot_box_stats_dataframe(
    data: pd.DataFrame,
    feature: str,
    group_by: str,
    layer: Optional[str] = None,
    comparisons: Optional[List[Tuple]] = None,
    test: str = "mannwhitney",
    p_adjust: str = "bonferroni",
    sig_only: bool = True,
    annotation: str = "stars",
    add_points: bool = False,
    point_size: Optional[float] = None,
    palette: Optional[Dict] = None,
    order: Optional[List] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
) -> ggplot:
    for column in (feature, group_by):
        if column not in data.columns:
            raise KeyError(
                f"Column {column!r} not found in the DataFrame. "
                f"Available: {list(data.columns)[:20]}"
            )

    df, groups_order = _build_violin_data(data[feature], data[group_by], order)

    p = _build_stat_annotated_plot(
        df, groups_order, group_by, feature,
        geom_boxplot(outlier_alpha=0.5),
        comparisons, test, p_adjust, sig_only, annotation, palette,
        x_label, y_label, title,
    )

    if add_points:
        ps = point_size or adaptive_size(len(df), size_max=0.8)
        p = p + geom_jitter(width=0.2, height=0.0, size=ps, alpha=0.4)

    return p


# ---------------------------------------------------------------------------
# Public: plot_scatter_marginal
# ---------------------------------------------------------------------------


@singledispatch
def plot_scatter_marginal(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    layer: Optional[str] = None,
    marginal: str = "density",
    size: Optional[float] = None,
    stroke: Optional[float] = None,
    alpha: float = 0.7,
    palette: Optional[Dict] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
) -> "Compose":
    """Scatter plot with marginal distributions.

    Composed from three plotnine panels (scatter, x-marginal, y-marginal)
    using plotnine's native composition system (requires plotnine >= 0.15).
    Width ratio scatter : right-marginal = 3 : 1; height ratio top-marginal :
    scatter = 1 : 3 (on plotnine >= 0.16).

    Args:
        data: DataFrame whose rows are cells/samples. ``x``, ``y``, and
            ``color`` are columns.
        x: Column mapped to the x-axis and top marginal.
        y: Column mapped to the y-axis and right marginal.
        color: Optional column mapped to point/marginal fill color.
        layer: Present for cross-container signature consistency. Has no
            effect for a plain DataFrame, which has no concept of layers.
        marginal: One of ``"density"``, ``"histogram"``, or ``"boxplot"``.
        size: Point size (``None`` → adaptive).
        stroke: Point stroke width (``None`` → adaptive).
        alpha: Point transparency.
        palette: ``{category: hex}`` color mapping.
        cmap: Matplotlib colormap for a continuous ``color``.
        title: Plot title (applied to the scatter panel).
        x_label: Override x-axis label.
        y_label: Override y-axis label.

    Returns:
        A ``plotnine.composition.Compose`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        KeyError: If a requested column is absent.
        ValueError: If ``marginal`` is not a supported mode.
    """
    raise _unsupported_type("plot_scatter_marginal", data)


@plot_scatter_marginal.register(pd.DataFrame)
def _plot_scatter_marginal_dataframe(
    data: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    layer: Optional[str] = None,
    marginal: str = "density",
    size: Optional[float] = None,
    stroke: Optional[float] = None,
    alpha: float = 0.7,
    palette: Optional[Dict] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
):
    from plotnine import geom_density, geom_histogram, geom_boxplot as _gbp, coord_flip, element_blank
    from plotnine.composition import plot_spacer
    from .scatter import plot_scatter
    from ._compose import HAS_LAYOUT

    if marginal not in _MARGINAL_MODES:
        raise ValueError(
            f"marginal must be one of {_MARGINAL_MODES}, got {marginal!r}."
        )

    required = [x, y] + ([color] if color is not None else [])
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in the DataFrame. "
            f"Available: {list(data.columns)[:20]}"
        )

    color_is_cont = bool(color is not None and pd.api.types.is_numeric_dtype(data[color]))

    marg_df = pd.DataFrame({"_mx_": data[x].to_numpy(), "_my_": data[y].to_numpy()})
    if color is not None:
        marg_df["_mc_"] = data[color].to_numpy()

    fill_col = "_mc_" if color is not None else None
    aes_x_kwargs: dict = {"x": "_mx_"}
    aes_y_kwargs: dict = {"x": "_my_"}
    if fill_col and not color_is_cont:
        aes_x_kwargs["fill"] = fill_col
        aes_y_kwargs["fill"] = fill_col

    _strip_top = theme(axis_title_x=element_blank(), legend_position="none")
    _strip_right = theme(axis_title_y=element_blank(), legend_position="none")

    if marginal == "density":
        p_top = (
            ggplot(marg_df) + aes(**aes_x_kwargs) + geom_density(alpha=0.4)
            + theme_classic() + labs(y="Density") + _strip_top
        )
        p_right = (
            ggplot(marg_df) + aes(**aes_y_kwargs) + geom_density(alpha=0.4)
            + coord_flip() + theme_classic() + labs(x="Density") + _strip_right
        )
    elif marginal == "histogram":
        p_top = (
            ggplot(marg_df) + aes(**aes_x_kwargs) + geom_histogram(bins=30, alpha=0.6)
            + theme_classic() + _strip_top
        )
        p_right = (
            ggplot(marg_df) + aes(**aes_y_kwargs) + geom_histogram(bins=30, alpha=0.6)
            + coord_flip() + theme_classic() + _strip_right
        )
    else:  # boxplot
        p_top = (
            ggplot(marg_df) + aes(x="_mx_", y="_mx_") + _gbp()
            + theme_classic() + _strip_top
        )
        p_right = (
            ggplot(marg_df) + aes(x="_my_", y="_my_") + _gbp()
            + theme_classic() + _strip_right
        )

    p_scatter = plot_scatter(
        data, x=x, y=y, color=color, layer=None, size=size, stroke=stroke,
        alpha=alpha, palette=palette, cmap=cmap, x_label=x_label, y_label=y_label,
        title=title,
    )

    top_row = p_top | plot_spacer()
    bot_row = p_scatter | p_right
    composition = top_row / bot_row

    if HAS_LAYOUT:
        from plotnine.composition import plot_layout
        composition = composition + plot_layout(widths=[3, 1], heights=[1, 3])

    return composition


# ---------------------------------------------------------------------------
# Public: plot_embedding_panel
# ---------------------------------------------------------------------------


@singledispatch
def plot_embedding_panel(
    data: pd.DataFrame,
    features: List[str],
    dimred: str = "X_umap",
    components: Tuple[int, int] = (1, 2),
    ncol: int = 3,
    layer: Optional[str] = None,
    size: Optional[float] = None,
    stroke: Optional[float] = None,
    alpha: float = 0.8,
    shared_scale: bool = True,
    cmap: str = "viridis",
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
) -> "Union[Compose, ggplot]":
    """Grid of embedding plots, one panel per feature.

    Each panel is a :func:`ggnomics.plot_reduced_dim` call. For continuous
    features, ``shared_scale=True`` computes a global vmin/vmax up front so
    all panels share the same color scale.

    Args:
        data: DataFrame containing embedding columns (matched against
            ``dimred`` the same way as :func:`ggnomics.plot_embedding`) plus
            one column per requested feature.
        features: Non-empty list of column names, one per panel. Every
            feature must be resolvable; otherwise the complete list of
            missing features is reported in one error.
        dimred: Embedding key (e.g. ``"X_pca"``, ``"X_umap"``).
        components: One-indexed ``(x, y)`` component pair to plot.
        ncol: Number of columns in the panel grid.
        layer: Present for cross-container signature consistency. Has no
            effect for a plain DataFrame, which has no concept of layers.
        size: Point size (``None`` → adaptive).
        stroke: Point stroke width (``None`` → adaptive).
        alpha: Point transparency.
        shared_scale: Compute a single global vmin/vmax across all
            continuous features so panels are color-comparable.
        cmap: Matplotlib colormap for continuous features.
        palette: ``{category: hex}`` mapping for categorical features.
        title: Overall composition title.

    Returns:
        A ``plotnine.composition.Compose`` object, or a bare
        ``plotnine.ggplot`` when only one feature is requested.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        ValueError: If ``features`` is empty, or if any feature cannot be
            resolved (the complete missing list is reported at once).
    """
    raise _unsupported_type("plot_embedding_panel", data)


@plot_embedding_panel.register(pd.DataFrame)
def _plot_embedding_panel_dataframe(
    data: pd.DataFrame,
    features: List[str],
    dimred: str = "X_umap",
    components: Tuple[int, int] = (1, 2),
    ncol: int = 3,
    layer: Optional[str] = None,
    size: Optional[float] = None,
    stroke: Optional[float] = None,
    alpha: float = 0.8,
    shared_scale: bool = True,
    cmap: str = "viridis",
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
):
    from .scatter import plot_reduced_dim
    from ._compose import grid as _grid, annotate_composition

    if not features:
        raise ValueError("`features` must be a non-empty list of column names.")

    missing = [f for f in features if f not in data.columns]
    if missing:
        raise ValueError(
            f"The following features could not be resolved in the DataFrame: "
            f"{missing}. Available (first 20): {list(data.columns)[:20]}"
        )

    global_vmin: Optional[float] = None
    global_vmax: Optional[float] = None
    if shared_scale:
        cont_vals: List[float] = []
        for feat in features:
            if pd.api.types.is_numeric_dtype(data[feat]):
                cont_vals.extend(data[feat].dropna().to_numpy().tolist())
        if cont_vals:
            global_vmin = float(np.min(cont_vals))
            global_vmax = float(np.max(cont_vals))

    plot_list = []
    for feat in features:
        is_cont = pd.api.types.is_numeric_dtype(data[feat])
        vmin = global_vmin if (shared_scale and is_cont) else None
        vmax = global_vmax if (shared_scale and is_cont) else None

        p = plot_reduced_dim(
            data,
            dimred=dimred,
            components=components,
            color=feat,
            layer=None,
            size=size,
            stroke=stroke,
            alpha=alpha,
            cmap=cmap,
            palette=palette if not is_cont else None,
            vmin=vmin,
            vmax=vmax,
            title=feat,
        )
        plot_list.append(p)

    composition = _grid(plot_list, ncol=ncol)
    if title:
        composition = annotate_composition(composition, title=title)
    return composition


__all__ = [
    "plot_violin_stats",
    "plot_box_stats",
    "plot_scatter_marginal",
    "plot_embedding_panel",
]
