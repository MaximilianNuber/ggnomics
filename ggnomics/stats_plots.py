"""Statistical plot functions: violin/box with significance bars, marginal scatter, embedding panel."""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

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

from ._accessor import DataAccessor
from ._utils import adaptive_size, HeatmapResult
from .signif._geom import _DeferredSignif, geom_signif


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
    data,
    feature: str,
    group_by: str,
    layer: Optional[str],
    order: Optional[List],
) -> Tuple[pd.DataFrame, List]:
    """Resolve feature and group, return long DataFrame and ordered group list."""
    acc = DataAccessor(data)
    feature_series, _ = acc.resolve_column(feature, layer=layer)
    obs_df = acc.obs().reset_index(drop=True)

    if group_by not in obs_df.columns:
        raise KeyError(
            f"group_by '{group_by}' not found in obs columns. "
            f"Available: {list(obs_df.columns)[:20]}"
        )
    group_series = obs_df[group_by].reset_index(drop=True)

    df = pd.DataFrame({
        "__feature__": feature_series.values,
        "__group__": group_series.values,
    })

    unique_groups = sorted(df["__group__"].unique().astype(str).tolist())
    if order is not None:
        groups_order = [str(g) for g in order if str(g) in unique_groups]
        # Append any not in order list at the end
        groups_order += [g for g in unique_groups if g not in groups_order]
    else:
        groups_order = unique_groups

    df["__group__"] = df["__group__"].astype(str)
    df["__group__"] = pd.Categorical(df["__group__"], categories=groups_order, ordered=True)

    return df, groups_order


# ---------------------------------------------------------------------------
# Public: plot_violin_stats
# ---------------------------------------------------------------------------


def plot_violin_stats(
    data,
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

    Resolves ``feature`` via DataAccessor (obs column or gene expression).
    Runs pairwise tests between groups defined by ``comparisons`` (or all
    pairs when *n_groups* ≤ 5 and ``comparisons`` is None).
    Draws significance brackets as ``geom_segment`` + ``geom_text`` layers
    using proportional spacing (fractions of y-range) so brackets always
    look correct regardless of data scale.
    """
    df, groups_order = _build_violin_data(data, feature, group_by, layer, order)

    p = (
        ggplot(df)
        + aes(x="__group__", y="__feature__", fill="__group__")
        + geom_violin(scale="width", trim=True)
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + labs(
            x=x_label or group_by,
            y=y_label or feature,
            fill=group_by,
        )
    )

    if palette is not None:
        p = p + scale_fill_manual(
            breaks=list(palette.keys()), values=list(palette.values())
        )
    else:
        p = p + scale_fill_brewer(type="qual", palette="Set2")

    if add_boxplot:
        p = p + geom_boxplot(width=0.1, fill="white", outlier_alpha=0.3)

    if add_points:
        ps = point_size or adaptive_size(len(df), size_max=0.8)
        p = p + geom_jitter(width=0.2, height=0.0, size=ps, alpha=0.4)

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
# Public: plot_box_stats
# ---------------------------------------------------------------------------


def plot_box_stats(
    data,
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

    Same signature as :func:`plot_violin_stats` but uses ``geom_boxplot``
    as the base layer. Shares all stat-annotation logic via
    :func:`_add_stat_annotations`.
    """
    df, groups_order = _build_violin_data(data, feature, group_by, layer, order)

    p = (
        ggplot(df)
        + aes(x="__group__", y="__feature__", fill="__group__")
        + geom_boxplot(outlier_alpha=0.5)
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + labs(
            x=x_label or group_by,
            y=y_label or feature,
            fill=group_by,
        )
    )

    if palette is not None:
        p = p + scale_fill_manual(
            breaks=list(palette.keys()), values=list(palette.values())
        )
    else:
        p = p + scale_fill_brewer(type="qual", palette="Set2")

    if add_points:
        ps = point_size or adaptive_size(len(df), size_max=0.8)
        p = p + geom_jitter(width=0.2, height=0.0, size=ps, alpha=0.4)

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
# Public: plot_scatter_marginal
# ---------------------------------------------------------------------------


def plot_scatter_marginal(
    data,
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
    """Scatter plot with marginal distributions.

    Composed from three plotnine panels (scatter, x-marginal, y-marginal)
    using plotnine's native composition system (requires plotnine ≥ 0.15).

    Returns a ``plotnine.composition.Compose`` object.
    Width ratio scatter : right-marginal = 3 : 1;
    height ratio top-marginal : scatter = 1 : 3 (on plotnine ≥ 0.16).
    """
    from plotnine import geom_density, geom_histogram, coord_flip, element_blank
    from plotnine.composition import plot_spacer
    from .scatter import plot_scatter
    from ._compose import HAS_LAYOUT

    acc = DataAccessor(data)
    x_series, _ = acc.resolve_column(x, layer=None)
    y_series, _ = acc.resolve_column(y, layer=None)

    color_series: Optional[pd.Series] = None
    color_is_cont = False
    if color is not None:
        color_series, color_is_cont = acc.resolve_column(color, layer=layer)

    marg_df = pd.DataFrame({"_mx_": x_series.values, "_my_": y_series.values})
    if color_series is not None:
        marg_df["_mc_"] = color_series.values

    fill_col = "_mc_" if color_series is not None else None
    aes_x_kwargs: dict = {"x": "_mx_"}
    aes_y_kwargs: dict = {"x": "_my_"}
    if fill_col and not color_is_cont:
        aes_x_kwargs["fill"] = fill_col
        aes_y_kwargs["fill"] = fill_col

    _strip_top = theme(axis_title_x=element_blank(), legend_position="none")
    _strip_right = theme(axis_title_y=element_blank(), legend_position="none")

    if marginal == "density":
        p_top = (
            ggplot(marg_df)
            + aes(**aes_x_kwargs)
            + geom_density(alpha=0.4)
            + theme_classic()
            + labs(y="Density")
            + _strip_top
        )
        p_right = (
            ggplot(marg_df)
            + aes(**aes_y_kwargs)
            + geom_density(alpha=0.4)
            + coord_flip()
            + theme_classic()
            + labs(x="Density")
            + _strip_right
        )
    elif marginal == "histogram":
        p_top = (
            ggplot(marg_df)
            + aes(**aes_x_kwargs)
            + geom_histogram(bins=30, alpha=0.6)
            + theme_classic()
            + _strip_top
        )
        p_right = (
            ggplot(marg_df)
            + aes(**aes_y_kwargs)
            + geom_histogram(bins=30, alpha=0.6)
            + coord_flip()
            + theme_classic()
            + _strip_right
        )
    else:  # boxplot
        from plotnine import geom_boxplot as _gbp
        p_top = (
            ggplot(marg_df)
            + aes(x="_mx_", y="_mx_")
            + _gbp()
            + theme_classic()
            + _strip_top
        )
        p_right = (
            ggplot(marg_df)
            + aes(x="_my_", y="_my_")
            + _gbp()
            + theme_classic()
            + _strip_right
        )

    p_scatter = plot_scatter(
        data,
        x=x,
        y=y,
        color=color,
        layer=layer,
        size=size,
        stroke=stroke,
        alpha=alpha,
        palette=palette,
        cmap=cmap,
        x_label=x_label,
        y_label=y_label,
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


def plot_embedding_panel(
    data,
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
    """Grid of embedding plots, one per feature.

    Each panel is a ``plot_reduced_dim`` call.  For continuous features,
    ``shared_scale=True`` computes global vmin/vmax before plotting so all
    panels share the same color scale.

    Composed with plotnine's native composition system (requires plotnine
    ≥ 0.15).  Returns a ``plotnine.composition.Compose`` object.
    """
    from .scatter import plot_reduced_dim
    from ._compose import grid as _grid, annotate_composition

    acc = DataAccessor(data)

    # Determine global vmin/vmax for continuous features
    global_vmin: Optional[float] = None
    global_vmax: Optional[float] = None
    if shared_scale:
        cont_vals: List[float] = []
        for feat in features:
            try:
                s, is_cont = acc.resolve_column(feat, layer=layer)
                if is_cont:
                    cont_vals.extend(s.dropna().values.tolist())
            except KeyError:
                pass
        if cont_vals:
            global_vmin = float(np.min(cont_vals))
            global_vmax = float(np.max(cont_vals))

    plot_list = []
    for feat in features:
        try:
            s, is_cont = acc.resolve_column(feat, layer=layer)
        except KeyError:
            is_cont = False

        vmin = global_vmin if (shared_scale and is_cont) else None
        vmax = global_vmax if (shared_scale and is_cont) else None

        p = plot_reduced_dim(
            data,
            dimred=dimred,
            components=components,
            color=feat,
            layer=layer,
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

    if not plot_list:
        raise ValueError("No features could be resolved.")

    composition = _grid(plot_list, ncol=ncol)
    if title:
        composition = annotate_composition(composition, title=title)
    return composition
