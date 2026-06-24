"""Scatter-plot hierarchy: _scatter_ggplot → plot_scatter → plot_reduced_dim → plot_umap/pca/tsne."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_point,
    theme_classic,
    ggtitle,
    labs,
    facet_wrap,
    scale_color_manual,
    scale_color_brewer,
    scale_color_cmap,
    coord_fixed,
)

from ._accessor import DataAccessor, _strip_x_prefix
from ._utils import adaptive_size, adaptive_stroke


# ---------------------------------------------------------------------------
# Layer 1: private plotnine builder
# ---------------------------------------------------------------------------


def _scatter_ggplot(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    color_is_continuous: bool = False,
    size: float = 1.0,
    stroke: float = 0.1,
    alpha: float = 0.8,
    palette: Optional[Dict] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    color_label: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    facet_by: Optional[str] = None,
    order: Optional[List] = None,
    aspect_ratio: Optional[float] = None,
) -> ggplot:
    """Pure plotnine scatter plot — no data extraction, df must already contain all columns.

    If *order* is not None and color is categorical, rows belonging to listed
    categories are moved to the end so they are drawn on top (Seurat style).
    """
    df = df.copy()

    # Reorder for draw-order (categorical color only)
    if order is not None and color is not None and not color_is_continuous:
        ordered_vals = [v for v in order if v in df[color].values]
        rest = df[~df[color].isin(ordered_vals)]
        pieces = [rest] + [df[df[color] == v] for v in ordered_vals]
        df = pd.concat(pieces, ignore_index=True)

    aes_kwargs: dict = {"x": x, "y": y}
    if color is not None:
        aes_kwargs["color"] = color

    p = (
        ggplot(df)
        + aes(**aes_kwargs)
        + geom_point(size=size, stroke=stroke, alpha=alpha)
        + theme_classic()
    )

    # Color scale
    if color is not None:
        if color_is_continuous:
            lim = [vmin, vmax] if (vmin is not None or vmax is not None) else None
            p = p + scale_color_cmap(cmap_name=cmap, limits=lim)
        else:
            if palette is not None:
                p = p + scale_color_manual(
                    breaks=list(palette.keys()),
                    values=list(palette.values()),
                )
            else:
                p = p + scale_color_brewer(type="qual", palette="Set2")

    # Facets
    if facet_by is not None:
        p = p + facet_wrap(facet_by)

    # Labels
    labs_kw: dict = {
        "x": x_label if x_label is not None else x,
        "y": y_label if y_label is not None else y,
    }
    if color is not None and color_label is not None:
        labs_kw["color"] = color_label
    p = p + labs(**labs_kw)

    # Aspect ratio
    if aspect_ratio is not None:
        p = p + coord_fixed(ratio=aspect_ratio)

    if title is not None:
        p = p + ggtitle(title)

    return p


# ---------------------------------------------------------------------------
# Layer 2: public DataAccessor-aware function
# ---------------------------------------------------------------------------


def plot_scatter(
    data,
    x: str,
    y: str,
    color: Optional[str] = None,
    size: Optional[float] = None,
    stroke: Optional[float] = None,
    alpha: float = 0.8,
    palette: Optional[Dict] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    layer: Optional[str] = None,
    facet_by: Optional[str] = None,
    order: Optional[List] = None,
    color_label: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    aspect_ratio: Optional[float] = None,
) -> ggplot:
    """General scatter plot for any two variables in a data object.

    ``x`` and ``y`` are resolved in priority order:
    obs / colData columns → feature names (expression fetched) →
    DataFrame columns (for plain DataFrames).

    ``color`` follows the same resolution; categorical obs columns use a
    discrete palette while numeric / expression columns use a continuous cmap.
    """
    acc = DataAccessor(data)
    n = acc.n_obs()

    x_series, _ = acc.resolve_column(x, layer=None)
    y_series, _ = acc.resolve_column(y, layer=None)

    color_series: Optional[pd.Series] = None
    color_is_continuous = False
    if color is not None:
        color_series, color_is_continuous = acc.resolve_column(color, layer=layer)

    facet_series: Optional[pd.Series] = None
    if facet_by is not None:
        obs_df = acc.obs()
        if facet_by not in obs_df.columns:
            raise KeyError(
                f"facet_by '{facet_by}' not found in obs columns. "
                f"Available: {list(obs_df.columns)[:20]}"
            )
        facet_series = obs_df[facet_by].reset_index(drop=True)

    # Assemble flat DataFrame with safe internal names
    df = pd.DataFrame({"_gg_x_": x_series.values, "_gg_y_": y_series.values})
    if color_series is not None:
        df["_gg_color_"] = color_series.values
    if facet_series is not None:
        df["_gg_facet_"] = facet_series.values

    if size is None:
        size = adaptive_size(n)
    if stroke is None:
        stroke = adaptive_stroke(n)

    return _scatter_ggplot(
        df,
        x="_gg_x_",
        y="_gg_y_",
        color="_gg_color_" if color_series is not None else None,
        color_is_continuous=color_is_continuous,
        size=size,
        stroke=stroke,
        alpha=alpha,
        palette=palette,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        color_label=color_label if color_label is not None else color,
        x_label=x_label if x_label is not None else x,
        y_label=y_label if y_label is not None else y,
        title=title,
        facet_by="_gg_facet_" if facet_series is not None else None,
        order=order,
        aspect_ratio=aspect_ratio,
    )


# ---------------------------------------------------------------------------
# Layer 3: embedding convenience wrapper
# ---------------------------------------------------------------------------


def plot_reduced_dim(
    data,
    dimred: str = "X_umap",
    components: Tuple[int, int] = (1, 2),
    color: Optional[str] = None,
    layer: Optional[str] = None,
    size: Optional[float] = None,
    stroke: Optional[float] = None,
    alpha: float = 0.8,
    palette: Optional[Dict] = None,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    facet_by: Optional[str] = None,
    order: Optional[List] = None,
    title: Optional[str] = None,
    color_label: Optional[str] = None,
) -> ggplot:
    """Plot a 2-D embedding stored in obsm / reducedDims.

    ``dimred`` is resolved case-insensitively and with or without the ``X_``
    prefix.  ``components=(1, 2)`` selects the 1st and 2nd dimensions
    (1-indexed).  The x and y axis labels default to e.g. ``"UMAP 1"`` /
    ``"UMAP 2"``.

    Calls ``plot_scatter`` internally after extracting the embedding columns
    and, if needed, pre-fetching gene expression for *color*.
    """
    acc = DataAccessor(data)

    emb_df = acc.get_embedding(dimred)
    comp_cols = list(emb_df.columns)
    ci, cj = components[0] - 1, components[1] - 1
    if ci >= len(comp_cols) or cj >= len(comp_cols):
        raise IndexError(
            f"components={components} out of range for embedding with "
            f"{len(comp_cols)} dimensions."
        )
    x_col = comp_cols[ci]
    y_col = comp_cols[cj]

    # Use safe prefixed names to avoid obs-column collisions
    x_safe = f"__emb_{x_col}__"
    y_safe = f"__emb_{y_col}__"
    emb_2d = emb_df[[x_col, y_col]].rename(columns={x_col: x_safe, y_col: y_safe})

    obs_df = acc.obs().reset_index(drop=True)
    work_df = pd.concat([emb_2d.reset_index(drop=True), obs_df], axis=1)

    # Pre-fetch gene expression for color if needed
    color_col = color
    if color is not None and color not in obs_df.columns:
        if color in acc.var_names():
            expr = acc.get_expression([color], layer=layer)
            work_df["__color_expr__"] = expr[color].values
            color_col = "__color_expr__"
        else:
            raise KeyError(
                f"'{color}' not found in obs columns or feature names. "
                f"Obs columns: {list(obs_df.columns)[:10]}. "
                f"Feature names (first 10): {acc.var_names()[:10]}"
            )

    base_name = _strip_x_prefix(dimred).upper()
    x_label = f"{base_name} {components[0]}"
    y_label = f"{base_name} {components[1]}"

    return plot_scatter(
        work_df,
        x=x_safe,
        y=y_safe,
        color=color_col,
        size=size,
        stroke=stroke,
        alpha=alpha,
        palette=palette,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        layer=None,
        facet_by=facet_by,
        order=order,
        color_label=color_label if color_label is not None else color,
        x_label=x_label,
        y_label=y_label,
        title=title,
    )


def plot_umap(
    data,
    color: Optional[str] = None,
    **kwargs,
) -> ggplot:
    """Convenience wrapper: ``plot_reduced_dim`` with ``dimred='X_umap'``."""
    return plot_reduced_dim(data, dimred="X_umap", color=color, **kwargs)


def plot_pca(
    data,
    color: Optional[str] = None,
    components: Tuple[int, int] = (1, 2),
    **kwargs,
) -> ggplot:
    """Convenience wrapper: ``plot_reduced_dim`` with ``dimred='X_pca'``."""
    return plot_reduced_dim(data, dimred="X_pca", components=components, color=color, **kwargs)


def plot_tsne(
    data,
    color: Optional[str] = None,
    **kwargs,
) -> ggplot:
    """Convenience wrapper: ``plot_reduced_dim`` with ``dimred='X_tsne'``."""
    return plot_reduced_dim(data, dimred="X_tsne", color=color, **kwargs)
