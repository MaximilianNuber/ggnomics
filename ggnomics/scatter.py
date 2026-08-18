"""Plotnine scatter and embedding plots with optional container dispatch."""

from __future__ import annotations

from functools import singledispatch
from typing import Dict, List, Optional, Tuple

import pandas as pd
from plotnine import (
    aes,
    coord_fixed,
    facet_wrap,
    geom_point,
    ggplot,
    ggtitle,
    labs,
    scale_color_brewer,
    scale_color_cmap,
    scale_color_manual,
    theme_classic,
)

from ._utils import adaptive_size, adaptive_stroke


def _strip_x_prefix(key: str) -> str:
    """Remove the leading ``X_`` used by the AnnData embedding convention."""

    return key[2:] if key.lower().startswith("x_") else key


def _embedding_key_candidates(key: str) -> list[str]:
    """Return common AnnData and SingleCellExperiment key variants."""

    stripped = _strip_x_prefix(key)
    candidates: list[str] = []
    for candidate in (
        key,
        f"X_{stripped}",
        stripped,
        stripped.upper(),
        key.upper(),
        key.lower(),
    ):
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _embedding_frame_from_dataframe(data: pd.DataFrame, key: str) -> pd.DataFrame:
    """Find embedding columns in a DataFrame and give them normalized names."""

    columns = list(data.columns)
    stripped = _strip_x_prefix(key).lower()

    def matches(column: str) -> bool:
        lowered = str(column).lower()
        return (
            lowered.startswith(f"{stripped}_")
            or lowered.startswith(f"x_{stripped}_")
            or (lowered.startswith(stripped) and len(lowered) > len(stripped) and lowered[len(stripped)].isdigit())
        )

    matched = sorted(column for column in columns if matches(column))
    if not matched:
        matched = [column for column in columns if str(column).lower() in {stripped, key.lower()}]
    if not matched:
        raise KeyError(f"Embedding {key!r} not found in DataFrame columns. Available columns: {columns[:20]}")

    frame = data[matched].copy().reset_index(drop=True)
    frame.columns = [f"{stripped}_{index + 1}" for index in range(len(matched))]
    return frame


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

    p = ggplot(df) + aes(**aes_kwargs) + geom_point(size=size, stroke=stroke, alpha=alpha) + theme_classic()

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
# Layer 2: public scatter dispatcher and DataFrame implementation
# ---------------------------------------------------------------------------


@singledispatch
def plot_scatter(
    data: pd.DataFrame,
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
    """Plot two columns from a pandas DataFrame.

    This DataFrame interface is the canonical plotting API shown by Jupyter.
    Optional AnnData and SingleCellExperiment implementations extract an
    equivalent DataFrame and then delegate to this implementation.

    Parameters
    ----------
    data:
        DataFrame containing ``x``, ``y``, and any requested color/facet
        columns.
    x, y:
        Columns mapped to the horizontal and vertical axes.
    color:
        Optional column mapped to color. Numeric columns receive a continuous
        scale; other columns receive a discrete scale.
    layer:
        Ignored for DataFrames. Optional container backends use it to select
        an expression layer or assay when ``color`` names a feature.
    """
    raise TypeError(
        f"plot_scatter does not support {type(data).__module__}."
        f"{type(data).__qualname__}. Pass a pandas.DataFrame or install the "
        "optional dependency for the requested genomics container."
    )


@plot_scatter.register(pd.DataFrame)
def _plot_scatter_dataframe(
    data: pd.DataFrame,
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
    del layer  # Layer selection is meaningful only for expression containers.

    for column in (x, y):
        if column not in data.columns:
            raise KeyError(f"Column {column!r} not found in DataFrame. Available: {list(data.columns)[:20]}")

    if color is not None and color not in data.columns:
        raise KeyError(f"Color column {color!r} not found in DataFrame. Available: {list(data.columns)[:20]}")
    if facet_by is not None and facet_by not in data.columns:
        raise KeyError(f"facet_by {facet_by!r} not found in DataFrame. Available: {list(data.columns)[:20]}")

    n = len(data)
    color_is_continuous = bool(color is not None and pd.api.types.is_numeric_dtype(data[color]))

    frame = pd.DataFrame({"_gg_x_": data[x].to_numpy(), "_gg_y_": data[y].to_numpy()})
    if color is not None:
        frame["_gg_color_"] = data[color].to_numpy()
    if facet_by is not None:
        frame["_gg_facet_"] = data[facet_by].to_numpy()

    if size is None:
        size = adaptive_size(n)
    if stroke is None:
        stroke = adaptive_stroke(n)

    return _scatter_ggplot(
        frame,
        x="_gg_x_",
        y="_gg_y_",
        color="_gg_color_" if color is not None else None,
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
        facet_by="_gg_facet_" if facet_by is not None else None,
        order=order,
        aspect_ratio=aspect_ratio,
    )


# ---------------------------------------------------------------------------
# Layer 3: public embedding dispatcher and DataFrame implementation
# ---------------------------------------------------------------------------


@singledispatch
def plot_embedding(
    data: pd.DataFrame,
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
    """Plot embedding columns from a pandas DataFrame.

    DataFrame embedding columns may use names such as ``UMAP1``, ``UMAP2``,
    ``umap_1``, or ``X_umap_1``. ``components`` is one-indexed.

    Optional container backends resolve ``dimred`` from ``AnnData.obsm`` or
    ``SingleCellExperiment.reduced_dimensions`` and delegate here. ``layer``
    is ignored for DataFrames and used by those backends for feature coloring.
    """
    raise TypeError(
        f"plot_embedding does not support {type(data).__module__}."
        f"{type(data).__qualname__}. Pass a pandas.DataFrame or install the "
        "optional dependency for the requested genomics container."
    )


@plot_embedding.register(pd.DataFrame)
def _plot_embedding_dataframe(
    data: pd.DataFrame,
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
    del layer

    emb_df = _embedding_frame_from_dataframe(data, dimred)
    comp_cols = list(emb_df.columns)
    ci, cj = components[0] - 1, components[1] - 1
    if ci < 0 or cj < 0 or ci >= len(comp_cols) or cj >= len(comp_cols):
        raise IndexError(f"components={components} out of range for embedding with {len(comp_cols)} dimensions.")
    x_col = comp_cols[ci]
    y_col = comp_cols[cj]

    # Use safe prefixed names to avoid obs-column collisions
    x_safe = f"__emb_{x_col}__"
    y_safe = f"__emb_{y_col}__"
    emb_2d = emb_df[[x_col, y_col]].rename(columns={x_col: x_safe, y_col: y_safe})

    work_df = pd.concat(
        [emb_2d.reset_index(drop=True), data.reset_index(drop=True)],
        axis=1,
    )

    if color is not None and color not in data.columns:
        raise KeyError(f"Color column {color!r} not found in DataFrame. Available: {list(data.columns)[:20]}")
    if facet_by is not None and facet_by not in data.columns:
        raise KeyError(f"facet_by {facet_by!r} not found in DataFrame. Available: {list(data.columns)[:20]}")

    base_name = _strip_x_prefix(dimred).upper()
    x_label = f"{base_name} {components[0]}"
    y_label = f"{base_name} {components[1]}"

    return plot_scatter(
        work_df,
        x=x_safe,
        y=y_safe,
        color=color,
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


# Compatibility names are true aliases: all share one dispatch registry.
plot_reduced_dim = plot_embedding
dim_plot = plot_embedding


def plot_umap(
    data,
    color: Optional[str] = None,
    **kwargs,
) -> ggplot:
    """Plot the first two dimensions of the ``X_umap`` embedding."""
    return plot_embedding(data, dimred="X_umap", color=color, **kwargs)


def plot_pca(
    data,
    color: Optional[str] = None,
    components: Tuple[int, int] = (1, 2),
    **kwargs,
) -> ggplot:
    """Plot selected dimensions of the ``X_pca`` embedding."""
    return plot_embedding(
        data,
        dimred="X_pca",
        components=components,
        color=color,
        **kwargs,
    )


def plot_tsne(
    data,
    color: Optional[str] = None,
    **kwargs,
) -> ggplot:
    """Plot the first two dimensions of the ``X_tsne`` embedding."""
    return plot_embedding(data, dimred="X_tsne", color=color, **kwargs)


__all__ = [
    "plot_scatter",
    "plot_embedding",
    "plot_reduced_dim",
    "dim_plot",
    "plot_umap",
    "plot_pca",
    "plot_tsne",
]
