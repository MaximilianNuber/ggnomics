"""plot_coldata and plot_rowdata — obs/var metadata scatter / violin / box / bar."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_violin,
    geom_boxplot,
    geom_bar,
    theme_classic,
    theme,
    element_text,
    ggtitle,
    labs,
    facet_wrap,
)

from ._accessor import DataAccessor
from ._utils import adaptive_size, color_scale


def plot_coldata(
    data,
    x: str,
    y: str,
    color_by: Optional[str] = None,
    size: Optional[float] = None,
    shape: str = "point",
    palette: Optional[Dict] = None,
    facet_by: Optional[str] = None,
    title: Optional[str] = None,
) -> ggplot:
    """Plot cell-level (obs / colData) metadata.

    Chooses the geometry based on column dtypes and the ``shape`` argument:

    - Both ``x`` and ``y`` numeric → scatter plot.
    - ``x`` categorical + ``shape="violin"`` → violin plot of ``y`` per ``x``.
    - ``x`` categorical + ``shape="box"``    → box plot of ``y`` per ``x``.
    - ``x`` categorical + ``shape="bar"``    → bar chart of mean ``y`` per ``x``.
    - ``x`` categorical + ``shape="point"``  → jittered strip plot.

    Args:
        data: ``pd.DataFrame``, ``anndata.AnnData``, or
            ``SingleCellExperiment``.
        x: Obs column for the x-axis.
        y: Obs column for the y-axis.
        color_by: Obs column to map to color.
        size: Point size (scatter / strip plots).  ``None`` → adaptive.
        shape: Geometry type — ``"point"``, ``"violin"``, ``"box"``,
            or ``"bar"``.
        palette: ``{category: hex}`` color mapping.
        facet_by: Obs column used as a faceting variable.
        title: Plot title.

    Returns:
        A ``plotnine.ggplot`` object.
    """
    acc = DataAccessor(data)
    obs_df = acc.obs().reset_index(drop=True)

    for col in (x, y):
        if col not in obs_df.columns:
            raise KeyError(f"Column '{col}' not found in obs. Available: {list(obs_df.columns)}")

    if color_by is not None and color_by not in obs_df.columns:
        raise KeyError(f"color_by '{color_by}' not found in obs.")

    x_numeric = pd.api.types.is_numeric_dtype(obs_df[x])
    y_numeric = pd.api.types.is_numeric_dtype(obs_df[y])

    aes_kwargs: dict = {"x": x, "y": y}
    if color_by is not None:
        aes_kwargs["color"] = color_by

    if x_numeric and y_numeric:
        # Scatter plot
        if size is None:
            size = adaptive_size(len(obs_df))
        p = (
            ggplot(obs_df)
            + aes(**aes_kwargs)
            + geom_point(size=size, alpha=0.7)
            + theme_classic()
        )
        if color_by is not None:
            p = p + color_scale(obs_df[color_by], palette=palette, type_="color")
    else:
        # Categorical x
        shape_lc = shape.lower()
        fill_col = color_by if color_by is not None else x
        aes_fill = {**{"x": x, "y": y}, "fill": fill_col}

        if shape_lc == "violin":
            p = (
                ggplot(obs_df)
                + aes(**aes_fill)
                + geom_violin(scale="width", trim=True)
                + theme_classic()
                + theme(axis_text_x=element_text(rotation=45, ha="right"))
            )
        elif shape_lc == "box":
            p = (
                ggplot(obs_df)
                + aes(**aes_fill)
                + geom_boxplot(outlier_alpha=0.5)
                + theme_classic()
                + theme(axis_text_x=element_text(rotation=45, ha="right"))
            )
        elif shape_lc == "bar":
            # Compute mean y per x group
            grp_means = obs_df.groupby(x, as_index=False)[y].mean()
            if fill_col == x:
                aes_bar = {"x": x, "y": y, "fill": fill_col}
                bar_df = grp_means.rename(columns={y: y})  # keep same name
                bar_df[fill_col] = bar_df[x]
            else:
                bar_df = obs_df
                aes_bar = {"x": x, "y": y, "fill": fill_col}
            p = (
                ggplot(grp_means)
                + aes(x=x, y=y)
                + geom_bar(stat="identity", alpha=0.85)
                + theme_classic()
                + theme(axis_text_x=element_text(rotation=45, ha="right"))
            )
        else:
            # strip / jitter
            if size is None:
                size = adaptive_size(len(obs_df), size_max=1.0)
            from plotnine import geom_jitter, position_jitter
            p = (
                ggplot(obs_df)
                + aes(**aes_fill)
                + geom_jitter(width=0.2, height=0.0, size=size, alpha=0.6)
                + theme_classic()
                + theme(axis_text_x=element_text(rotation=45, ha="right"))
            )

        if shape_lc != "bar":
            p = p + color_scale(obs_df[fill_col], palette=palette, type_="fill")

    if facet_by is not None:
        p = p + facet_wrap(facet_by)

    if title is not None:
        p = p + ggtitle(title)

    return p


def plot_rowdata(
    data,
    x: str,
    y: str,
    color_by: Optional[str] = None,
    size: Optional[float] = None,
    title: Optional[str] = None,
) -> ggplot:
    """Scatter plot of feature-level (var / rowData) metadata.

    Args:
        data: ``pd.DataFrame``, ``anndata.AnnData``, or
            ``SingleCellExperiment``.
        x: Var/rowData column for the x-axis.
        y: Var/rowData column for the y-axis.
        color_by: Var column to map to color.
        size: Point size.  ``None`` → adaptive.
        title: Plot title.

    Returns:
        A ``plotnine.ggplot`` object.
    """
    acc = DataAccessor(data)

    # Extract rowData / var
    if acc.object_type() == "dataframe":
        # For DataFrames we treat the DataFrame itself as row metadata
        var_df = acc.obs().reset_index(drop=True)
    elif acc.object_type() == "anndata":
        var_df = acc._data.var.copy().reset_index(drop=True)
    else:
        # SCE
        rd = acc._data.row_data
        if hasattr(rd, "to_pandas"):
            var_df = rd.to_pandas().copy().reset_index(drop=True)
        else:
            var_df = pd.DataFrame(rd).copy().reset_index(drop=True)

    for col in (x, y):
        if col not in var_df.columns:
            raise KeyError(f"Column '{col}' not found in var/rowData. Available: {list(var_df.columns)}")

    n = len(var_df)
    if size is None:
        size = adaptive_size(n)

    aes_kwargs: dict = {"x": x, "y": y}
    if color_by is not None:
        aes_kwargs["color"] = color_by

    p = (
        ggplot(var_df)
        + aes(**aes_kwargs)
        + geom_point(size=size, alpha=0.7)
        + theme_classic()
    )

    if color_by is not None:
        if color_by not in var_df.columns:
            raise KeyError(f"color_by '{color_by}' not found in var.")
        p = p + color_scale(var_df[color_by], palette=None, type_="color")

    if title is not None:
        p = p + ggtitle(title)

    return p
