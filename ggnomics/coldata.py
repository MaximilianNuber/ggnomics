"""Plots for observation- and feature-level metadata."""

from __future__ import annotations

from functools import singledispatch
from typing import Dict, Optional

import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_bar,
    geom_boxplot,
    geom_jitter,
    geom_point,
    geom_violin,
    ggplot,
    ggtitle,
    theme,
    theme_classic,
)

from ._utils import adaptive_size, color_scale


def _unsupported_type(function_name: str, data: object) -> TypeError:
    return TypeError(
        f"{function_name} does not support {type(data).__module__}."
        f"{type(data).__qualname__}. Pass a pandas.DataFrame or install the "
        "optional dependency for a supported genomics container."
    )


def _require_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    location: str,
) -> None:
    missing = [column for column in columns if column not in data.columns]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in {location}. "
            f"Available: {list(data.columns)}"
        )


@singledispatch
def plot_coldata(
    data: pd.DataFrame,
    x: str,
    y: str,
    color_by: Optional[str] = None,
    size: Optional[float] = None,
    shape: str = "point",
    palette: Optional[Dict] = None,
    facet_by: Optional[str] = None,
    title: Optional[str] = None,
) -> ggplot:
    """Plot columns from a cell- or sample-metadata DataFrame.

    Both numeric columns produce a scatter plot. With a categorical ``x``,
    ``shape`` selects a point, violin, box, or bar geometry. Container-specific
    implementations convert their column metadata to a DataFrame and then use
    this implementation.

    Args:
        data: DataFrame whose rows are observations and columns are metadata.
        x: Column mapped to the x-axis.
        y: Column mapped to the y-axis.
        color_by: Optional column mapped to color or fill.
        size: Point size. ``None`` chooses a size from the number of rows.
        shape: One of ``"point"``, ``"violin"``, ``"box"``, or ``"bar"``.
        palette: Optional ``{category: color}`` mapping.
        facet_by: Optional column used to facet the plot.
        title: Optional plot title.

    Returns:
        A ``plotnine.ggplot`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        KeyError: If a requested column is absent.
    """
    raise _unsupported_type("plot_coldata", data)


@plot_coldata.register(pd.DataFrame)
def _plot_coldata_dataframe(
    data: pd.DataFrame,
    x: str,
    y: str,
    color_by: Optional[str] = None,
    size: Optional[float] = None,
    shape: str = "point",
    palette: Optional[Dict] = None,
    facet_by: Optional[str] = None,
    title: Optional[str] = None,
) -> ggplot:
    obs_df = data.copy().reset_index(drop=True)
    requested = (x, y) + ((color_by,) if color_by is not None else ())
    requested += ((facet_by,) if facet_by is not None else ())
    _require_columns(obs_df, requested, location="the metadata DataFrame")

    x_numeric = pd.api.types.is_numeric_dtype(obs_df[x])
    y_numeric = pd.api.types.is_numeric_dtype(obs_df[y])

    aes_kwargs: dict = {"x": x, "y": y}
    if color_by is not None:
        aes_kwargs["color"] = color_by

    if x_numeric and y_numeric:
        if size is None:
            size = adaptive_size(len(obs_df))
        plot = (
            ggplot(obs_df)
            + aes(**aes_kwargs)
            + geom_point(size=size, alpha=0.7)
            + theme_classic()
        )
        if color_by is not None:
            plot += color_scale(obs_df[color_by], palette=palette, type_="color")
    else:
        shape = shape.lower()
        if shape not in {"point", "violin", "box", "bar"}:
            raise ValueError(
                "shape must be one of 'point', 'violin', 'box', or 'bar'."
            )

        fill_column = color_by if color_by is not None else x
        fill_aes = {"x": x, "y": y, "fill": fill_column}
        rotated_labels = theme(
            axis_text_x=element_text(rotation=45, ha="right")
        )

        if shape == "violin":
            plot = (
                ggplot(obs_df)
                + aes(**fill_aes)
                + geom_violin(scale="width", trim=True)
                + theme_classic()
                + rotated_labels
            )
        elif shape == "box":
            plot = (
                ggplot(obs_df)
                + aes(**fill_aes)
                + geom_boxplot(outlier_alpha=0.5)
                + theme_classic()
                + rotated_labels
            )
        elif shape == "bar":
            group_columns = [x]
            if color_by is not None and color_by != x:
                group_columns.append(color_by)
            means = (
                obs_df.groupby(group_columns, observed=True, as_index=False)[y]
                .mean()
            )
            bar_aes = {"x": x, "y": y, "fill": fill_column}
            plot = (
                ggplot(means)
                + aes(**bar_aes)
                + geom_bar(stat="identity", alpha=0.85)
                + theme_classic()
                + rotated_labels
            )
        else:
            if size is None:
                size = adaptive_size(len(obs_df), size_max=1.0)
            plot = (
                ggplot(obs_df)
                + aes(**fill_aes)
                + geom_jitter(width=0.2, height=0.0, size=size, alpha=0.6)
                + theme_classic()
                + rotated_labels
            )

        plot += color_scale(obs_df[fill_column], palette=palette, type_="fill")

    if facet_by is not None:
        plot += facet_wrap(facet_by)
    if title is not None:
        plot += ggtitle(title)
    return plot


@singledispatch
def plot_rowdata(
    data: pd.DataFrame,
    x: str,
    y: str,
    color_by: Optional[str] = None,
    size: Optional[float] = None,
    title: Optional[str] = None,
) -> ggplot:
    """Scatter-plot columns from a feature-metadata DataFrame.

    Args:
        data: DataFrame whose rows are features and columns are metadata.
        x: Column mapped to the x-axis.
        y: Column mapped to the y-axis.
        color_by: Optional column mapped to color.
        size: Point size. ``None`` chooses a size from the number of rows.
        title: Optional plot title.

    Returns:
        A ``plotnine.ggplot`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        KeyError: If a requested column is absent.
    """
    raise _unsupported_type("plot_rowdata", data)


@plot_rowdata.register(pd.DataFrame)
def _plot_rowdata_dataframe(
    data: pd.DataFrame,
    x: str,
    y: str,
    color_by: Optional[str] = None,
    size: Optional[float] = None,
    title: Optional[str] = None,
) -> ggplot:
    var_df = data.copy().reset_index(drop=True)
    requested = (x, y) + ((color_by,) if color_by is not None else ())
    _require_columns(var_df, requested, location="the feature-metadata DataFrame")

    if size is None:
        size = adaptive_size(len(var_df))

    aes_kwargs: dict = {"x": x, "y": y}
    if color_by is not None:
        aes_kwargs["color"] = color_by

    plot = (
        ggplot(var_df)
        + aes(**aes_kwargs)
        + geom_point(size=size, alpha=0.7)
        + theme_classic()
    )
    if color_by is not None:
        plot += color_scale(var_df[color_by], palette=None, type_="color")
    if title is not None:
        plot += ggtitle(title)
    return plot


__all__ = ["plot_coldata", "plot_rowdata"]
