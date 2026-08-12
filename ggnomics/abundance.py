"""Cluster- or category-composition plots."""

from __future__ import annotations

from functools import singledispatch
from typing import Dict, Optional

import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_bar,
    ggplot,
    ggtitle,
    labs,
    position_fill,
    position_stack,
    scale_y_continuous,
    theme,
    theme_classic,
)

from ._utils import color_scale


@singledispatch
def plot_abundance(
    data: pd.DataFrame,
    group_by: str,
    color_by: str,
    normalize: bool = True,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
) -> ggplot:
    """Plot category composition from an observation-metadata DataFrame.

    Args:
        data: DataFrame whose rows are observations and columns are metadata.
        group_by: Column defining the x-axis groups.
        color_by: Column defining the stacked categories and fill colors.
        normalize: Show proportions when true, otherwise raw row counts.
        palette: Optional ``{category: color}`` mapping for ``color_by``.
        title: Optional plot title.

    Returns:
        A ``plotnine.ggplot`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        KeyError: If a requested column is absent.
    """
    raise TypeError(
        f"plot_abundance does not support {type(data).__module__}."
        f"{type(data).__qualname__}. Pass a pandas.DataFrame or install the "
        "optional dependency for a supported genomics container."
    )


@plot_abundance.register(pd.DataFrame)
def _plot_abundance_dataframe(
    data: pd.DataFrame,
    group_by: str,
    color_by: str,
    normalize: bool = True,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
) -> ggplot:
    obs_df = data.copy().reset_index(drop=True)

    missing = [
        column for column in (group_by, color_by) if column not in obs_df.columns
    ]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in the metadata DataFrame. "
            f"Available: {list(obs_df.columns)}"
        )

    counts = (
        obs_df.groupby([group_by, color_by], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )

    if normalize:
        totals = counts.groupby(group_by, observed=True)["n"].transform("sum")
        counts["fraction"] = counts["n"] / totals
        y_column = "fraction"
        y_label = "Fraction"
        position = position_fill()
    else:
        y_column = "n"
        y_label = "Cell count"
        position = position_stack()

    plot = (
        ggplot(counts)
        + aes(x=group_by, y=y_column, fill=color_by)
        + geom_bar(stat="identity", position=position)
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + labs(y=y_label, x=group_by, fill=color_by)
        + color_scale(obs_df[color_by], palette=palette, type_="fill")
    )

    if normalize:
        plot += scale_y_continuous(labels=lambda values: [f"{v:.0%}" for v in values])
    if title is not None:
        plot += ggtitle(title)
    return plot


__all__ = ["plot_abundance"]
