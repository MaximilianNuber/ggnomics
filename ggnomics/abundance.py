"""plot_abundance — cluster composition stacked bar chart."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_bar,
    position_stack,
    position_fill,
    theme_classic,
    theme,
    element_text,
    ggtitle,
    labs,
    scale_y_continuous,
)

from ._accessor import DataAccessor
from ._utils import color_scale


def plot_abundance(
    data,
    group_by: str,
    color_by: str,
    normalize: bool = True,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
) -> ggplot:
    """Stacked bar chart of cell-type / cluster composition.

    Shows how ``color_by`` categories are distributed within each
    ``group_by`` category.

    Args:
        data: ``pd.DataFrame``, ``anndata.AnnData``, or
            ``SingleCellExperiment``.
        group_by: Obs column defining x-axis groups (e.g. ``"cluster"``).
        color_by: Obs column defining fill colors (e.g. ``"sample"`` or
            ``"condition"``).
        normalize: If ``True``, show proportions (fractions 0–1); otherwise
            show raw cell counts.
        palette: ``{category: hex}`` color mapping for ``color_by``.
        title: Plot title.

    Returns:
        A ``plotnine.ggplot`` object.
    """
    acc = DataAccessor(data)
    obs_df = acc.obs().reset_index(drop=True)

    for col in (group_by, color_by):
        if col not in obs_df.columns:
            raise KeyError(f"Column '{col}' not found in obs. Available: {list(obs_df.columns)}")

    counts = (
        obs_df.groupby([group_by, color_by], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )

    if normalize:
        totals = counts.groupby(group_by, observed=True)["n"].transform("sum")
        counts["fraction"] = counts["n"] / totals
        y_col = "fraction"
        y_label = "Fraction"
    else:
        y_col = "n"
        y_label = "Cell count"

    position = position_fill() if normalize else position_stack()

    p = (
        ggplot(counts)
        + aes(x=group_by, y=y_col, fill=color_by)
        + geom_bar(stat="identity", position=position)
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + labs(y=y_label, x=group_by, fill=color_by)
    )

    if normalize:
        p = p + scale_y_continuous(labels=lambda l: [f"{v:.0%}" for v in l])

    p = p + color_scale(obs_df[color_by], palette=palette, type_="fill")

    if title is not None:
        p = p + ggtitle(title)

    return p
