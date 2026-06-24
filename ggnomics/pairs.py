"""plot_pairs — scatterplot matrix of embedding components."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_point,
    facet_grid,
    theme_classic,
    theme,
    element_text,
    element_blank,
    element_rect,
    ggtitle,
    labs,
)

from ._accessor import DataAccessor
from ._utils import adaptive_size, color_scale


def plot_pairs(
    data,
    dimred: str = "X_pca",
    n_components: int = 4,
    color_by: Optional[str] = None,
    size: Optional[float] = None,
    alpha: float = 0.6,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
) -> ggplot:
    """Scatterplot matrix of the first ``n_components`` embedding components.

    Each panel ``(row, col)`` shows a scatter of component ``row`` (y-axis)
    against component ``col`` (x-axis).  All panels share the same color
    mapping so they can be read as a pair-wise comparison.

    Diagonal panels (where row == col) show the same component on both axes
    and appear as a perfect 45° line; they serve as dimensional dividers.

    Args:
        data: ``pd.DataFrame``, ``anndata.AnnData``, or
            ``SingleCellExperiment``.
        dimred: Embedding key (e.g. ``"X_pca"``, ``"X_umap"``).
        n_components: Number of components to include (creates an
            ``n_components × n_components`` grid).
        color_by: Obs column to map to point color.
        size: Point size.  ``None`` → adaptive (based on n_cells × n_panels).
        alpha: Point transparency.
        palette: ``{category: hex}`` color mapping.
        title: Plot title (added as a label above the grid).

    Returns:
        A ``plotnine.ggplot`` object.
    """
    acc = DataAccessor(data)
    n = acc.n_obs()
    emb_df = acc.get_embedding(dimred)
    obs_df = acc.obs().reset_index(drop=True)

    # Limit to available components
    max_comps = emb_df.shape[1]
    k = min(n_components, max_comps)
    if k < 2:
        raise ValueError(
            f"Need at least 2 components for a pairs plot; embedding has {max_comps}."
        )

    comp_cols = list(emb_df.columns[:k])
    emb_k = emb_df[comp_cols].reset_index(drop=True)

    if color_by is not None and color_by not in obs_df.columns:
        raise KeyError(f"color_by '{color_by}' not found in obs.")

    # Build long format: one row per (cell, pair)
    records = []
    for i, row_dim in enumerate(comp_cols):
        for j, col_dim in enumerate(comp_cols):
            tmp = pd.DataFrame(
                {
                    "x_val": emb_k[col_dim].values,
                    "y_val": emb_k[row_dim].values,
                    "row_dim": row_dim,
                    "col_dim": col_dim,
                }
            )
            if color_by is not None:
                tmp[color_by] = obs_df[color_by].values
            records.append(tmp)

    long_df = pd.concat(records, ignore_index=True)

    # Fix facet ordering
    long_df["row_dim"] = pd.Categorical(long_df["row_dim"], categories=comp_cols, ordered=True)
    long_df["col_dim"] = pd.Categorical(long_df["col_dim"], categories=comp_cols, ordered=True)

    # Adaptive size (account for n_panels)
    n_panels = k * k
    effective_n = n * n_panels
    if size is None:
        size = adaptive_size(effective_n, size_max=0.8, size_min=0.1)

    aes_kwargs: dict = {"x": "x_val", "y": "y_val"}
    if color_by is not None:
        aes_kwargs["color"] = color_by

    p = (
        ggplot(long_df)
        + aes(**aes_kwargs)
        + geom_point(size=size, alpha=alpha)
        + facet_grid("row_dim ~ col_dim", scales="free")
        + theme_classic()
        + theme(
            axis_text=element_text(size=6),
            strip_text=element_text(size=7),
            axis_title=element_blank(),
        )
    )

    if color_by is not None:
        p = p + color_scale(obs_df[color_by], palette=palette, type_="color")

    if title is not None:
        p = p + ggtitle(title)

    return p
