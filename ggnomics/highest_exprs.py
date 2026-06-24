"""plot_highest_exprs — top-N most abundant genes boxplot."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_boxplot,
    coord_flip,
    theme_classic,
    theme,
    element_text,
    ggtitle,
    labs,
)

from ._accessor import DataAccessor
from ._utils import color_scale


def plot_highest_exprs(
    data,
    n: int = 50,
    layer: Optional[str] = None,
    color_cells_by: Optional[str] = None,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
) -> ggplot:
    """Boxplot of the top-``n`` highest-expressed genes.

    For each gene, expression is normalised as a fraction of each cell's
    total library size.  Genes are ranked by their median normalised
    expression across cells and the top ``n`` are shown.  Boxes are ordered
    descending (highest-expressing gene at the top).

    Args:
        data: ``pd.DataFrame``, ``anndata.AnnData``, or
            ``SingleCellExperiment``.
        n: Number of top genes to display.
        layer: Expression layer/assay (``None`` → default).
        color_cells_by: Obs column used to color individual data-point
            boxes.  When supplied, a ``fill`` aesthetic is mapped; the
            distribution per category is shown as a separate box.
        palette: ``{category: hex}`` color mapping for ``color_cells_by``.
        title: Plot title.

    Returns:
        A ``plotnine.ggplot`` object.
    """
    acc = DataAccessor(data)
    var_names = acc.var_names()
    obs_df = acc.obs().reset_index(drop=True)

    # Retrieve full expression matrix for all genes
    all_features = var_names
    expr_df = acc.get_expression(all_features, layer=layer)  # cells × genes

    # For DataFrames, var_names includes metadata columns; keep only numeric
    expr_df = expr_df.select_dtypes(include=[np.number])
    all_features = list(expr_df.columns)
    if not all_features:
        raise ValueError("No numeric feature columns found in the data.")

    # Compute library sizes and normalise
    lib_sizes = expr_df.values.sum(axis=1, keepdims=True).astype(float)
    lib_sizes[lib_sizes == 0] = 1.0  # avoid divide-by-zero
    norm_expr = expr_df.values / lib_sizes  # cells × genes, fraction

    # Rank genes by median normalised expression
    medians = np.median(norm_expr, axis=0)
    top_idx = np.argsort(medians)[::-1][:n]
    top_genes = [all_features[i] for i in top_idx]
    top_mat = norm_expr[:, top_idx]  # cells × top-n

    # Build long format
    long_records = []
    for j, gene in enumerate(top_genes):
        vals = top_mat[:, j]
        rec = pd.DataFrame({"feature": gene, "fraction": vals})
        if color_cells_by is not None and color_cells_by in obs_df.columns:
            rec[color_cells_by] = obs_df[color_cells_by].values
        long_records.append(rec)

    long_df = pd.concat(long_records, ignore_index=True)

    # Order features descending by median (highest at top after coord_flip)
    ordered_genes = top_genes  # already ordered descending
    long_df["feature"] = pd.Categorical(long_df["feature"], categories=ordered_genes[::-1], ordered=True)

    aes_kwargs: dict = {"x": "feature", "y": "fraction"}
    if color_cells_by is not None and color_cells_by in obs_df.columns:
        aes_kwargs["fill"] = color_cells_by

    p = (
        ggplot(long_df)
        + aes(**aes_kwargs)
        + geom_boxplot(outlier_size=0.5, outlier_alpha=0.4)
        + coord_flip()
        + theme_classic()
        + theme(axis_text_y=element_text(size=7))
        + labs(x="", y="Fraction of total counts")
    )

    if color_cells_by is not None and color_cells_by in obs_df.columns:
        p = p + color_scale(obs_df[color_cells_by], palette=palette, type_="fill")

    if title is not None:
        p = p + ggtitle(title)

    return p
