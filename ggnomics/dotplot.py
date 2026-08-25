from __future__ import annotations

from functools import singledispatch
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    geom_point,
    ggplot,
    ggtitle,
    scale_color_gradient,
    scale_size_continuous,
    theme_classic,
)


@singledispatch
def marker_dotplot(data: Any, *args, **kwargs):  # pragma: no cover
    raise TypeError("Unsupported input for marker_dotplot. Pass a tidy DataFrame or use the matrix+groups helper.")


@marker_dotplot.register(pd.DataFrame)
def _(
    df: pd.DataFrame,
    gene: str = "gene",
    group: str = "group",
    mean_col: str = "mean",
    pct_col: str = "pct",
    *,
    size_range: tuple[float, float] = (0.5, 4.0),
    color_low: Optional[str] = None,
    color_high: Optional[str] = None,
    title: Optional[str] = None,
):
    p = (
        ggplot(df)
        + aes(x=group, y=gene, size=pct_col, color=mean_col)
        + geom_point()
        + scale_size_continuous(range=size_range)
        + theme_classic()
    )
    if color_low is not None and color_high is not None:
        p = p + scale_color_gradient(low=color_low, high=color_high)
    if title:
        p = p + ggtitle(title)
    return p


def marker_dotplot_from_matrix(
    expr: pd.DataFrame | np.ndarray,
    groups: pd.Series | np.ndarray,
    genes: Sequence[str],
    *,
    expr_threshold: float = 0.0,
    title: Optional[str] = None,
):
    """
    Compute per-group mean expression and percent expressed for genes, then plot.
    `expr` is cells × genes. `groups` is length=cells.
    """
    if isinstance(expr, pd.DataFrame):
        X = expr.values
        all_genes = list(expr.columns)
    else:
        X = np.asarray(expr)
        all_genes = [f"g{i}" for i in range(X.shape[1])]

    groups = pd.Series(np.asarray(groups), name="group")

    # Indices for selected genes
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    idxs = [gene_to_idx[g] for g in genes if g in gene_to_idx]
    if len(idxs) == 0:
        raise ValueError("None of the requested genes were found.")

    # Build tidy table
    records = []
    gdf = groups.reset_index(drop=True)
    for g, j in zip(genes, idxs):
        vals = X[:, j]
        s = pd.Series(vals)
        tmp = pd.DataFrame({"val": s, "group": gdf})
        agg = tmp.groupby("group").agg(mean=("val", "mean"), pct=("val", lambda v: float((v > expr_threshold).mean())))
        agg = agg.reset_index()
        agg["gene"] = g
        records.append(agg)
    df = pd.concat(records, ignore_index=True)
    return marker_dotplot(df, title=title)


# SCE convenience wrapper. Duck-types on `sce` (via the shared helpers in
# ggnomics.violin) rather than importing SingleCellExperiment, so it needs no
# optional-dependency guard: it only touches the singlecellexperiment package
# if and when a caller actually passes such an object.
def marker_dotplot_sce(
    sce,
    genes: Sequence[str],
    group_col: str,
    *,
    assay: str = "logcounts",
    expr_threshold: float = 0.0,
    title: Optional[str] = None,
):
    from .violin import _get_assay_matrix, _get_col_data, _get_row_names

    mat = _get_assay_matrix(sce, assay)
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    genes_all = list(_get_row_names(sce) or [])
    X = mat.T if mat.shape[0] == len(genes_all) else mat
    meta = _get_col_data(sce)
    return marker_dotplot_from_matrix(X, meta[group_col].values, genes, expr_threshold=expr_threshold, title=title)
