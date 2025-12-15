# plotting.py

# This module provides base plotting helpers built on plotnine and
# convenience wrappers for BiocPy SingleCellExperiment and SummarizedExperiment.

from typing import Optional, Dict, Sequence

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_point,
    theme_classic,
    scale_color_manual,
    scale_fill_manual,
    ggtitle,
)

# from .utils import PCAResult  # adjust import path if needed

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.decomposition import PCA


@dataclass
class PCAResult:
    scores: np.ndarray                 # samples × components
    loadings: np.ndarray               # features × components
    explained_variance_ratio: np.ndarray
    feature_names: Optional[np.ndarray] = None
    sample_names: Optional[np.ndarray] = None
    model: Optional[PCA] = None

    def scores_to_pandas(self):
        scores_df = pd.DataFrame(self.scores)
        scores_df.columns = ["pc_"+str(i+1) for i in range(scores_df.shape[1])]
        if self.sample_names is not None:
            scores_df.index = self.sample_names
        return scores_df

# ---------------------------------------------------------------------
# Core plotting helper: DataFrame -> ggplot
# ---------------------------------------------------------------------

def plot_reduced_dim(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    fill: Optional[str] = None,
    size: Optional[float] = None,
    stroke: float = 0.0,
    alpha: float = 1.0,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
    autosize_min: float = 0.2,
    autosize_max: float = 1.5,
    autosize_ref: int = 5_000,
):
    """
    Generic 2D embedding plot for a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Contains columns for x, y, and (optionally) color/fill.
    x, y : str
        Column names for the reduced dimensions.
    color : str, optional
        Column name to map to color aesthetic.
    fill : str, optional
        Column name to map to fill aesthetic.
    size : float
        Point size.
    stroke : float
        Point stroke.
    alpha : float
        Point alpha.
    palette : dict, optional
        Mapping {category: color} to use with scale_color_manual/scale_fill_manual.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    aes_kwargs = {"x": x, "y": y}
    if color is not None:
        aes_kwargs["color"] = color
    if fill is not None:
        aes_kwargs["fill"] = fill

    # Auto-size points based on number of rows when size not provided.
    # More points -> smaller size; fewer points -> larger size.
    # Uses a simple scaling: size = autosize_max * sqrt(autosize_ref / n)
    # and clamps to [autosize_min, autosize_max]. For large single-cell datasets,
    # this tends toward ~0.3–0.5, aligning with your preferred default.
    n_points = len(df)
    if size is None:
        if n_points <= 0:
            computed_size = autosize_max
        else:
            computed_size = autosize_max * np.sqrt(autosize_ref / float(n_points))
        size = float(np.clip(computed_size, autosize_min, autosize_max))

    p = (
        ggplot(df)
        + aes(**aes_kwargs)
        + geom_point(size=size, stroke=stroke, alpha=alpha)
        + theme_classic()
    )

    if palette is not None and color is not None:
        p = p + scale_color_manual(
            breaks=list(palette.keys()),
            values=list(palette.values()),
            # guide=None,
        )

    if palette is not None and fill is not None:
        p = p + scale_fill_manual(
            breaks=list(palette.keys()),
            values=list(palette.values()),
            # guide=None,
        )

    if title is not None:
        p = p + ggtitle(title)

    return p


# ---------------------------------------------------------------------
# SummarizedExperiment + PCAResult -> plot_df / plot
# ---------------------------------------------------------------------

def build_plot_df_from_se_and_pca(
    se,
    pca_res: PCAResult,
    n_comps: int = 2,
    prefix: str = "pca",
) -> pd.DataFrame:
    """
    se + PCAResult -> plot_df.

    Columns for reduced dims are named like `"{prefix}_1"`, `"{prefix}_2"`, ...
    and all columns from se.col_data are attached.
    """
    k = min(n_comps, pca_res.scores.shape[1])
    scores = pca_res.scores[:, :k]

    dim_cols = [f"{prefix}_{i+1}" for i in range(k)]
    plot_df = pd.DataFrame(scores, columns=dim_cols)

    meta = se.col_data.to_pandas().copy()
    # keep row alignment
    if getattr(se, "column_names", None) is not None:
        plot_df.index = se.column_names
        meta = meta.reindex(plot_df.index)

    plot_df = pd.concat([plot_df, meta], axis=1)
    return plot_df


def plot_pca_from_se(
    se,
    pca_res: PCAResult,
    x_comp: int = 1,
    y_comp: int = 2,
    prefix: str = "pca",
    color: Optional[str] = None,
    fill: Optional[str] = None,
    palette: Optional[Dict] = None,
    size: Optional[float] = None,
    title: Optional[str] = None,
):
    """
    Convenience wrapper: se + PCAResult -> ggplot via plot_reduced_dim.
    """
    n_comps = max(x_comp, y_comp)
    plot_df = build_plot_df_from_se_and_pca(
        se,
        pca_res,
        n_comps=n_comps,
        prefix=prefix,
    )

    x = f"{prefix}_{x_comp}"
    y = f"{prefix}_{y_comp}"

    return plot_reduced_dim(
        plot_df,
        x=x,
        y=y,
        color=color,
        fill=fill,
        size=size,
        palette=palette,
        title=title,
    )


# ---------------------------------------------------------------------
# SingleCellExperiment -> plot_df / plot
# ---------------------------------------------------------------------

def build_plot_df_from_sce(
    sce,
    dim_name: str,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    """
    sce -> plot_df from a named reduced dimension.

    Columns are named `"{prefix}_1"`, `"{prefix}_2"`, ...
    and sce.col_data is attached.
    """
    # adapt to actual API: maybe sce.reduced_dim(dim_name) or sce.reduced_dims[dim_name]
    # Try common access patterns for reduced dimensions
    if hasattr(sce, "reduced_dim"):
        emb = sce.reduced_dim(dim_name)
    elif hasattr(sce, "reduced_dims") and isinstance(sce.reduced_dims, dict):
        emb = sce.reduced_dims[dim_name]
    else:
        raise AttributeError("SingleCellExperiment object has no reduced_dim or reduced_dims")

    # cells × components
    emb = np.asarray(emb)
    n_comps = emb.shape[1]

    if prefix is None:
        prefix = dim_name.lower()

    dim_cols = [f"{prefix}_{i+1}" for i in range(n_comps)]
    plot_df = pd.DataFrame(emb, columns=dim_cols)

    meta = sce.col_data.to_pandas().copy()
    if getattr(sce, "column_names", None) is not None:
        plot_df.index = sce.column_names
        meta = meta.reindex(plot_df.index)

    plot_df = pd.concat([plot_df, meta], axis=1)
    return plot_df


def plot_reduced_dim_sce(
    sce,
    dim_name: str,
    x_comp: int = 1,
    y_comp: int = 2,
    prefix: Optional[str] = None,
    color: Optional[str] = None,
    fill: Optional[str] = None,
    palette: Optional[Dict] = None,
    size: Optional[float] = None,
    title: Optional[str] = None,
):
    """
    Convenience wrapper: sce + reducedDim -> ggplot.
    """
    plot_df = build_plot_df_from_sce(sce, dim_name=dim_name, prefix=prefix)
    if prefix is None:
        prefix = dim_name.lower()

    x = f"{prefix}_{x_comp}"
    y = f"{prefix}_{y_comp}"

    return plot_reduced_dim(
        plot_df,
        x=x,
        y=y,
        color=color,
        fill=fill,
        size=size,
        palette=palette,
        title=title,
    )


# ---------------------------------------------------------------------
# Additional helpers (stubs) to extend the package
# ---------------------------------------------------------------------

def overlay_gene_expression(
    df: pd.DataFrame,
    x: str,
    y: str,
    expression: str,
    title: Optional[str] = None,
):
    """
    Scatter plot of embedding colored by a gene expression column.
    """
    return plot_reduced_dim(df, x=x, y=y, color=expression, title=title)


# Marker dot-plot moved to gg_singlecell.dotplot module
