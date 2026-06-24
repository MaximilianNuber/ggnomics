"""Expression plots: violin, dot plot, and heatmap."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_violin,
    geom_jitter,
    geom_point,
    geom_tile,
    theme_classic,
    theme,
    element_text,
    element_blank,
    ggtitle,
    labs,
    facet_wrap,
    scale_color_cmap,
    scale_fill_cmap,
    scale_size_continuous,
)

from ._accessor import DataAccessor
from ._utils import adaptive_size, color_scale, to_long


def plot_expression(
    data,
    features: List[str],
    group_by: str,
    layer: Optional[str] = None,
    color_by: Optional[str] = None,
    palette: Optional[Dict] = None,
    ncol: Optional[int] = None,
    log1p: bool = False,
    add_points: bool = False,
    point_size: Optional[float] = None,
    title: Optional[str] = None,
) -> ggplot:
    """Violin plot of feature expression across cell groups.

    Each ``feature`` becomes one facet panel; the x-axis shows ``group_by``
    categories and the y-axis shows expression level.

    Args:
        data: ``pd.DataFrame``, ``anndata.AnnData``, or
            ``SingleCellExperiment``.
        features: List of feature/gene names to plot.
        group_by: Obs column whose categories define the x-axis groups.
        layer: Expression layer/assay (``None`` → default X / counts).
        color_by: Obs column to map to fill color.  Defaults to
            ``group_by``.
        palette: ``{category: hex}`` color mapping.
        ncol: Number of columns in ``facet_wrap`` layout.
        log1p: If ``True``, log1p-transform expression values before
            plotting.
        add_points: Overlay jittered points on violins.
        point_size: Size of jittered points (``None`` → adaptive).
        title: Plot title.

    Returns:
        A ``plotnine.ggplot`` object.
    """
    acc = DataAccessor(data)
    obs_df = acc.obs().reset_index(drop=True)
    expr_df = acc.get_expression(features, layer=layer)

    if group_by not in obs_df.columns:
        raise KeyError(f"group_by '{group_by}' not found in obs. Available: {list(obs_df.columns)}")

    fill_col = color_by if color_by is not None else group_by
    if fill_col not in obs_df.columns:
        raise KeyError(f"color_by '{fill_col}' not found in obs.")

    # Build long format
    id_cols = [group_by]
    if fill_col != group_by:
        id_cols.append(fill_col)
    id_cols_unique = list(dict.fromkeys(id_cols))

    wide = pd.concat(
        [obs_df[id_cols_unique].reset_index(drop=True), expr_df.reset_index(drop=True)],
        axis=1,
    )

    long = to_long(wide, id_vars=id_cols_unique, value_vars=features)

    if log1p:
        long["expression"] = np.log1p(long["expression"].values)

    aes_kwargs = {"x": group_by, "y": "expression", "fill": fill_col}

    p = (
        ggplot(long)
        + aes(**aes_kwargs)
        + geom_violin(scale="width", trim=True)
        + facet_wrap("feature", ncol=ncol, scales="free_y")
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
    )

    if add_points:
        ps = point_size if point_size is not None else adaptive_size(len(obs_df), size_max=1.0)
        p = p + geom_jitter(width=0.2, height=0.0, size=ps, alpha=0.4)

    p = p + color_scale(obs_df[fill_col], palette=palette, type_="fill")

    if title is not None:
        p = p + ggtitle(title)

    return p


def plot_dot(
    data,
    features: List[str],
    group_by: str,
    layer: Optional[str] = None,
    scale: bool = True,
    dot_max: float = 1.0,
    dot_min: float = 0.0,
    col_min: float = -2.5,
    col_max: float = 2.5,
    palette: str = "viridis",
    title: Optional[str] = None,
) -> ggplot:
    """Classic dot plot: mean expression (color) and fraction expressing (size).

    Each dot's color encodes the mean expression of that feature in that
    group; dot size encodes the fraction of cells with non-zero expression.

    Args:
        data: ``pd.DataFrame``, ``anndata.AnnData``, or
            ``SingleCellExperiment``.
        features: Feature/gene names to display (y-axis).
        group_by: Obs column for groups (x-axis).
        layer: Expression layer/assay.
        scale: If ``True``, scale mean expression per gene to ``[0, 1]``
            before plotting.
        dot_max: Maximum dot size (mapped to 100 % expressing).
        dot_min: Minimum dot size.
        col_min: Clip scaled expression below this value.
        col_max: Clip scaled expression above this value.
        palette: Matplotlib colormap name for the color scale.
        title: Plot title.

    Returns:
        A ``plotnine.ggplot`` object.
    """
    acc = DataAccessor(data)
    obs_df = acc.obs().reset_index(drop=True)
    expr_df = acc.get_expression(features, layer=layer)

    if group_by not in obs_df.columns:
        raise KeyError(f"group_by '{group_by}' not found in obs.")

    groups = obs_df[group_by].reset_index(drop=True)

    # Compute per-group stats
    records = []
    for feat in features:
        vals = expr_df[feat].values
        series = pd.Series(vals)
        grp_series = groups
        tmp = pd.DataFrame({"val": series, "group": grp_series})
        agg = tmp.groupby("group").agg(
            mean_expr=("val", "mean"),
            frac_expr=("val", lambda v: float((v > 0).mean())),
        ).reset_index()
        agg["feature"] = feat
        records.append(agg)

    stats = pd.concat(records, ignore_index=True)

    if scale:
        # Scale mean_expr per feature to [col_min, col_max].
        # Use transform so the "feature" column is not dropped.
        def _scale_vals(vals: pd.Series) -> pd.Series:
            mn, mx = vals.min(), vals.max()
            rng = mx - mn
            if rng == 0:
                return pd.Series(0.0, index=vals.index)
            return (vals - mn) / rng * (col_max - col_min) + col_min

        stats = stats.copy()
        stats["mean_expr"] = stats.groupby("feature")["mean_expr"].transform(_scale_vals)
        stats["mean_expr"] = stats["mean_expr"].clip(col_min, col_max)

    # Clip fraction
    stats["frac_expr"] = stats["frac_expr"].clip(dot_min, dot_max)

    p = (
        ggplot(stats)
        + aes(x="group", y="feature", size="frac_expr", color="mean_expr")
        + geom_point()
        + scale_color_cmap(cmap_name=palette)
        + scale_size_continuous(range=(1.0, 6.0))
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + labs(size="Fraction\nexpressing", color="Mean\nexpression")
    )

    if title is not None:
        p = p + ggtitle(title)

    return p


def plot_heatmap(
    data,
    features: List[str],
    group_by: Optional[str] = None,
    layer: Optional[str] = None,
    scale: bool = True,
    cluster_rows: bool = True,
    cluster_cols: bool = False,
    palette: str = "RdBu_r",
    title: Optional[str] = None,
    show_colnames: bool = False,
) -> ggplot:
    """Gene-expression heatmap using ``geom_tile``.

    Rows correspond to features/genes and columns correspond to cells or,
    when ``group_by`` is provided, to group-level mean expression.

    Args:
        data: ``pd.DataFrame``, ``anndata.AnnData``, or
            ``SingleCellExperiment``.
        features: Feature/gene names to display.
        group_by: When supplied, average expression per group is plotted
            instead of per-cell values (much faster for large datasets).
        layer: Expression layer/assay.
        scale: Z-score each feature (row) before plotting.
        cluster_rows: Hierarchically cluster rows (features).
        cluster_cols: Hierarchically cluster columns (cells/groups).
        palette: Diverging Matplotlib colormap name (default ``"RdBu_r"``).
        title: Plot title.
        show_colnames: Whether to render column axis text.

    Returns:
        A ``plotnine.ggplot`` object.
    """
    acc = DataAccessor(data)
    obs_df = acc.obs().reset_index(drop=True)
    expr_df = acc.get_expression(features, layer=layer)  # cells × features

    if group_by is not None:
        if group_by not in obs_df.columns:
            raise KeyError(f"group_by '{group_by}' not found in obs.")
        grp = obs_df[group_by].values
        mat_dict = {}
        for feat in features:
            vals = expr_df[feat].values
            tmp = pd.Series(vals)
            grp_s = pd.Series(grp)
            means = tmp.groupby(grp_s).mean()
            mat_dict[feat] = means
        mean_df = pd.DataFrame(mat_dict).T  # features × groups
        mat = mean_df.values
        col_labels = list(mean_df.columns.astype(str))
        row_labels = list(mean_df.index)
    else:
        mat = expr_df.values.T  # features × cells
        col_labels = [str(i) for i in range(mat.shape[1])]
        row_labels = features

    # Z-score per row (feature)
    if scale:
        mu = mat.mean(axis=1, keepdims=True)
        sd = mat.std(axis=1, keepdims=True) + 1e-9
        mat = (mat - mu) / sd

    # Hierarchical clustering
    if cluster_rows and mat.shape[0] > 1:
        row_labels, mat = _hclust_order(mat, labels=row_labels, axis=0)

    if cluster_cols and mat.shape[1] > 1:
        col_labels, mat = _hclust_order(mat, labels=col_labels, axis=1)

    # Long format for plotnine
    df_long = (
        pd.DataFrame(mat, index=row_labels, columns=col_labels)
        .reset_index(names="feature")
        .melt(id_vars="feature", var_name="sample", value_name="value")
    )

    # Preserve ordering via Categorical
    df_long["feature"] = pd.Categorical(df_long["feature"], categories=row_labels, ordered=True)
    df_long["sample"] = pd.Categorical(df_long["sample"], categories=col_labels, ordered=True)

    col_text = element_text(rotation=90, ha="right", size=7) if show_colnames else element_blank()

    p = (
        ggplot(df_long)
        + aes(x="sample", y="feature", fill="value")
        + geom_tile()
        + scale_fill_cmap(cmap_name=palette)
        + theme_classic()
        + theme(
            axis_text_x=col_text,
            axis_text_y=element_text(size=7),
        )
        + labs(x="", y="", fill="Z-score" if scale else "Expression")
    )

    if title is not None:
        p = p + ggtitle(title)

    return p


def _hclust_order(mat: np.ndarray, labels: list, axis: int):
    """Reorder rows or columns of ``mat`` by hierarchical clustering linkage.

    Args:
        mat: 2-D numpy array.
        labels: Labels corresponding to the given axis.
        axis: ``0`` for rows, ``1`` for columns.

    Returns:
        Tuple ``(reordered_labels, reordered_mat)``.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    if axis == 1:
        mat = mat.T

    dist = pdist(mat, metric="euclidean")
    Z = linkage(dist, method="average")
    order = leaves_list(Z)
    reordered_mat = mat[order]
    reordered_labels = [labels[i] for i in order]

    if axis == 1:
        reordered_mat = reordered_mat.T
        return reordered_labels, reordered_mat

    return reordered_labels, reordered_mat
