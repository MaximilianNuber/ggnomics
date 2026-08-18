"""Expression plots: violin, dot plot, and heatmap."""

from __future__ import annotations

from functools import singledispatch
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_blank,
    element_text,
    facet_wrap,
    geom_jitter,
    geom_point,
    geom_tile,
    geom_violin,
    ggplot,
    ggtitle,
    labs,
    scale_color_cmap,
    scale_fill_cmap,
    scale_size_continuous,
    theme,
    theme_classic,
)

from ._utils import adaptive_size, color_scale, to_long


def _unsupported_type(function_name: str, data: object) -> TypeError:
    return TypeError(
        f"{function_name} does not support {type(data).__module__}."
        f"{type(data).__qualname__}. Pass a pandas.DataFrame or install the "
        "optional dependency for a supported genomics container."
    )


def _validate_features(features: List[str]) -> None:
    if not features:
        raise ValueError("`features` must be a non-empty list of feature names.")
    seen = set()
    duplicates = sorted({f for f in features if f in seen or seen.add(f)})
    if duplicates:
        raise ValueError(f"`features` contains duplicate names, which would make the result ambiguous: {duplicates}")


def _validate_no_collision(features: List[str], metadata_names: List[str]) -> None:
    collisions = sorted(set(features) & set(metadata_names))
    if collisions:
        raise ValueError(
            f"Name(s) {collisions} are used both as requested feature(s) and "
            "as metadata column(s), which is ambiguous. Rename one side or "
            "request a disjoint set of names."
        )


def _require_columns(data: pd.DataFrame, columns: List[str], *, location: str) -> None:
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise KeyError(f"Column(s) {missing} not found in {location}. Available: {list(data.columns)}")


# ---------------------------------------------------------------------------
# plot_expression
# ---------------------------------------------------------------------------


@singledispatch
def plot_expression(
    data: pd.DataFrame,
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
    """Violin plot of feature expression across cell/sample groups.

    Each ``feature`` becomes one facet panel; the x-axis shows ``group_by``
    categories and the y-axis shows expression level.

    Args:
        data: DataFrame whose rows are cells/samples. Requested ``features``
            and ``group_by``/``color_by`` are columns.
        features: Non-empty list of feature/gene column names to plot. Must
            not contain duplicates or collide with ``group_by``/``color_by``.
        group_by: Column whose categories define the x-axis groups.
        layer: Present for cross-container signature consistency. Has no
            effect for a plain DataFrame, which has no concept of layers.
        color_by: Column to map to fill color. Defaults to ``group_by``.
        palette: ``{category: hex}`` color mapping.
        ncol: Number of columns in ``facet_wrap`` layout.
        log1p: If ``True``, log1p-transform expression values before
            plotting.
        add_points: Overlay jittered points on violins.
        point_size: Size of jittered points (``None`` → adaptive).
        title: Plot title.

    Returns:
        A ``plotnine.ggplot`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        ValueError: If ``features`` is empty, contains duplicates, or
            collides with ``group_by``/``color_by``.
        KeyError: If a requested column is absent.
    """
    raise _unsupported_type("plot_expression", data)


@plot_expression.register(pd.DataFrame)
def _plot_expression_dataframe(
    data: pd.DataFrame,
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
    _validate_features(features)
    fill_col = color_by if color_by is not None else group_by
    metadata_names = list(dict.fromkeys([group_by, fill_col]))
    _validate_no_collision(features, metadata_names)
    _require_columns(data, metadata_names + features, location="the DataFrame")

    obs_df = data.reset_index(drop=True)
    id_cols = metadata_names

    wide = pd.concat(
        [obs_df[id_cols], obs_df[features]],
        axis=1,
    )
    long = to_long(wide, id_vars=id_cols, value_vars=features)
    long["feature"] = pd.Categorical(long["feature"], categories=features, ordered=True)

    if log1p:
        long["expression"] = np.log1p(long["expression"].to_numpy())

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


# ---------------------------------------------------------------------------
# plot_dot
# ---------------------------------------------------------------------------


@singledispatch
def plot_dot(
    data: pd.DataFrame,
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
        data: DataFrame whose rows are cells/samples. Requested ``features``
            and ``group_by`` are columns.
        features: Non-empty list of feature/gene column names to display
            (y-axis). Must not contain duplicates or collide with
            ``group_by``.
        group_by: Column for groups (x-axis).
        layer: Present for cross-container signature consistency. Has no
            effect for a plain DataFrame, which has no concept of layers.
        scale: If ``True``, scale mean expression per gene to
            ``[col_min, col_max]`` before plotting.
        dot_max: Maximum dot size (mapped to 100% expressing).
        dot_min: Minimum dot size.
        col_min: Clip scaled expression below this value.
        col_max: Clip scaled expression above this value.
        palette: Matplotlib colormap name for the color scale.
        title: Plot title.

    Returns:
        A ``plotnine.ggplot`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        ValueError: If ``features`` is empty, contains duplicates, or
            collides with ``group_by``.
        KeyError: If a requested column is absent.
    """
    raise _unsupported_type("plot_dot", data)


@plot_dot.register(pd.DataFrame)
def _plot_dot_dataframe(
    data: pd.DataFrame,
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
    _validate_features(features)
    _validate_no_collision(features, [group_by])
    _require_columns(data, [group_by] + features, location="the DataFrame")

    obs_df = data.reset_index(drop=True)
    groups = obs_df[group_by]

    records = []
    for feat in features:
        tmp = pd.DataFrame({"val": obs_df[feat].to_numpy(), "group": groups.to_numpy()})
        agg = (
            tmp.groupby("group", observed=True)
            .agg(
                mean_expr=("val", "mean"),
                frac_expr=("val", lambda v: float((v > 0).mean())),
            )
            .reset_index()
        )
        agg["feature"] = feat
        records.append(agg)

    stats = pd.concat(records, ignore_index=True)

    if scale:

        def _scale_vals(vals: pd.Series) -> pd.Series:
            mn, mx = vals.min(), vals.max()
            rng = mx - mn
            if rng == 0:
                return pd.Series(0.0, index=vals.index)
            return (vals - mn) / rng * (col_max - col_min) + col_min

        stats = stats.copy()
        stats["mean_expr"] = stats.groupby("feature")["mean_expr"].transform(_scale_vals)
        stats["mean_expr"] = stats["mean_expr"].clip(col_min, col_max)

    stats["frac_expr"] = stats["frac_expr"].clip(dot_min, dot_max)
    stats["feature"] = pd.Categorical(stats["feature"], categories=features, ordered=True)
    stats["group"] = pd.Categorical(stats["group"], categories=list(dict.fromkeys(groups)), ordered=True)

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


# ---------------------------------------------------------------------------
# plot_heatmap
# ---------------------------------------------------------------------------


@singledispatch
def plot_heatmap(
    data: pd.DataFrame,
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
        data: DataFrame whose rows are cells/samples. Requested ``features``
            and ``group_by`` are columns.
        features: Non-empty list of feature/gene column names to display.
        group_by: When supplied, average expression per group is plotted
            instead of per-cell values (much faster for large datasets).
        layer: Present for cross-container signature consistency. Has no
            effect for a plain DataFrame, which has no concept of layers.
        scale: Z-score each feature (row) before plotting.
        cluster_rows: Hierarchically cluster rows (features). Requires SciPy.
        cluster_cols: Hierarchically cluster columns (cells/groups). Requires
            SciPy.
        palette: Diverging Matplotlib colormap name (default ``"RdBu_r"``).
        title: Plot title.
        show_colnames: Whether to render column axis text.

    Returns:
        A ``plotnine.ggplot`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        ValueError: If ``features`` is empty or contains duplicates.
        KeyError: If a requested column is absent.
        ImportError: If clustering is requested and SciPy is not installed.
    """
    raise _unsupported_type("plot_heatmap", data)


@plot_heatmap.register(pd.DataFrame)
def _plot_heatmap_dataframe(
    data: pd.DataFrame,
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
    _validate_features(features)
    required = features + ([group_by] if group_by is not None else [])
    _require_columns(data, required, location="the DataFrame")
    if group_by is not None:
        _validate_no_collision(features, [group_by])

    obs_df = data.reset_index(drop=True)
    expr_df = obs_df[features]

    if group_by is not None:
        grp = obs_df[group_by]
        mean_df = expr_df.groupby(grp, observed=True).mean().T  # features x groups
        mat = mean_df.to_numpy(dtype=float)
        col_labels = list(mean_df.columns.astype(str))
        row_labels = list(mean_df.index)
    else:
        mat = expr_df.to_numpy(dtype=float).T  # features x cells
        col_labels = [str(i) for i in range(mat.shape[1])]
        row_labels = features

    if scale:
        mu = mat.mean(axis=1, keepdims=True)
        sd = mat.std(axis=1, keepdims=True) + 1e-9
        mat = (mat - mu) / sd

    if cluster_rows and mat.shape[0] > 1:
        row_labels, mat = _hclust_order(mat, labels=row_labels, axis=0)

    if cluster_cols and mat.shape[1] > 1:
        col_labels, mat = _hclust_order(mat, labels=col_labels, axis=1)

    df_long = (
        pd.DataFrame(mat, index=row_labels, columns=col_labels)
        .reset_index(names="feature")
        .melt(id_vars="feature", var_name="sample", value_name="value")
    )

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

    Raises:
        ImportError: If SciPy is not installed.
    """
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist
    except ImportError as exc:
        raise ImportError(
            "Hierarchical clustering in plot_heatmap requires SciPy. "
            "Install with: pip install 'ggnomics[stats]' (or: pip install scipy)"
        ) from exc

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


__all__ = ["plot_expression", "plot_dot", "plot_heatmap"]
