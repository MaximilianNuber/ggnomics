"""plot_highest_exprs — top-N most abundant genes boxplot."""

from __future__ import annotations

import warnings
from functools import singledispatch
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    coord_flip,
    element_text,
    geom_boxplot,
    ggplot,
    ggtitle,
    labs,
    theme,
    theme_classic,
)

from ._utils import add_scale, color_scale, display_categorical


def _unsupported_type(data: object) -> TypeError:
    return TypeError(
        f"plot_highest_exprs does not support {type(data).__module__}."
        f"{type(data).__qualname__}. Pass a pandas.DataFrame or install the "
        "optional dependency for a supported genomics container."
    )


@singledispatch
def plot_highest_exprs(
    data: pd.DataFrame,
    n: int = 50,
    layer: Optional[str] = None,
    features: Optional[List[str]] = None,
    color_cells_by: Optional[str] = None,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
) -> ggplot:
    """Boxplot of the top-``n`` highest-expressed genes.

    For each candidate gene, expression is normalised as a fraction of each
    cell's total library size (the summed expression of every candidate
    gene). Genes are ranked by their median normalised expression across
    cells and the top ``n`` are shown, ordered descending (highest-expressing
    gene at the top after ``coord_flip``).

    Args:
        data: DataFrame whose rows are cells/samples. ``features`` (or, if
            omitted, every numeric column) are treated as candidate genes.
        n: Number of top genes to display. Must be ``>= 1``.
        layer: Present for cross-container signature consistency. Has no
            effect for a plain DataFrame, which has no concept of layers.
        features: Explicit candidate gene columns. Recommended: omitting
            this relies on numeric-column auto-detection, which can
            silently include non-gene numeric metadata (e.g. QC columns).
        color_cells_by: Column used to color/fill individual boxes by
            category. Raises ``KeyError`` if supplied but absent.
        palette: ``{category: hex}`` color mapping for ``color_cells_by``.
        title: Plot title.

    Returns:
        A ``plotnine.ggplot`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        ValueError: If ``n < 1`` or no candidate gene columns are found.
        KeyError: If ``features`` or ``color_cells_by`` reference columns
            absent from ``data``.
    """
    raise _unsupported_type(data)


@plot_highest_exprs.register(pd.DataFrame)
def _plot_highest_exprs_dataframe(
    data: pd.DataFrame,
    n: int = 50,
    layer: Optional[str] = None,
    features: Optional[List[str]] = None,
    color_cells_by: Optional[str] = None,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
) -> ggplot:
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")

    if color_cells_by is not None and color_cells_by not in data.columns:
        raise KeyError(f"color_cells_by {color_cells_by!r} not found in the DataFrame. Available: {list(data.columns)}")

    if features is not None:
        missing = [f for f in features if f not in data.columns]
        if missing:
            raise KeyError(
                f"features not found in the DataFrame: {missing}. Available (first 20): {list(data.columns)[:20]}"
            )
        candidate_df = data[features]
    else:
        warnings.warn(
            "plot_highest_exprs: no explicit `features` given for a DataFrame; "
            "falling back to auto-detected numeric columns as candidate genes. "
            "This can silently include non-gene numeric metadata (e.g. QC "
            "columns). Pass `features=[...]` explicitly for reliable results.",
            UserWarning,
            stacklevel=2,
        )
        candidate_df = data.select_dtypes(include=[np.number])

    all_features = list(candidate_df.columns)
    if not all_features:
        raise ValueError("No candidate gene columns found in the data.")

    obs_df = data.reset_index(drop=True)
    expr = candidate_df.reset_index(drop=True).to_numpy(dtype=float)

    lib_sizes = expr.sum(axis=1, keepdims=True)
    lib_sizes[lib_sizes == 0] = 1.0
    norm_expr = expr / lib_sizes

    medians = np.median(norm_expr, axis=0)
    order = np.argsort(medians)[::-1][:n]
    top_genes = [all_features[i] for i in order]
    top_mat = norm_expr[:, order]

    long_df = _long_fraction_frame(top_mat, top_genes, obs_df, color_cells_by)
    return _build_highest_exprs_plot(long_df, top_genes, color_cells_by, palette, title)


def _long_fraction_frame(
    fraction_matrix: np.ndarray,
    feature_order: List[str],
    obs_df: pd.DataFrame,
    color_cells_by: Optional[str],
) -> pd.DataFrame:
    """Build a long-format ``{feature, fraction, [color_cells_by]}`` frame."""

    records = []
    for j, gene in enumerate(feature_order):
        rec = pd.DataFrame({"feature": gene, "fraction": fraction_matrix[:, j]})
        if color_cells_by is not None:
            rec[color_cells_by] = obs_df[color_cells_by].to_numpy()
        records.append(rec)
    return pd.concat(records, ignore_index=True)


def _build_highest_exprs_plot(
    long_df: pd.DataFrame,
    feature_order: List[str],
    color_cells_by: Optional[str],
    palette: Optional[Dict],
    title: Optional[str],
) -> ggplot:
    """Shared plot construction, reused by every container adapter."""

    long_df = long_df.copy()
    long_df["feature"] = display_categorical(long_df["feature"], feature_order[::-1])

    aes_kwargs: dict = {"x": "feature", "y": "fraction"}
    if color_cells_by is not None:
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

    if color_cells_by is not None:
        p = add_scale(p, color_scale(long_df[color_cells_by], palette=palette, type_="fill"))

    if title is not None:
        p = p + ggtitle(title)

    return p


def _sparse_aware_top_n_medians(
    matrix,
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank columns of an obs-by-feature matrix by exact median fraction.

    ``matrix`` may be dense or any SciPy sparse matrix. Per-cell library size
    (row sum across every column) and the per-column median of the
    library-size-normalised values are computed without densifying the
    complete matrix — only the requested top-``n`` columns are ever
    materialised as dense arrays by the caller.

    Returns:
        Tuple ``(top_indices, top_fraction_matrix)`` where ``top_indices``
        is sorted by descending median (length ``min(n, n_features)``) and
        ``top_fraction_matrix`` has shape ``(n_obs, len(top_indices))``.
    """

    if hasattr(matrix, "tocsc"):
        csc = matrix.tocsc()
        n_obs, n_features = csc.shape
        lib_sizes = np.asarray(csc.sum(axis=1)).reshape(-1)
        lib_sizes_safe = np.where(lib_sizes == 0, 1.0, lib_sizes)

        medians = np.empty(n_features)
        for j in range(n_features):
            start, end = csc.indptr[j], csc.indptr[j + 1]
            rows = csc.indices[start:end]
            vals = np.asarray(csc.data[start:end], dtype=float) / lib_sizes_safe[rows]
            medians[j] = _median_with_implicit_zeros(vals, n_obs)

        order = np.argsort(medians)[::-1][:n]
        # Densify only the selected top-n columns.
        top_sparse = csc[:, order]
        top_dense = np.asarray(top_sparse.toarray(), dtype=float)
        top_fraction = top_dense / lib_sizes_safe.reshape(-1, 1)
        return order, top_fraction

    dense = np.asarray(matrix, dtype=float)
    lib_sizes = dense.sum(axis=1, keepdims=True)
    lib_sizes_safe = np.where(lib_sizes == 0, 1.0, lib_sizes)
    norm = dense / lib_sizes_safe
    medians = np.median(norm, axis=0)
    order = np.argsort(medians)[::-1][:n]
    return order, norm[:, order]


def _median_with_implicit_zeros(nonzero_values: np.ndarray, n_total: int) -> float:
    """Exact median of ``n_total`` non-negative values given only the non-zeros."""

    n_zeros = n_total - len(nonzero_values)
    sorted_vals = np.sort(nonzero_values)

    def _at(index: int) -> float:
        if index < n_zeros:
            return 0.0
        return float(sorted_vals[index - n_zeros])

    if n_total % 2 == 1:
        return _at(n_total // 2)
    return (_at(n_total // 2 - 1) + _at(n_total // 2)) / 2.0


__all__ = ["plot_highest_exprs"]
