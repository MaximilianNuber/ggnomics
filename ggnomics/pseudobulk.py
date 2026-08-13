"""Pseudobulk QC and DE visualisation."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_bar,
    geom_hline,
    geom_violin,
    geom_boxplot,
    theme_classic,
    theme,
    element_text,
    ggtitle,
    labs,
    position_stack,
    scale_fill_brewer,
    scale_fill_manual,
)

from ._compose import annotate_composition

if TYPE_CHECKING:
    from plotnine.composition import Compose


def _unsupported_type(function_name: str, data: object) -> TypeError:
    return TypeError(
        f"{function_name} does not support {type(data).__module__}."
        f"{type(data).__qualname__}. Pass a pandas.DataFrame or install the "
        "optional dependency for a supported genomics container."
    )


# ---------------------------------------------------------------------------
# Shared data-preparation and plot-construction helpers
#
# These are intentionally small and independently testable, and are reused
# by every container adapter to avoid duplicating plot construction.
# ---------------------------------------------------------------------------


def _sparse_row_sums(matrix) -> np.ndarray:
    """Row-wise sum of a dense array or any SciPy sparse matrix, no densifying."""

    return np.asarray(matrix.sum(axis=1)).reshape(-1)


def _pseudobulk_aggregate(
    matrix,
    sample_ids: np.ndarray,
    unique_samples: np.ndarray,
) -> np.ndarray:
    """Per-sample mean expression (obs x features -> samples x features).

    ``matrix`` may be dense or any SciPy sparse matrix. Aggregation is done
    via a sparse indicator-matrix product, so the complete single-cell
    matrix is never densified — only the small ``(n_samples, n_features)``
    result is.
    """

    sample_index = {sample: i for i, sample in enumerate(unique_samples)}
    n_samples = len(unique_samples)
    n_obs = matrix.shape[0]
    rows = np.fromiter((sample_index[s] for s in sample_ids), dtype=int, count=n_obs)
    cols = np.arange(n_obs)
    counts_per_sample = np.bincount(rows, minlength=n_samples).reshape(-1, 1).astype(float)

    if hasattr(matrix, "tocsr"):
        from scipy import sparse

        indicator = sparse.csr_matrix(
            (np.ones(n_obs), (rows, cols)), shape=(n_samples, n_obs)
        )
        summed = indicator @ matrix
        summed = np.asarray(summed.todense()) if hasattr(summed, "todense") else np.asarray(summed)
    else:
        dense = np.asarray(matrix, dtype=float)
        summed = np.zeros((n_samples, dense.shape[1]))
        for sample, index in sample_index.items():
            summed[index] = dense[sample_ids == sample].sum(axis=0)

    return summed / counts_per_sample


def _sample_condition_map(
    obs_df: pd.DataFrame,
    sample_by: str,
    condition_by: str,
    unique_samples: np.ndarray,
) -> Dict:
    """Map each sample to its single condition value.

    Raises:
        ValueError: If any sample maps to more than one distinct condition.
    """

    condition_map: Dict = {}
    for sample in unique_samples:
        values = obs_df.loc[obs_df[sample_by].to_numpy() == sample, condition_by].unique()
        if len(values) > 1:
            raise ValueError(
                f"Sample {sample!r} maps to multiple {condition_by!r} values: "
                f"{list(values)}. condition_by must be constant within each sample."
            )
        condition_map[sample] = values[0] if len(values) == 1 else None
    return condition_map


def _counts_panel_data(obs_df: pd.DataFrame, sample_by: str, group_by: str) -> pd.DataFrame:
    return (
        obs_df.groupby([sample_by, group_by], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )


def _counts_panel_plot(
    counts_df: pd.DataFrame,
    sample_by: str,
    group_by: str,
    min_cells: int,
    palette: Optional[Dict],
) -> ggplot:
    p = (
        ggplot(counts_df)
        + aes(x=sample_by, y="n_cells", fill=group_by)
        + geom_bar(stat="identity", position=position_stack())
        + geom_hline(yintercept=min_cells, linetype="dashed", alpha=0.7, color="red")
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + labs(x="Sample", y="Cell count", fill=group_by, title="Cells per sample-group")
    )
    if palette is not None:
        p = p + scale_fill_manual(breaks=list(palette.keys()), values=list(palette.values()))
    else:
        p = p + scale_fill_brewer(type="qual", palette="Set2")
    return p


def _libsize_panel_data(
    sample_ids: np.ndarray,
    library_sizes: np.ndarray,
    sample_by: str,
    condition_by: Optional[str],
    condition_values: Optional[np.ndarray],
) -> pd.DataFrame:
    df = pd.DataFrame({
        sample_by: sample_ids,
        "__log_lib__": np.log10(np.maximum(library_sizes, 1)),
    })
    if condition_by is not None:
        df[condition_by] = condition_values
    return df


def _libsize_panel_plot(
    libsize_df: pd.DataFrame,
    sample_by: str,
    condition_by: Optional[str],
    palette: Optional[Dict],
) -> ggplot:
    fill_col = condition_by if condition_by is not None else sample_by
    p = (
        ggplot(libsize_df)
        + aes(x=sample_by, y="__log_lib__", fill=fill_col)
        + geom_violin(scale="width", trim=True, alpha=0.7)
        + geom_boxplot(width=0.1, fill="white", outlier_alpha=0.3)
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + labs(x="Sample", y="log10(total counts)", title="Library size per sample")
    )
    if palette is not None and condition_by is not None:
        p = p + scale_fill_manual(breaks=list(palette.keys()), values=list(palette.values()))
    else:
        p = p + scale_fill_brewer(type="qual", palette="Set2")
    return p


def _pca_panel_data(
    pb_mat: np.ndarray,
    unique_samples: np.ndarray,
    sample_by: str,
    condition_by: Optional[str],
    condition_map: Optional[Dict],
) -> pd.DataFrame:
    n_samples = pb_mat.shape[0]
    if n_samples < 2:
        raise ValueError(
            f"Pseudobulk PCA requires at least 2 samples; got {n_samples}."
        )

    try:
        from sklearn.decomposition import PCA as SkPCA
    except ImportError as exc:
        raise ImportError(
            "Pseudobulk PCA requires scikit-learn. "
            "Install with: pip install 'ggnomics[pseudobulk]'"
        ) from exc

    n_components = min(2, n_samples - 1, pb_mat.shape[1])
    pca = SkPCA(n_components=n_components)
    coords = pca.fit_transform(pb_mat)
    pc2 = coords[:, 1] if coords.shape[1] > 1 else np.zeros(n_samples)

    df = pd.DataFrame({
        "__pc1__": coords[:, 0],
        "__pc2__": pc2,
        sample_by: unique_samples,
    })
    if condition_by is not None and condition_map is not None:
        df[condition_by] = [condition_map[s] for s in unique_samples]
    return df


def _pca_panel_plot(
    pca_df: pd.DataFrame,
    sample_by: str,
    condition_by: Optional[str],
) -> ggplot:
    from .scatter import plot_scatter

    color_col = condition_by if condition_by is not None else sample_by
    return plot_scatter(
        pca_df, x="__pc1__", y="__pc2__", color=color_col,
        x_label="PC1", y_label="PC2", title="Pseudobulk PCA",
    )


# ---------------------------------------------------------------------------
# plot_pseudobulk_qc
# ---------------------------------------------------------------------------


@singledispatch
def plot_pseudobulk_qc(
    data: pd.DataFrame,
    sample_by: str,
    group_by: str,
    condition_by: Optional[str] = None,
    features: Optional[List[str]] = None,
    min_cells: int = 10,
    palette: Optional[Dict] = None,
    ncol: int = 2,
) -> "Compose":
    """Three-panel QC figure for pseudobulk analysis setup.

    Panel 1 -- Cells per sample-group combination (barplot, fill = ``group_by``).
    Panel 2 -- Library size distribution (violin of log10 total counts per sample).
    Panel 3 -- Pseudobulk sample PCA (PC1 vs PC2 scatter colored by ``condition_by``).

    Composed with plotnine's native composition system (requires plotnine
    >= 0.15). Returns a ``plotnine.composition.Compose`` object.

    Args:
        data: DataFrame whose rows are cells. ``sample_by``, ``group_by``,
            and ``condition_by`` are metadata columns; ``features`` are
            expression columns.
        sample_by: Column identifying the pseudobulk sample each cell
            belongs to.
        group_by: Column used to color/stack cell counts within each sample.
        condition_by: Optional column used to color the PCA panel and the
            library-size panel. Must be constant within each sample.
        features: Expression columns used for library size (when no
            ``n_counts`` column is present) and for the pseudobulk PCA
            matrix. Required for the PCA panel; metadata columns are never
            guessed as expression.
        min_cells: Dashed reference line in the cell-count panel.
        palette: ``{category: hex}`` color mapping.
        ncol: Present for cross-container signature consistency (the panel
            layout is fixed at ``(p1 | p2) / p3``).

    Returns:
        A ``plotnine.composition.Compose`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        KeyError: If a requested column is absent.
        ValueError: If library size or the PCA matrix cannot be determined,
            if a sample maps to multiple ``condition_by`` values, or if
            fewer than 2 samples are available for PCA.
        ImportError: If scikit-learn is not installed.
    """
    raise _unsupported_type("plot_pseudobulk_qc", data)


@plot_pseudobulk_qc.register(pd.DataFrame)
def _plot_pseudobulk_qc_dataframe(
    data: pd.DataFrame,
    sample_by: str,
    group_by: str,
    condition_by: Optional[str] = None,
    features: Optional[List[str]] = None,
    min_cells: int = 10,
    palette: Optional[Dict] = None,
    ncol: int = 2,
):
    required = [sample_by, group_by] + ([condition_by] if condition_by is not None else [])
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in the DataFrame. "
            f"Available: {list(data.columns)[:20]}"
        )
    if features is not None:
        missing_f = [f for f in features if f not in data.columns]
        if missing_f:
            raise KeyError(
                f"features not found in the DataFrame: {missing_f}. "
                f"Available (first 20): {list(data.columns)[:20]}"
            )

    obs_df = data.reset_index(drop=True)

    counts_df = _counts_panel_data(obs_df, sample_by, group_by)
    p1 = _counts_panel_plot(counts_df, sample_by, group_by, min_cells, palette)

    if "n_counts" in obs_df.columns:
        library_sizes = obs_df["n_counts"].to_numpy(dtype=float)
    elif features:
        library_sizes = obs_df[features].to_numpy(dtype=float).sum(axis=1)
    else:
        raise ValueError(
            "Cannot determine library size: no 'n_counts' column is present "
            "and no `features` were given. Pass `features=[...]` explicitly."
        )

    condition_values = obs_df[condition_by].to_numpy() if condition_by is not None else None
    libsize_df = _libsize_panel_data(
        obs_df[sample_by].to_numpy(), library_sizes, sample_by, condition_by, condition_values
    )
    p2 = _libsize_panel_plot(libsize_df, sample_by, condition_by, palette)

    if not features:
        raise ValueError(
            "Cannot compute pseudobulk PCA: no `features` were given. Pass "
            "`features=[...]` to select the expression columns to use."
        )

    sample_ids = obs_df[sample_by].to_numpy()
    unique_samples = pd.unique(sample_ids)
    pb_mat = _pseudobulk_aggregate(obs_df[features].to_numpy(dtype=float), sample_ids, unique_samples)
    condition_map = (
        _sample_condition_map(obs_df, sample_by, condition_by, unique_samples)
        if condition_by is not None else None
    )
    pca_df = _pca_panel_data(pb_mat, unique_samples, sample_by, condition_by, condition_map)
    p3 = _pca_panel_plot(pca_df, sample_by, condition_by)

    return (p1 | p2) / p3


# ---------------------------------------------------------------------------
# plot_pseudobulk_de
# ---------------------------------------------------------------------------


def plot_pseudobulk_de(
    results: Dict[str, pd.DataFrame],
    logfc_col: str = "log2FoldChange",
    pval_col: str = "padj",
    gene_col: Optional[str] = None,
    pval_threshold: float = 0.05,
    logfc_threshold: float = 1.0,
    top_n_label: int = 5,
    mode: str = "volcano",
    ncol: int = 3,
    title: Optional[str] = None,
) -> "Union[Compose, ggplot]":
    """Visualise DE results across multiple clusters or contrasts.

    Args:
        results: Mapping of cluster/contrast name to a DESeq2-style DE
            DataFrame. This function is DataFrame/result-table based and
            does not need container dispatch.
        logfc_col: Log2 fold-change column.
        pval_col: Adjusted p-value column used for significance.
        gene_col: Column with gene names (``None`` uses the index).
        pval_threshold: Significance threshold on ``pval_col``.
        logfc_threshold: Absolute log2FC threshold for "significant".
        top_n_label: Number of top genes labeled per volcano panel.
        mode: ``"volcano"`` -- one volcano panel per cluster, arranged in a
            grid (plotnine composition, requires plotnine >= 0.15).
            ``"summary_bar"`` -- barplot of n_up / n_down per cluster.
            ``"upset"`` -- native Plotnine UpSet plot (via
            :mod:`ggnomics.upset`) of shared significant genes across
            clusters/contrasts.
        ncol: Grid columns for ``mode="volcano"``.
        title: Overall title.

    Returns:
        A ``plotnine.composition.Compose`` (``mode="volcano"`` or
        ``mode="upset"``), or a ``plotnine.ggplot`` (``mode="summary_bar"``).

    Raises:
        ValueError: If ``results`` is empty, ``mode`` is unknown, or no
            significant genes remain in any cluster/contrast under the given
            thresholds (``mode="upset"``).
    """
    if mode == "volcano":
        from .de_plots import plot_volcano
        from ._compose import grid as _grid

        plots = []
        for name, df in results.items():
            p = plot_volcano(
                df,
                logfc_col=logfc_col,
                pval_col=pval_col,
                gene_col=gene_col,
                logfc_threshold=logfc_threshold,
                pval_threshold=pval_threshold,
                label_top_n=top_n_label,
                title=name,
            )
            plots.append(p)

        if not plots:
            raise ValueError("results dict is empty.")

        composition = _grid(plots, ncol=ncol)
        if title:
            composition = annotate_composition(composition, title=title)
        return composition

    if mode == "summary_bar":
        rows = []
        for name, df in results.items():
            df2 = df.dropna(subset=[logfc_col, pval_col]).copy()
            sig = df2[pval_col] < pval_threshold
            up = int((sig & (df2[logfc_col] >= logfc_threshold)).sum())
            down = int((sig & (df2[logfc_col] <= -logfc_threshold)).sum())
            rows.extend([
                {"cluster": name, "direction": "up", "n": up},
                {"cluster": name, "direction": "down", "n": -down},
            ])
        bar_df = pd.DataFrame(rows)

        p = (
            ggplot(bar_df)
            + aes(x="cluster", y="n", fill="direction")
            + geom_bar(stat="identity", position="identity", alpha=0.85)
            + geom_hline(yintercept=0, linetype="solid", color="black")
            + scale_fill_manual(values={"up": "#E41A1C", "down": "#377EB8"})
            + theme_classic()
            + theme(axis_text_x=element_text(rotation=45, ha="right"))
            + labs(x="Cluster", y="# Significant genes", fill="Direction")
        )
        if title is not None:
            p = p + ggtitle(title)
        return p

    if mode == "upset":
        from .upset import upset as plot_upset

        sig_genes: Dict[str, set] = {}
        for name, df in results.items():
            df2 = df.dropna(subset=[logfc_col, pval_col]).copy()
            sig = df2[pval_col] < pval_threshold
            sig_de = df2[sig & (df2[logfc_col].abs() >= logfc_threshold)]
            gene_series = sig_de[gene_col] if gene_col and gene_col in sig_de.columns else sig_de.index.astype(str)
            sig_genes[name] = set(gene_series.astype(str).tolist())

        all_genes = sorted(set.union(*sig_genes.values())) if sig_genes else []
        if not all_genes:
            raise ValueError("No significant genes found in any cluster with the given thresholds.")

        contrasts = list(results.keys())
        membership = pd.DataFrame({"gene": all_genes})
        for name in contrasts:
            membership[name] = membership["gene"].isin(sig_genes[name])

        composition = plot_upset(membership, contrasts)
        if title:
            composition = annotate_composition(composition, title=title)
        return composition

    raise ValueError(f"Unknown mode '{mode}'. Choose 'volcano', 'summary_bar', or 'upset'.")


__all__ = ["plot_pseudobulk_qc", "plot_pseudobulk_de"]
