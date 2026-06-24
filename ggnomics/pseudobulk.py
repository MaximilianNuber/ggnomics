"""Pseudobulk QC and DE visualisation."""

from __future__ import annotations

from typing import Dict, List, Optional

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
)

from ._accessor import DataAccessor
from ._compose import annotate_composition
from ._utils import HeatmapResult


# ---------------------------------------------------------------------------
# plot_pseudobulk_qc
# ---------------------------------------------------------------------------


def plot_pseudobulk_qc(
    data,
    sample_by: str,
    group_by: str,
    condition_by: Optional[str] = None,
    min_cells: int = 10,
    palette: Optional[Dict] = None,
    ncol: int = 2,
):
    """Three-panel QC figure for pseudobulk analysis setup.

    Panel 1 — Cells per sample-group combination (barplot, fill = *group_by*).
    Panel 2 — Library size distribution (violin of log10 total counts per sample).
    Panel 3 — Pseudobulk sample PCA (PC1 vs PC2 scatter colored by *condition_by*).

    Composed with plotnine's native composition system (requires plotnine ≥ 0.15).
    Returns a ``plotnine.composition.Compose`` object.
    """

    acc = DataAccessor(data)
    obs_df = acc.obs().reset_index(drop=True)

    for col in (sample_by, group_by):
        if col not in obs_df.columns:
            raise KeyError(f"'{col}' not found in obs columns.")
    if condition_by is not None and condition_by not in obs_df.columns:
        raise KeyError(f"condition_by '{condition_by}' not found in obs columns.")

    # ---- Panel 1: cells per sample-group ----
    counts_df = (
        obs_df.groupby([sample_by, group_by], observed=True)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    p1 = (
        ggplot(counts_df)
        + aes(x=sample_by, y="n_cells", fill=group_by)
        + geom_bar(stat="identity", position=position_stack())
        + geom_hline(yintercept=min_cells, linetype="dashed", alpha=0.7, color="red")
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + labs(x="Sample", y="Cell count", fill=group_by, title="Cells per sample-group")
    )
    if palette is not None:
        from plotnine import scale_fill_manual
        p1 = p1 + scale_fill_manual(breaks=list(palette.keys()), values=list(palette.values()))
    else:
        p1 = p1 + scale_fill_brewer(type="qual", palette="Set2")

    # ---- Panel 2: library size distribution ----
    # Total counts per cell (use first assay / X)
    try:
        if acc.object_type() == "anndata":
            import scipy.sparse as sp
            X = data.X
            total_counts = np.asarray(X.sum(axis=1)).ravel() if sp.issparse(X) else X.sum(axis=1)
        elif acc.object_type() in ("sce", "se"):
            mat = data.assay("counts")
            total_counts = np.asarray(mat.sum(axis=0)).ravel()
        else:
            gene_cols = [c for c in obs_df.columns if c not in (sample_by, group_by, condition_by or "")]
            total_counts = obs_df.get("n_counts", obs_df[gene_cols].sum(axis=1) if gene_cols else pd.Series(np.ones(len(obs_df)))).values
    except Exception:
        total_counts = np.ones(len(obs_df))

    libsize_df = pd.DataFrame({
        sample_by: obs_df[sample_by].values,
        "__log_lib__": np.log10(np.maximum(total_counts, 1)),
    })
    if condition_by is not None:
        libsize_df[condition_by] = obs_df[condition_by].values

    fill_col = condition_by if condition_by is not None else sample_by
    p2_aes = aes(x=sample_by, y="__log_lib__", fill=fill_col)
    p2 = (
        ggplot(libsize_df)
        + p2_aes
        + geom_violin(scale="width", trim=True, alpha=0.7)
        + geom_boxplot(width=0.1, fill="white", outlier_alpha=0.3)
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + labs(x="Sample", y="log10(total counts)", title="Library size per sample")
    )
    if palette is not None and condition_by is not None:
        from plotnine import scale_fill_manual
        p2 = p2 + scale_fill_manual(breaks=list(palette.keys()), values=list(palette.values()))
    else:
        p2 = p2 + scale_fill_brewer(type="qual", palette="Set2")

    # ---- Panel 3: pseudobulk PCA ----
    try:
        from sklearn.decomposition import PCA as SkPCA

        if acc.object_type() == "anndata":
            import scipy.sparse as sp
            X_full = data.X
            X_full = np.asarray(X_full.toarray() if sp.issparse(X_full) else X_full)
            samples = obs_df[sample_by].values
        elif acc.object_type() == "dataframe":
            gene_cols = [c for c in obs_df.columns if c.startswith("Gene")]
            X_full = obs_df[gene_cols].values if gene_cols else np.zeros((len(obs_df), 1))
            samples = obs_df[sample_by].values
        else:
            mat = data.assay("counts")
            X_full = np.asarray(mat).T  # cells × genes
            samples = obs_df[sample_by].values

        unique_samples = np.unique(samples)
        pb_rows = []
        for smp in unique_samples:
            mask = samples == smp
            pb_rows.append(X_full[mask].mean(axis=0))
        pb_mat = np.array(pb_rows)  # samples × genes

        pca = SkPCA(n_components=min(2, pb_mat.shape[0] - 1, pb_mat.shape[1]))
        coords = pca.fit_transform(pb_mat)

        pca_df = pd.DataFrame({
            "__pc1__": coords[:, 0],
            "__pc2__": coords[:, 1] if coords.shape[1] > 1 else np.zeros(len(unique_samples)),
            sample_by: unique_samples,
        })
        if condition_by is not None:
            # Get condition for each sample (majority vote)
            cond_map = {}
            for smp in unique_samples:
                mask = obs_df[sample_by].values == smp
                vals = obs_df.loc[mask, condition_by].values
                cond_map[smp] = pd.Series(vals).mode().iloc[0] if len(vals) > 0 else "unknown"
            pca_df[condition_by] = pca_df[sample_by].map(cond_map)

        from .scatter import plot_scatter
        color_col = condition_by if condition_by is not None else sample_by
        p3 = plot_scatter(
            pca_df,
            x="__pc1__",
            y="__pc2__",
            color=color_col,
            x_label="PC1",
            y_label="PC2",
            title="Pseudobulk PCA",
        )
    except Exception:
        # Fallback: empty placeholder
        p3 = (
            ggplot(pd.DataFrame({"x": [0], "y": [0], "label": ["PCA unavailable"]}))
            + aes(x="x", y="y")
            + geom_point()
            + theme_classic()
            + ggtitle("Pseudobulk PCA (unavailable)")
        )

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
):
    """Visualise DE results across multiple clusters or contrasts.

    Parameters
    ----------
    results : dict
        Mapping of cluster name → DE DataFrame (DESeq2-style).
    mode : str
        ``"volcano"`` — one volcano per cluster, arranged in a grid
        (plotnine composition, requires plotnine ≥ 0.15).
        ``"summary_bar"`` — barplot of n_up / n_down per cluster (ggplot).
        ``"upset"`` — UpSet plot of shared significant genes (requires upsetplot).
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

        from plotnine import scale_fill_manual
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
        try:
            from upsetplot import from_memberships, UpSet
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "upsetplot is required for mode='upset'. "
                "Install with: pip install upsetplot>=0.8"
            ) from e

        sig_genes: Dict[str, set] = {}
        for name, df in results.items():
            df2 = df.dropna(subset=[logfc_col, pval_col]).copy()
            sig = df2[pval_col] < pval_threshold
            sig_de = df2[sig & (df2[logfc_col].abs() >= logfc_threshold)]
            gene_series = sig_de[gene_col] if gene_col and gene_col in sig_de.columns else sig_de.index.astype(str)
            sig_genes[name] = set(gene_series.astype(str).tolist())

        all_genes = set.union(*sig_genes.values()) if sig_genes else set()
        if not all_genes:
            raise ValueError("No significant genes found in any cluster with the given thresholds.")

        memberships = []
        for gene in all_genes:
            membership = tuple(k for k, v in sig_genes.items() if gene in v)
            memberships.append(membership)

        # from_memberships returns a non-unique Series; use subset_size="count"
        # so UpSet counts elements per combination rather than summing values.
        upset_data = from_memberships(memberships)
        UpSet(upset_data, subset_size="count").plot()
        if title:
            plt.suptitle(title)
        return plt.gcf()

    raise ValueError(f"Unknown mode '{mode}'. Choose 'volcano', 'summary_bar', or 'upset'.")
