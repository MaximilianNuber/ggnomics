"""Tests for the scatter plot hierarchy (scatter.py)."""

import pandas as pd
import pytest
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import plot_scatter, plot_reduced_dim, plot_umap, plot_pca, plot_tsne
from ggnomics._utils import adaptive_size


# ---------------------------------------------------------------------------
# plot_scatter — DataFrame
# ---------------------------------------------------------------------------


def test_scatter_df_basic(mock_df):
    p = plot_scatter(mock_df, x="UMAP1", y="UMAP2")
    assert isinstance(p, ggplot_class)


def test_scatter_df_color_categorical(mock_df):
    p = plot_scatter(mock_df, x="UMAP1", y="UMAP2", color="cluster")
    assert isinstance(p, ggplot_class)


def test_scatter_df_color_numeric(mock_df):
    p = plot_scatter(mock_df, x="n_counts", y="n_genes_detected", color="mito_frac")
    assert isinstance(p, ggplot_class)


def test_scatter_df_obs_cols(mock_df):
    p = plot_scatter(mock_df, x="n_counts", y="n_genes_detected", color="cluster")
    assert isinstance(p, ggplot_class)


def test_scatter_df_facet(mock_df):
    p = plot_scatter(mock_df, x="UMAP1", y="UMAP2", color="cluster", facet_by="batch")
    assert isinstance(p, ggplot_class)
    assert p.facet is not None


def test_scatter_df_order(mock_df):
    clusters = sorted(mock_df["cluster"].unique().tolist())
    p = plot_scatter(mock_df, x="UMAP1", y="UMAP2", color="cluster", order=clusters)
    assert isinstance(p, ggplot_class)


def test_scatter_df_title(mock_df):
    p = plot_scatter(mock_df, x="UMAP1", y="UMAP2", title="My Scatter")
    assert isinstance(p, ggplot_class)


def test_scatter_df_unknown_x_raises(mock_df):
    with pytest.raises(KeyError, match="not found"):
        plot_scatter(mock_df, x="nonexistent_col", y="UMAP2")


def test_scatter_df_unknown_color_raises(mock_df):
    with pytest.raises(KeyError, match="not found"):
        plot_scatter(mock_df, x="UMAP1", y="UMAP2", color="nonexistent_col")


# ---------------------------------------------------------------------------
# plot_scatter — AnnData (obs columns + gene expression)
# ---------------------------------------------------------------------------


def test_scatter_adata_obs_cols(mock_adata):
    p = plot_scatter(mock_adata, x="n_counts", y="n_genes_detected", color="cluster")
    assert isinstance(p, ggplot_class)


def test_scatter_adata_gene_x(mock_adata):
    """x resolved as gene expression (var_name)."""
    p = plot_scatter(mock_adata, x="Gene0001", y="n_counts")
    assert isinstance(p, ggplot_class)


def test_scatter_adata_gene_color(mock_adata):
    p = plot_scatter(mock_adata, x="n_counts", y="n_genes_detected", color="Gene0001")
    assert isinstance(p, ggplot_class)


def test_scatter_adata_gene_color_layer(mock_adata):
    p = plot_scatter(
        mock_adata, x="n_counts", y="n_genes_detected",
        color="Gene0001", layer="logcounts"
    )
    assert isinstance(p, ggplot_class)


def test_scatter_adata_facet(mock_adata):
    p = plot_scatter(mock_adata, x="n_counts", y="n_genes_detected",
                     color="cluster", facet_by="batch")
    assert isinstance(p, ggplot_class)
    assert p.facet is not None


# ---------------------------------------------------------------------------
# plot_scatter — SCE
# ---------------------------------------------------------------------------


def test_scatter_sce(mock_sce):
    p = plot_scatter(mock_sce, x="n_counts", y="n_genes_detected", color="cluster")
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# Adaptive sizing
# ---------------------------------------------------------------------------


def test_adaptive_size_larger_for_small_n():
    # 50 cells: clips to size_max=1.5; 50000 cells: well below the ref → shrinks
    size_small = adaptive_size(50)
    size_huge = adaptive_size(50_000)
    assert size_small > size_huge


def test_adaptive_size_clamp():
    assert adaptive_size(0) == 1.5
    assert adaptive_size(10_000_000) >= 0.2


# ---------------------------------------------------------------------------
# plot_reduced_dim — DataFrame
# ---------------------------------------------------------------------------


def test_reduced_dim_df_umap(mock_df):
    p = plot_reduced_dim(mock_df, dimred="UMAP", color="cluster")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_pca(mock_df):
    p = plot_reduced_dim(mock_df, dimred="PCA", components=(1, 2))
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_no_color(mock_df):
    p = plot_reduced_dim(mock_df, dimred="UMAP")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_gene_color(mock_df):
    p = plot_reduced_dim(mock_df, dimred="UMAP", color="Gene0001")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_facet(mock_df):
    p = plot_reduced_dim(mock_df, dimred="UMAP", color="cluster", facet_by="batch")
    assert isinstance(p, ggplot_class)
    assert p.facet is not None


def test_reduced_dim_df_order(mock_df):
    clusters = sorted(mock_df["cluster"].unique().tolist())
    p = plot_reduced_dim(mock_df, dimred="UMAP", color="cluster", order=clusters)
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_missing_embedding_raises(mock_df):
    with pytest.raises(KeyError):
        plot_reduced_dim(mock_df, dimred="TSNE_NONEXISTENT")


def test_reduced_dim_df_missing_color_raises(mock_df):
    with pytest.raises(KeyError):
        plot_reduced_dim(mock_df, dimred="UMAP", color="nonexistent_col")


# ---------------------------------------------------------------------------
# plot_reduced_dim — AnnData
# ---------------------------------------------------------------------------


def test_reduced_dim_adata_umap(mock_adata):
    p = plot_reduced_dim(mock_adata, dimred="X_umap", color="cluster")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_adata_umap_prefix_insensitive(mock_adata):
    p = plot_reduced_dim(mock_adata, dimred="umap", color="cluster")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_adata_umap_case_insensitive(mock_adata):
    p = plot_reduced_dim(mock_adata, dimred="X_UMAP", color="cluster")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_adata_pca(mock_adata):
    p = plot_reduced_dim(mock_adata, dimred="X_pca", components=(1, 2))
    assert isinstance(p, ggplot_class)


def test_reduced_dim_adata_gene_color(mock_adata):
    p = plot_reduced_dim(mock_adata, dimred="X_umap", color="Gene0001")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_adata_gene_color_layer(mock_adata):
    p = plot_reduced_dim(mock_adata, dimred="X_umap", color="Gene0001", layer="logcounts")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_adata_missing_dimred_raises(mock_adata):
    with pytest.raises(KeyError):
        plot_reduced_dim(mock_adata, dimred="X_tsne_missing")


# ---------------------------------------------------------------------------
# plot_reduced_dim — SCE
# ---------------------------------------------------------------------------


def test_reduced_dim_sce_umap(mock_sce):
    p = plot_reduced_dim(mock_sce, dimred="UMAP", color="cluster")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_sce_gene_color(mock_sce):
    p = plot_reduced_dim(mock_sce, dimred="UMAP", color="Gene0001")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_sce_missing_dimred_raises(mock_sce):
    with pytest.raises(KeyError):
        plot_reduced_dim(mock_sce, dimred="TSNE_NONEXISTENT")


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------


def test_plot_umap(mock_adata):
    p = plot_umap(mock_adata, color="cluster")
    assert isinstance(p, ggplot_class)


def test_plot_pca(mock_adata):
    p = plot_pca(mock_adata, color="cluster", components=(1, 2))
    assert isinstance(p, ggplot_class)


def test_plot_tsne_missing_raises(mock_adata):
    with pytest.raises(KeyError):
        plot_tsne(mock_adata, color="cluster")


# ---------------------------------------------------------------------------
# Adaptive sizing: size should be smaller for larger n
# ---------------------------------------------------------------------------


def test_scatter_adaptive_size_large_vs_small():
    from ggnomics._utils import adaptive_size

    # 50-cell dataset clips to size_max; 50k-cell dataset shrinks below it
    assert adaptive_size(50) > adaptive_size(50_000)
