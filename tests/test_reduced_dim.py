"""Tests for plot_reduced_dim."""

import pytest
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import plot_reduced_dim
from ggnomics._utils import adaptive_size


# ---------------------------------------------------------------------------
# DataFrame
# ---------------------------------------------------------------------------


def test_reduced_dim_df_returns_ggplot(mock_df):
    p = plot_reduced_dim(mock_df, dimred="UMAP", color="cluster")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_no_color(mock_df):
    p = plot_reduced_dim(mock_df, dimred="UMAP")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_color_numeric(mock_df):
    p = plot_reduced_dim(mock_df, dimred="UMAP", color="n_counts")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_pca(mock_df):
    p = plot_reduced_dim(mock_df, dimred="PCA", components=(1, 2))
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_pca_components(mock_df):
    p = plot_reduced_dim(mock_df, dimred="PCA", components=(2, 3))
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_facet(mock_df):
    p = plot_reduced_dim(mock_df, dimred="UMAP", color="cluster", facet_by="batch")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_order(mock_df):
    clusters = sorted(mock_df["cluster"].unique())
    p = plot_reduced_dim(mock_df, dimred="UMAP", color="cluster", order=clusters)
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_title(mock_df):
    p = plot_reduced_dim(mock_df, dimred="UMAP", title="My UMAP")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_explicit_size(mock_df):
    p = plot_reduced_dim(mock_df, dimred="UMAP", size=2.0)
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_gene_color(mock_df):
    # Gene0001 exists as a column in the DataFrame
    p = plot_reduced_dim(mock_df, dimred="UMAP", color="Gene0001")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_df_missing_embedding(mock_df):
    with pytest.raises(KeyError):
        plot_reduced_dim(mock_df, dimred="TSNE_NONEXISTENT")


def test_reduced_dim_df_missing_color(mock_df):
    with pytest.raises(KeyError):
        plot_reduced_dim(mock_df, dimred="UMAP", color="nonexistent_col")


# ---------------------------------------------------------------------------
# Adaptive sizing
# ---------------------------------------------------------------------------


def test_adaptive_size_decreases():
    # 500 cells vs 50_000 cells — both above the floor but well separated
    size_mid = adaptive_size(500)
    size_huge = adaptive_size(50_000)
    assert size_mid > size_huge


def test_adaptive_size_clipped():
    s = adaptive_size(n=0)
    assert s == 1.5  # defaults to size_max when n==0

    s = adaptive_size(n=10_000_000)
    assert s >= 0.2  # never below size_min


# ---------------------------------------------------------------------------
# AnnData
# ---------------------------------------------------------------------------


def test_reduced_dim_adata_returns_ggplot(mock_adata):
    p = plot_reduced_dim(mock_adata, dimred="X_umap", color="cluster")
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


def test_reduced_dim_adata_missing_embedding(mock_adata):
    with pytest.raises(KeyError):
        plot_reduced_dim(mock_adata, dimred="X_tsne_missing")


# ---------------------------------------------------------------------------
# SCE
# ---------------------------------------------------------------------------


def test_reduced_dim_sce_returns_ggplot(mock_sce):
    p = plot_reduced_dim(mock_sce, dimred="UMAP", color="cluster")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_sce_gene_color(mock_sce):
    p = plot_reduced_dim(mock_sce, dimred="UMAP", color="Gene0001")
    assert isinstance(p, ggplot_class)


def test_reduced_dim_sce_missing_embedding(mock_sce):
    with pytest.raises(KeyError):
        plot_reduced_dim(mock_sce, dimred="TSNE_NONEXISTENT")
