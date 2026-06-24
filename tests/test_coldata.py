"""Tests for plot_coldata and plot_rowdata."""

import pytest
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import plot_coldata, plot_rowdata


# ---------------------------------------------------------------------------
# plot_coldata — DataFrame
# ---------------------------------------------------------------------------


def test_coldata_scatter_df(mock_df):
    p = plot_coldata(mock_df, x="n_counts", y="n_genes_detected")
    assert isinstance(p, ggplot_class)


def test_coldata_scatter_color(mock_df):
    p = plot_coldata(mock_df, x="n_counts", y="n_genes_detected", color_by="mito_frac")
    assert isinstance(p, ggplot_class)


def test_coldata_violin(mock_df):
    p = plot_coldata(mock_df, x="cluster", y="n_counts", shape="violin")
    assert isinstance(p, ggplot_class)


def test_coldata_box(mock_df):
    p = plot_coldata(mock_df, x="cluster", y="n_counts", shape="box")
    assert isinstance(p, ggplot_class)


def test_coldata_bar(mock_df):
    p = plot_coldata(mock_df, x="cluster", y="n_counts", shape="bar")
    assert isinstance(p, ggplot_class)


def test_coldata_point_strip(mock_df):
    p = plot_coldata(mock_df, x="cluster", y="n_counts", shape="point")
    assert isinstance(p, ggplot_class)


def test_coldata_facet(mock_df):
    p = plot_coldata(mock_df, x="n_counts", y="n_genes_detected", facet_by="batch")
    assert isinstance(p, ggplot_class)


def test_coldata_missing_col(mock_df):
    with pytest.raises(KeyError):
        plot_coldata(mock_df, x="nonexistent", y="n_counts")


# ---------------------------------------------------------------------------
# plot_coldata — AnnData
# ---------------------------------------------------------------------------


def test_coldata_adata_scatter(mock_adata):
    p = plot_coldata(mock_adata, x="n_counts", y="n_genes_detected")
    assert isinstance(p, ggplot_class)


def test_coldata_adata_violin(mock_adata):
    p = plot_coldata(mock_adata, x="cluster", y="n_counts", shape="violin")
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# plot_coldata — SCE
# ---------------------------------------------------------------------------


def test_coldata_sce_scatter(mock_sce):
    p = plot_coldata(mock_sce, x="n_counts", y="n_genes_detected")
    assert isinstance(p, ggplot_class)


def test_coldata_sce_violin(mock_sce):
    p = plot_coldata(mock_sce, x="cluster", y="n_counts", shape="violin")
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# plot_rowdata — AnnData
# ---------------------------------------------------------------------------


def test_rowdata_adata_returns_ggplot(mock_adata):
    p = plot_rowdata(mock_adata, x="mean_expr", y="mean_expr")
    assert isinstance(p, ggplot_class)
