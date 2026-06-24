"""Tests for plot_pairs."""

import pytest
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import plot_pairs


def test_pairs_df_returns_ggplot(mock_df):
    p = plot_pairs(mock_df, dimred="PCA", n_components=3)
    assert isinstance(p, ggplot_class)


def test_pairs_df_color(mock_df):
    p = plot_pairs(mock_df, dimred="PCA", n_components=3, color_by="cluster")
    assert isinstance(p, ggplot_class)


def test_pairs_df_title(mock_df):
    p = plot_pairs(mock_df, dimred="PCA", n_components=2, title="Pairs plot")
    assert isinstance(p, ggplot_class)


def test_pairs_df_2_components(mock_df):
    p = plot_pairs(mock_df, dimred="UMAP", n_components=2)
    assert isinstance(p, ggplot_class)


def test_pairs_df_bad_color(mock_df):
    with pytest.raises(KeyError):
        plot_pairs(mock_df, dimred="PCA", n_components=2, color_by="nonexistent")


def test_pairs_adata_returns_ggplot(mock_adata):
    p = plot_pairs(mock_adata, dimred="X_pca", n_components=3)
    assert isinstance(p, ggplot_class)


def test_pairs_sce_returns_ggplot(mock_sce):
    p = plot_pairs(mock_sce, dimred="PCA", n_components=3)
    assert isinstance(p, ggplot_class)
