"""Tests for plot_highest_exprs."""

import pytest
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import plot_highest_exprs


def test_highest_exprs_df_returns_ggplot(mock_df):
    p = plot_highest_exprs(mock_df, n=10)
    assert isinstance(p, ggplot_class)


def test_highest_exprs_df_color_cells(mock_df):
    p = plot_highest_exprs(mock_df, n=10, color_cells_by="cluster")
    assert isinstance(p, ggplot_class)


def test_highest_exprs_df_n50(mock_df):
    p = plot_highest_exprs(mock_df, n=50)
    assert isinstance(p, ggplot_class)


def test_highest_exprs_adata_returns_ggplot(mock_adata):
    p = plot_highest_exprs(mock_adata, n=10)
    assert isinstance(p, ggplot_class)


def test_highest_exprs_adata_layer(mock_adata):
    p = plot_highest_exprs(mock_adata, n=10, layer="logcounts")
    assert isinstance(p, ggplot_class)


def test_highest_exprs_sce_returns_ggplot(mock_sce):
    p = plot_highest_exprs(mock_sce, n=10)
    assert isinstance(p, ggplot_class)
