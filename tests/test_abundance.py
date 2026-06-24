"""Tests for plot_abundance."""

import pytest
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import plot_abundance


def test_abundance_df_normalized(mock_df):
    p = plot_abundance(mock_df, group_by="cluster", color_by="sample", normalize=True)
    assert isinstance(p, ggplot_class)


def test_abundance_df_counts(mock_df):
    p = plot_abundance(mock_df, group_by="cluster", color_by="sample", normalize=False)
    assert isinstance(p, ggplot_class)


def test_abundance_df_title(mock_df):
    p = plot_abundance(mock_df, group_by="cluster", color_by="batch", title="Composition")
    assert isinstance(p, ggplot_class)


def test_abundance_df_missing_col(mock_df):
    with pytest.raises(KeyError):
        plot_abundance(mock_df, group_by="nonexistent", color_by="sample")


def test_abundance_adata_returns_ggplot(mock_adata):
    p = plot_abundance(mock_adata, group_by="cluster", color_by="sample")
    assert isinstance(p, ggplot_class)


def test_abundance_sce_returns_ggplot(mock_sce):
    p = plot_abundance(mock_sce, group_by="cluster", color_by="sample")
    assert isinstance(p, ggplot_class)
