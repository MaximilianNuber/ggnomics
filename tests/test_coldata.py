"""Tests for observation- and feature-metadata plots."""

import pandas as pd
import pytest
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import plot_coldata, plot_rowdata


@pytest.mark.parametrize("shape", ["point", "violin", "box", "bar"])
def test_coldata_categorical_geometries(mock_df, shape):
    plot = plot_coldata(mock_df, x="cluster", y="n_counts", shape=shape)
    assert isinstance(plot, ggplot_class)


def test_coldata_scatter_color_and_facet(mock_df):
    plot = plot_coldata(
        mock_df,
        x="n_counts",
        y="n_genes_detected",
        color_by="mito_frac",
        facet_by="batch",
    )
    assert isinstance(plot, ggplot_class)


def test_coldata_missing_column(mock_df):
    with pytest.raises(KeyError, match="not_there"):
        plot_coldata(mock_df, x="not_there", y="n_counts")


def test_coldata_rejects_unknown_shape(mock_df):
    with pytest.raises(ValueError, match="shape must be"):
        plot_coldata(mock_df, x="cluster", y="n_counts", shape="hex")


def test_coldata_base_docstring_describes_dataframe():
    assert "metadata DataFrame" in plot_coldata.__doc__


def test_coldata_dataframe_is_registered():
    assert plot_coldata.dispatch(pd.DataFrame) is not plot_coldata.dispatch(object)


def test_rowdata_dataframe(mock_adata):
    plot = plot_rowdata(mock_adata.var, x="mean_expr", y="mean_expr")
    assert isinstance(plot, ggplot_class)


def test_coldata_anndata(mock_adata):
    plot = plot_coldata(mock_adata, x="cluster", y="n_counts", shape="violin")
    assert isinstance(plot, ggplot_class)


def test_rowdata_anndata(mock_adata):
    plot = plot_rowdata(mock_adata, x="mean_expr", y="mean_expr")
    assert isinstance(plot, ggplot_class)


def test_coldata_sce(mock_sce):
    plot = plot_coldata(mock_sce, x="cluster", y="n_counts", shape="box")
    assert isinstance(plot, ggplot_class)


def test_rowdata_sce(mock_sce):
    plot = plot_rowdata(mock_sce, x="mean_expr", y="mean_expr")
    assert isinstance(plot, ggplot_class)


def test_coldata_summarizedexperiment(mock_se):
    plot = plot_coldata(
        mock_se,
        x="condition",
        y="library_size",
        shape="violin",
    )
    assert isinstance(plot, ggplot_class)


def test_rowdata_summarizedexperiment(mock_se):
    plot = plot_rowdata(mock_se, x="mean_expr", y="mean_expr")
    assert isinstance(plot, ggplot_class)
