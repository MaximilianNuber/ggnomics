"""Tests for plot_abundance."""

import pandas as pd
import pytest
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import plot_abundance


@pytest.mark.parametrize("normalize", [True, False])
def test_abundance_dataframe(mock_df, normalize):
    plot = plot_abundance(
        mock_df,
        group_by="cluster",
        color_by="sample",
        normalize=normalize,
    )
    assert isinstance(plot, ggplot_class)


def test_abundance_title(mock_df):
    plot = plot_abundance(
        mock_df,
        group_by="cluster",
        color_by="batch",
        title="Composition",
    )
    assert isinstance(plot, ggplot_class)


def test_abundance_missing_column(mock_df):
    with pytest.raises(KeyError, match="not_there"):
        plot_abundance(mock_df, group_by="not_there", color_by="sample")


def test_abundance_dataframe_is_registered():
    assert plot_abundance.dispatch(pd.DataFrame) is not plot_abundance.dispatch(object)


def test_abundance_anndata(mock_adata):
    plot = plot_abundance(mock_adata, group_by="cluster", color_by="sample")
    assert isinstance(plot, ggplot_class)


def test_abundance_sce(mock_sce):
    plot = plot_abundance(mock_sce, group_by="cluster", color_by="sample")
    assert isinstance(plot, ggplot_class)


def test_abundance_summarizedexperiment(mock_se):
    plot = plot_abundance(mock_se, group_by="condition", color_by="batch")
    assert isinstance(plot, ggplot_class)
