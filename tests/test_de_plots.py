"""Tests for de_plots.py."""

import numpy as np
import pandas as pd
import pytest
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import plot_volcano, plot_ma, plot_coef_lollipop, plot_coef_expression
from ggnomics._utils import HeatmapResult


# ---------------------------------------------------------------------------
# plot_volcano
# ---------------------------------------------------------------------------


def test_volcano_returns_ggplot(mock_de_results):
    p = plot_volcano(mock_de_results, gene_col="gene")
    assert isinstance(p, ggplot_class)


def test_volcano_defaults(mock_de_results):
    p = plot_volcano(mock_de_results)
    assert isinstance(p, ggplot_class)


def test_volcano_label_genes(mock_de_results):
    """Requesting label_genes should add a geom_text layer."""
    p = plot_volcano(mock_de_results, gene_col="gene", label_genes=["Gene0001"])
    assert isinstance(p, ggplot_class)
    text_layers = [l for l in p.layers if "text" in type(l.geom).__name__.lower()]
    assert len(text_layers) > 0


def test_volcano_no_labels(mock_de_results):
    p = plot_volcano(mock_de_results, label_top_n=0)
    assert isinstance(p, ggplot_class)


def test_volcano_xlim_ylim(mock_de_results):
    p = plot_volcano(mock_de_results, xlim=(-3, 3), ylim=(0, 10))
    assert isinstance(p, ggplot_class)


def test_volcano_custom_thresholds(mock_de_results):
    p = plot_volcano(mock_de_results, logfc_threshold=0.5, pval_threshold=0.01)
    assert isinstance(p, ggplot_class)


def test_volcano_title(mock_de_results):
    p = plot_volcano(mock_de_results, title="My Volcano")
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# plot_ma
# ---------------------------------------------------------------------------


def test_ma_returns_ggplot(mock_de_results):
    p = plot_ma(mock_de_results)
    assert isinstance(p, ggplot_class)


def test_ma_with_gene_labels(mock_de_results):
    p = plot_ma(mock_de_results, gene_col="gene", label_top_n=5)
    assert isinstance(p, ggplot_class)


def test_ma_title(mock_de_results):
    p = plot_ma(mock_de_results, title="MA Plot")
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# plot_coef_lollipop
# ---------------------------------------------------------------------------


def test_coef_lollipop_series(mock_coefs):
    coef_series = mock_coefs["coefficient"]
    p = plot_coef_lollipop(coef_series, top_n=10)
    assert isinstance(p, ggplot_class)


def test_coef_lollipop_dataframe(mock_coefs):
    p = plot_coef_lollipop(mock_coefs, top_n=10)
    assert isinstance(p, ggplot_class)


def test_coef_lollipop_top_n(mock_coefs):
    p = plot_coef_lollipop(mock_coefs, top_n=5)
    assert isinstance(p, ggplot_class)
    # Only top-5 features should be in the data layer
    point_layers = [l for l in p.layers if "point" in type(l.geom).__name__.lower()]
    assert len(point_layers) > 0


def test_coef_lollipop_se_col(mock_coefs):
    p = plot_coef_lollipop(mock_coefs, top_n=10, se_col="se")
    assert isinstance(p, ggplot_class)


def test_coef_lollipop_no_zero_line(mock_coefs):
    p = plot_coef_lollipop(mock_coefs, top_n=10, zero_line=False)
    assert isinstance(p, ggplot_class)


def test_coef_lollipop_title(mock_coefs):
    p = plot_coef_lollipop(mock_coefs, top_n=10, title="Coefficients")
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# plot_coef_expression
# ---------------------------------------------------------------------------


def test_coef_expression_dot(mock_adata, mock_coefs):
    coef_series = mock_coefs["coefficient"]
    # Use only features that exist in the AnnData var_names
    from ggnomics._accessor import DataAccessor
    var_names = DataAccessor(mock_adata).var_names()
    coef_filtered = coef_series[coef_series.index.isin(var_names)]
    if coef_filtered.empty:
        pytest.skip("No coefficient features overlap with AnnData var_names")
    p = plot_coef_expression(mock_adata, coef_filtered, group_by="cluster", plot_type="dot", top_n=5)
    assert isinstance(p, ggplot_class)


def test_coef_expression_violin(mock_adata, mock_coefs):
    coef_series = mock_coefs["coefficient"]
    from ggnomics._accessor import DataAccessor
    var_names = DataAccessor(mock_adata).var_names()
    coef_filtered = coef_series[coef_series.index.isin(var_names)]
    if coef_filtered.empty:
        pytest.skip("No coefficient features overlap with AnnData var_names")
    p = plot_coef_expression(mock_adata, coef_filtered, group_by="cluster", plot_type="violin", top_n=3)
    assert isinstance(p, ggplot_class)


def test_coef_expression_heatmap(mock_adata, mock_coefs):
    coef_series = mock_coefs["coefficient"]
    from ggnomics._accessor import DataAccessor
    var_names = DataAccessor(mock_adata).var_names()
    coef_filtered = coef_series[coef_series.index.isin(var_names)]
    if coef_filtered.empty:
        pytest.skip("No coefficient features overlap with AnnData var_names")
    result = plot_coef_expression(mock_adata, coef_filtered, group_by="cluster", plot_type="heatmap", top_n=5)
    assert isinstance(result, HeatmapResult)
    assert isinstance(result.plot, ggplot_class)
