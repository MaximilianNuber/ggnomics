"""Tests for plot_expression, plot_dot, plot_heatmap."""

import pytest
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import plot_expression, plot_dot, plot_heatmap

FEATURES = ["Gene0001", "Gene0002", "Gene0003"]


# ---------------------------------------------------------------------------
# plot_expression — DataFrame
# ---------------------------------------------------------------------------


def test_expression_df_returns_ggplot(mock_df):
    p = plot_expression(mock_df, features=FEATURES, group_by="cluster")
    assert isinstance(p, ggplot_class)


def test_expression_df_log1p(mock_df):
    p = plot_expression(mock_df, features=FEATURES, group_by="cluster", log1p=True)
    assert isinstance(p, ggplot_class)


def test_expression_df_add_points(mock_df):
    p = plot_expression(mock_df, features=FEATURES, group_by="cluster", add_points=True)
    assert isinstance(p, ggplot_class)


def test_expression_df_color_by(mock_df):
    p = plot_expression(mock_df, features=FEATURES, group_by="cluster", color_by="batch")
    assert isinstance(p, ggplot_class)


def test_expression_df_bad_group(mock_df):
    with pytest.raises(KeyError):
        plot_expression(mock_df, features=FEATURES, group_by="nonexistent")


# ---------------------------------------------------------------------------
# plot_expression — AnnData
# ---------------------------------------------------------------------------


def test_expression_adata_returns_ggplot(mock_adata):
    p = plot_expression(mock_adata, features=FEATURES, group_by="cluster")
    assert isinstance(p, ggplot_class)


def test_expression_adata_layer(mock_adata):
    p = plot_expression(mock_adata, features=FEATURES, group_by="cluster", layer="logcounts")
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# plot_expression — SCE
# ---------------------------------------------------------------------------


def test_expression_sce_returns_ggplot(mock_sce):
    p = plot_expression(mock_sce, features=FEATURES, group_by="cluster")
    assert isinstance(p, ggplot_class)


def test_expression_sce_layer(mock_sce):
    p = plot_expression(mock_sce, features=FEATURES, group_by="cluster", layer="logcounts")
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# plot_dot — DataFrame
# ---------------------------------------------------------------------------


def test_dot_df_returns_ggplot(mock_df):
    p = plot_dot(mock_df, features=FEATURES, group_by="cluster")
    assert isinstance(p, ggplot_class)


def test_dot_df_no_scale(mock_df):
    p = plot_dot(mock_df, features=FEATURES, group_by="cluster", scale=False)
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# plot_dot — AnnData / SCE
# ---------------------------------------------------------------------------


def test_dot_adata_returns_ggplot(mock_adata):
    p = plot_dot(mock_adata, features=FEATURES, group_by="cluster")
    assert isinstance(p, ggplot_class)


def test_dot_sce_returns_ggplot(mock_sce):
    p = plot_dot(mock_sce, features=FEATURES, group_by="cluster")
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# plot_heatmap — DataFrame
# ---------------------------------------------------------------------------


def test_heatmap_df_returns_ggplot(mock_df):
    p = plot_heatmap(mock_df, features=FEATURES)
    assert isinstance(p, ggplot_class)


def test_heatmap_df_group_by(mock_df):
    p = plot_heatmap(mock_df, features=FEATURES, group_by="cluster")
    assert isinstance(p, ggplot_class)


def test_heatmap_df_no_cluster(mock_df):
    p = plot_heatmap(mock_df, features=FEATURES, cluster_rows=False, cluster_cols=False)
    assert isinstance(p, ggplot_class)


def test_heatmap_df_no_scale(mock_df):
    p = plot_heatmap(mock_df, features=FEATURES, scale=False)
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# plot_heatmap — AnnData / SCE
# ---------------------------------------------------------------------------


def test_heatmap_adata_returns_ggplot(mock_adata):
    p = plot_heatmap(mock_adata, features=FEATURES, group_by="cluster")
    assert isinstance(p, ggplot_class)


def test_heatmap_sce_returns_ggplot(mock_sce):
    p = plot_heatmap(mock_sce, features=FEATURES, group_by="cluster")
    assert isinstance(p, ggplot_class)
