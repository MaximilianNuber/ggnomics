"""Tests for plot_heatmap (extended)."""

import numpy as np
import pytest
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import plot_heatmap
from ggnomics.expression import _hclust_order

FEATURES = ["Gene0001", "Gene0002", "Gene0003", "Gene0004", "Gene0005"]


def test_heatmap_basic(mock_df):
    p = plot_heatmap(mock_df, features=FEATURES)
    assert isinstance(p, ggplot_class)


def test_heatmap_group_by(mock_df):
    p = plot_heatmap(mock_df, features=FEATURES, group_by="cluster")
    assert isinstance(p, ggplot_class)


def test_heatmap_cluster_cols(mock_df):
    p = plot_heatmap(mock_df, features=FEATURES, cluster_cols=True, group_by="cluster")
    assert isinstance(p, ggplot_class)


def test_heatmap_no_scale(mock_df):
    p = plot_heatmap(mock_df, features=FEATURES, scale=False)
    assert isinstance(p, ggplot_class)


def test_heatmap_show_colnames(mock_df):
    p = plot_heatmap(mock_df, features=FEATURES, show_colnames=True, group_by="cluster")
    assert isinstance(p, ggplot_class)


def test_hclust_order_rows():
    mat = np.array([[1, 0], [0, 1], [1, 1]])
    labels, reordered = _hclust_order(mat, labels=["a", "b", "c"], axis=0)
    assert len(labels) == 3
    assert reordered.shape == mat.shape


def test_hclust_order_cols():
    mat = np.array([[1, 0, 2], [0, 1, 0]])
    labels, reordered = _hclust_order(mat, labels=["x", "y", "z"], axis=1)
    assert len(labels) == 3
    assert reordered.shape == mat.shape
