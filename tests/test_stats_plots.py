"""Tests for stats_plots.py."""

import numpy as np
import pandas as pd
import pytest
from plotnine.ggplot import ggplot as ggplot_class
from plotnine.composition import Compose

from ggnomics import plot_violin_stats, plot_box_stats, plot_scatter_marginal, plot_embedding_panel


FEATURES = ["Gene0001", "Gene0002"]


# ---------------------------------------------------------------------------
# plot_violin_stats — DataFrame
# ---------------------------------------------------------------------------


def test_violin_stats_df_returns_ggplot(mock_df):
    p = plot_violin_stats(mock_df, feature="Gene0001", group_by="cluster")
    assert isinstance(p, ggplot_class)


def test_violin_stats_df_no_boxplot(mock_df):
    p = plot_violin_stats(mock_df, feature="Gene0001", group_by="cluster", add_boxplot=False)
    assert isinstance(p, ggplot_class)


def test_violin_stats_df_add_points(mock_df):
    p = plot_violin_stats(mock_df, feature="Gene0001", group_by="cluster", add_points=True)
    assert isinstance(p, ggplot_class)


def test_violin_stats_df_single_comparison(mock_df):
    p = plot_violin_stats(
        mock_df, feature="Gene0001", group_by="cluster",
        comparisons=[("C0", "C1")],
    )
    assert isinstance(p, ggplot_class)


def test_violin_stats_df_sig_only_no_bars(mock_df):
    """With sig_only=True and a single non-significant pair, no bar layers added."""
    n = 100
    df = pd.DataFrame({
        "feat": np.ones(n),
        "group": ["A"] * 50 + ["B"] * 50,
    })
    p = plot_violin_stats(df, feature="feat", group_by="group",
                          comparisons=[("A", "B")], sig_only=True)
    assert isinstance(p, ggplot_class)
    seg_layers = [l for l in p.layers if "segment" in type(l.geom).__name__.lower()]
    assert len(seg_layers) == 0


def test_violin_stats_annotation_pvalue(mock_df):
    p = plot_violin_stats(
        mock_df, feature="Gene0001", group_by="cluster",
        comparisons=[("C0", "C1")],
        sig_only=False,
        annotation="pvalue",
    )
    assert isinstance(p, ggplot_class)
    text_layers = [l for l in p.layers if "text" in type(l.geom).__name__.lower()]
    if text_layers:
        for l in text_layers:
            layer_data = getattr(l, "_data", getattr(l, "data", None))
            if layer_data is not None and hasattr(layer_data, "columns") and "label" in layer_data.columns:
                labels = layer_data["label"].values
                assert any("p=" in str(lbl) for lbl in labels)


def test_violin_stats_annotation_stars(mock_df):
    p = plot_violin_stats(
        mock_df, feature="Gene0001", group_by="cluster",
        comparisons=[("C0", "C1")],
        sig_only=False,
        annotation="stars",
    )
    assert isinstance(p, ggplot_class)


def test_violin_stats_annotation_padj(mock_df):
    p = plot_violin_stats(
        mock_df, feature="Gene0001", group_by="cluster",
        comparisons=[("C0", "C1")],
        sig_only=False,
        annotation="padj",
    )
    assert isinstance(p, ggplot_class)


def test_violin_stats_adata_gene(mock_adata):
    p = plot_violin_stats(mock_adata, feature="Gene0001", group_by="cluster")
    assert isinstance(p, ggplot_class)


def test_violin_stats_order(mock_df):
    p = plot_violin_stats(
        mock_df, feature="Gene0001", group_by="cluster",
        order=["C4", "C3", "C2", "C1", "C0"],
    )
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# plot_box_stats
# ---------------------------------------------------------------------------


def test_box_stats_df_returns_ggplot(mock_df):
    p = plot_box_stats(mock_df, feature="Gene0001", group_by="cluster")
    assert isinstance(p, ggplot_class)


def test_box_stats_has_boxplot_layer(mock_df):
    p = plot_box_stats(mock_df, feature="Gene0001", group_by="cluster")
    geom_names = [type(l.geom).__name__.lower() for l in p.layers]
    assert any("boxplot" in g for g in geom_names)


def test_box_stats_adata_gene(mock_adata):
    p = plot_box_stats(mock_adata, feature="Gene0001", group_by="cluster")
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# plot_scatter_marginal — now returns Compose (plotnine native composition)
# ---------------------------------------------------------------------------


def test_scatter_marginal_density_returns_compose(mock_df):
    result = plot_scatter_marginal(mock_df, x="UMAP1", y="UMAP2", marginal="density")
    assert isinstance(result, Compose)


def test_scatter_marginal_histogram_returns_compose(mock_df):
    result = plot_scatter_marginal(mock_df, x="UMAP1", y="UMAP2", marginal="histogram")
    assert isinstance(result, Compose)


def test_scatter_marginal_with_color(mock_df):
    result = plot_scatter_marginal(mock_df, x="UMAP1", y="UMAP2", color="cluster")
    assert isinstance(result, Compose)


def test_scatter_marginal_adata(mock_adata):
    result = plot_scatter_marginal(mock_adata, x="n_counts", y="n_genes_detected")
    assert isinstance(result, Compose)


def test_scatter_marginal_with_title(mock_df):
    result = plot_scatter_marginal(mock_df, x="UMAP1", y="UMAP2", title="QC scatter")
    assert isinstance(result, Compose)


# ---------------------------------------------------------------------------
# plot_embedding_panel — now returns Compose
# ---------------------------------------------------------------------------


def test_embedding_panel_basic(mock_adata):
    result = plot_embedding_panel(mock_adata, features=["Gene0001", "Gene0002", "Gene0003"], ncol=2)
    assert isinstance(result, Compose)


def test_embedding_panel_shared_scale(mock_adata):
    result = plot_embedding_panel(
        mock_adata, features=["Gene0001", "Gene0002"],
        ncol=2, shared_scale=True,
    )
    assert isinstance(result, Compose)


def test_embedding_panel_no_shared_scale(mock_adata):
    result = plot_embedding_panel(
        mock_adata, features=["Gene0001", "Gene0002"],
        ncol=2, shared_scale=False,
    )
    assert isinstance(result, Compose)


def test_embedding_panel_with_title(mock_adata):
    result = plot_embedding_panel(
        mock_adata, features=["Gene0001", "Gene0002"],
        ncol=2, title="Expression Panel",
    )
    assert isinstance(result, Compose)


def test_embedding_panel_ncol1(mock_adata):
    result = plot_embedding_panel(mock_adata, features=["Gene0001", "Gene0002"], ncol=1)
    assert isinstance(result, Compose)


def test_embedding_panel_categorical_feature(mock_adata):
    """Categorical obs column as feature uses discrete palette, not cmap.
    Single feature → grid returns the bare ggplot (not a Compose)."""
    from plotnine.ggplot import ggplot as ggplot_class
    result = plot_embedding_panel(mock_adata, features=["cluster"], ncol=1)
    assert isinstance(result, (ggplot_class, Compose))
