import numpy as np
import pandas as pd
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import (
    plot_reduced_dim,
    expression_violin,
    volcano_plot,
    heatmap_from_matrix,
    marker_dotplot_from_matrix,
    qc_scatter,
    qc_histogram,
    cluster_composition_barplot,
    ridge_density,
)


def test_plot_reduced_dim_returns_plot():
    # New API: dimred selects embedding columns by prefix
    df = pd.DataFrame({"x_umap_1": [0.0, 1.0], "x_umap_2": [0.1, 0.2], "cluster": ["A", "B"]})
    p = plot_reduced_dim(df, dimred="umap", color="cluster")
    assert isinstance(p, ggplot_class)


def test_expression_violin_dataframe():
    df = pd.DataFrame({"expr": np.random.rand(20), "cluster": np.random.choice(list("AB"), 20)})
    p = expression_violin(df, value="expr", group="cluster")
    assert isinstance(p, ggplot_class)


def test_volcano_plot():
    de = pd.DataFrame({"log2FC": np.random.randn(30), "FDR": np.clip(np.random.rand(30), 1e-6, 1.0)})
    p = volcano_plot(de, log2fc="log2FC", padj="FDR")
    assert isinstance(p, ggplot_class)


def test_heatmap_from_matrix():
    mat = np.random.randn(5, 3)
    p = heatmap_from_matrix(mat)
    assert isinstance(p, ggplot_class)


def test_marker_dotplot_from_matrix():
    X = np.abs(np.random.randn(40, 4))
    groups = np.random.choice(["c0", "c1"], size=40)
    p = marker_dotplot_from_matrix(X, groups=groups, genes=["g0", "g1"])  # will fallback to g0/g1 labels
    assert isinstance(p, ggplot_class)


def test_qc_plots():
    df = pd.DataFrame({
        "n_counts": np.random.gamma(2, 100, size=100),
        "n_genes": np.random.gamma(2, 50, size=100),
        "mito_frac": np.random.rand(100),
        "cluster": np.random.choice(list("ABC"), 100),
        "condition": np.random.choice(["ctrl", "stim"], 100),
    })
    p1 = qc_scatter(df, x="n_counts", y="n_genes", color="mito_frac")
    p2 = qc_histogram(df, value="mito_frac")
    p3 = cluster_composition_barplot(df, cluster_col="cluster", group_col="condition")
    assert isinstance(p1, ggplot_class) and isinstance(p2, ggplot_class) and isinstance(p3, ggplot_class)


def test_ridge_density():
    df = pd.DataFrame({
        "expr": np.random.randn(200),
        "cluster": np.random.choice(list("ABC"), 200),
    })
    p = ridge_density(df, value="expr", group="cluster")
    assert isinstance(p, ggplot_class)
