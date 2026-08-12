"""Tests for plot_expression, plot_dot, plot_heatmap."""

import numpy as np
import pandas as pd
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


# ---------------------------------------------------------------------------
# plot_expression / plot_dot / plot_heatmap — SummarizedExperiment
# ---------------------------------------------------------------------------


def test_expression_se_returns_ggplot(mock_se):
    p = plot_expression(mock_se, features=FEATURES, group_by="condition")
    assert isinstance(p, ggplot_class)
    p.draw()


def test_dot_se_returns_ggplot(mock_se):
    p = plot_dot(mock_se, features=FEATURES, group_by="condition")
    assert isinstance(p, ggplot_class)
    p.draw()


def test_heatmap_se_returns_ggplot(mock_se):
    p = plot_heatmap(mock_se, features=FEATURES, group_by="condition")
    assert isinstance(p, ggplot_class)
    p.draw()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_expression_empty_features_raises(mock_df):
    with pytest.raises(ValueError, match="non-empty"):
        plot_expression(mock_df, features=[], group_by="cluster")


def test_expression_duplicate_features_raises(mock_df):
    with pytest.raises(ValueError, match="duplicate"):
        plot_expression(mock_df, features=["Gene0001", "Gene0001"], group_by="cluster")


def test_expression_feature_metadata_collision_raises(mock_df):
    with pytest.raises(ValueError, match="ambiguous"):
        plot_expression(mock_df, features=["cluster"], group_by="cluster")


def test_dot_feature_metadata_collision_raises(mock_df):
    with pytest.raises(ValueError, match="ambiguous"):
        plot_dot(mock_df, features=["cluster"], group_by="cluster")


def test_heatmap_empty_features_raises(mock_df):
    with pytest.raises(ValueError, match="non-empty"):
        plot_heatmap(mock_df, features=[])


def test_expression_layer_ignored_for_dataframe(mock_df):
    p1 = plot_expression(mock_df, features=FEATURES, group_by="cluster")
    p2 = plot_expression(mock_df, features=FEATURES, group_by="cluster", layer="nonexistent")
    pd.testing.assert_frame_equal(
        p1.data.reset_index(drop=True), p2.data.reset_index(drop=True)
    )


def test_expression_unsupported_type_raises():
    with pytest.raises(TypeError, match="plot_expression does not support"):
        plot_expression(object(), features=FEATURES, group_by="cluster")


# ---------------------------------------------------------------------------
# Sparse behavior
# ---------------------------------------------------------------------------


def test_expression_sparse_dense_equivalence():
    anndata = pytest.importorskip("anndata")
    from scipy import sparse

    rng = np.random.default_rng(0)
    n_cells, n_genes = 100, 10
    X = rng.negative_binomial(3, 0.5, size=(n_cells, n_genes)).astype(float)
    gene_names = [f"Gene{i+1:04d}" for i in range(n_genes)]

    def _make(x):
        ad = anndata.AnnData(X=x.copy())
        ad.var_names = gene_names
        ad.obs["cluster"] = rng.choice(["A", "B"], n_cells)
        return ad

    ad_dense = _make(X)
    ad_sparse = _make(sparse.csr_matrix(X))

    p_dense = plot_expression(ad_dense, features=gene_names[:3], group_by="cluster")
    p_sparse = plot_expression(ad_sparse, features=gene_names[:3], group_by="cluster")

    np.testing.assert_allclose(
        p_dense.data["expression"].to_numpy(),
        p_sparse.data["expression"].to_numpy(),
    )
