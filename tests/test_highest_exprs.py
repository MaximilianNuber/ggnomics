"""Tests for plot_highest_exprs."""

import numpy as np
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


def test_highest_exprs_se_returns_ggplot(mock_se):
    p = plot_highest_exprs(mock_se, n=10)
    assert isinstance(p, ggplot_class)
    p.draw()


def test_highest_exprs_adata_explicit_features(mock_adata):
    features = list(mock_adata.var_names[:5])
    p = plot_highest_exprs(mock_adata, n=3, features=features)
    assert isinstance(p, ggplot_class)
    assert set(p.data["feature"].cat.categories) <= set(features)


def test_highest_exprs_df_no_features_warns(mock_df):
    with pytest.warns(UserWarning, match="no explicit `features`"):
        plot_highest_exprs(mock_df, n=5)


def test_highest_exprs_df_explicit_features_no_warning(mock_df, recwarn):
    plot_highest_exprs(mock_df, n=5, features=["Gene0001", "Gene0002", "Gene0003"])
    assert not any("no explicit `features`" in str(w.message) for w in recwarn.list)


def test_highest_exprs_n_must_be_positive(mock_df):
    with pytest.raises(ValueError, match="n must be >= 1"):
        plot_highest_exprs(mock_df, n=0)


def test_highest_exprs_adata_n_must_be_positive(mock_adata):
    with pytest.raises(ValueError, match="n must be >= 1"):
        plot_highest_exprs(mock_adata, n=0)


def test_highest_exprs_missing_color_cells_by_raises(mock_df):
    with pytest.raises(KeyError):
        plot_highest_exprs(mock_df, n=5, features=["Gene0001"], color_cells_by="nope")


def test_highest_exprs_adata_missing_color_cells_by_raises(mock_adata):
    with pytest.raises(KeyError):
        plot_highest_exprs(mock_adata, n=5, color_cells_by="nope")


def test_highest_exprs_sparse_dense_equivalence():
    anndata = pytest.importorskip("anndata")
    from scipy import sparse

    rng = np.random.default_rng(1)
    n_cells, n_genes = 150, 20
    X = rng.negative_binomial(2, 0.4, size=(n_cells, n_genes)).astype(float)
    gene_names = [f"Gene{i + 1:04d}" for i in range(n_genes)]

    def _make(x):
        ad = anndata.AnnData(X=x.copy())
        ad.var_names = gene_names
        return ad

    ad_dense = _make(X)
    ad_sparse = _make(sparse.csr_matrix(X))

    p_dense = plot_highest_exprs(ad_dense, n=5)
    p_sparse = plot_highest_exprs(ad_sparse, n=5)

    assert list(p_dense.data["feature"].cat.categories) == list(p_sparse.data["feature"].cat.categories)
    d1 = p_dense.data.sort_values("feature").reset_index(drop=True)
    d2 = p_sparse.data.sort_values("feature").reset_index(drop=True)
    np.testing.assert_allclose(d1["fraction"].to_numpy(), d2["fraction"].to_numpy())
