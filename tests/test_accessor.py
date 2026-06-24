"""Tests for DataAccessor."""

import numpy as np
import pandas as pd
import pytest

from ggnomics._accessor import DataAccessor


# ---------------------------------------------------------------------------
# DataFrame
# ---------------------------------------------------------------------------


def test_accessor_dataframe_type(mock_df):
    acc = DataAccessor(mock_df)
    assert acc.object_type() == "dataframe"


def test_accessor_dataframe_n_obs(mock_df):
    acc = DataAccessor(mock_df)
    assert acc.n_obs() == len(mock_df)


def test_accessor_dataframe_obs_names(mock_df):
    acc = DataAccessor(mock_df)
    names = acc.obs_names()
    assert len(names) == len(mock_df)
    assert names[0] == str(mock_df.index[0])


def test_accessor_dataframe_var_names(mock_df):
    acc = DataAccessor(mock_df)
    var = acc.var_names()
    assert "cluster" in var
    assert "UMAP1" in var


def test_accessor_dataframe_obs(mock_df):
    acc = DataAccessor(mock_df)
    obs = acc.obs()
    assert isinstance(obs, pd.DataFrame)
    assert "cluster" in obs.columns


def test_accessor_dataframe_get_embedding(mock_df):
    acc = DataAccessor(mock_df)
    emb = acc.get_embedding("UMAP")
    assert isinstance(emb, pd.DataFrame)
    assert emb.shape == (len(mock_df), 2)
    assert list(emb.columns) == ["umap_1", "umap_2"]


def test_accessor_dataframe_get_embedding_pca(mock_df):
    acc = DataAccessor(mock_df)
    emb = acc.get_embedding("PCA")
    assert emb.shape[0] == len(mock_df)
    assert emb.shape[1] >= 5


def test_accessor_dataframe_get_expression(mock_df):
    acc = DataAccessor(mock_df)
    expr = acc.get_expression(["Gene0001", "Gene0002"])
    assert isinstance(expr, pd.DataFrame)
    assert expr.shape == (len(mock_df), 2)
    assert list(expr.columns) == ["Gene0001", "Gene0002"]


def test_accessor_dataframe_missing_embedding(mock_df):
    acc = DataAccessor(mock_df)
    with pytest.raises(KeyError):
        acc.get_embedding("TSNE_NONEXISTENT")


def test_accessor_dataframe_missing_feature(mock_df):
    acc = DataAccessor(mock_df)
    with pytest.raises(KeyError):
        acc.get_expression(["NonExistentGene"])


def test_accessor_unsupported_type():
    with pytest.raises(TypeError):
        DataAccessor(object())


# ---------------------------------------------------------------------------
# AnnData
# ---------------------------------------------------------------------------


def test_accessor_anndata_type(mock_adata):
    acc = DataAccessor(mock_adata)
    assert acc.object_type() == "anndata"


def test_accessor_anndata_n_obs(mock_adata):
    acc = DataAccessor(mock_adata)
    assert acc.n_obs() == mock_adata.n_obs


def test_accessor_anndata_embedding(mock_adata):
    acc = DataAccessor(mock_adata)
    emb = acc.get_embedding("X_pca")
    assert isinstance(emb, pd.DataFrame)
    assert emb.shape[0] == mock_adata.n_obs


def test_accessor_anndata_embedding_no_prefix(mock_adata):
    acc = DataAccessor(mock_adata)
    emb = acc.get_embedding("pca")
    assert emb.shape[0] == mock_adata.n_obs


def test_accessor_anndata_embedding_umap(mock_adata):
    acc = DataAccessor(mock_adata)
    emb = acc.get_embedding("X_umap")
    assert emb.shape == (mock_adata.n_obs, 2)


def test_accessor_anndata_expression(mock_adata):
    acc = DataAccessor(mock_adata)
    expr = acc.get_expression(["Gene0001", "Gene0002"])
    assert expr.shape == (mock_adata.n_obs, 2)


def test_accessor_anndata_expression_layer(mock_adata):
    acc = DataAccessor(mock_adata)
    expr = acc.get_expression(["Gene0001"], layer="logcounts")
    assert expr.shape == (mock_adata.n_obs, 1)


def test_accessor_anndata_missing_layer(mock_adata):
    acc = DataAccessor(mock_adata)
    with pytest.raises(KeyError):
        acc.get_expression(["Gene0001"], layer="nonexistent_layer")


def test_accessor_anndata_missing_embedding(mock_adata):
    acc = DataAccessor(mock_adata)
    with pytest.raises(KeyError):
        acc.get_embedding("X_tsne_nonexistent")


# ---------------------------------------------------------------------------
# SCE
# ---------------------------------------------------------------------------


def test_accessor_sce_type(mock_sce):
    acc = DataAccessor(mock_sce)
    assert acc.object_type() == "sce"


def test_accessor_sce_n_obs(mock_sce):
    acc = DataAccessor(mock_sce)
    assert acc.n_obs() == mock_sce.shape[1]


def test_accessor_sce_var_names(mock_sce):
    acc = DataAccessor(mock_sce)
    vn = acc.var_names()
    assert len(vn) == mock_sce.shape[0]
    assert vn[0] == "Gene0001"


def test_accessor_sce_embedding(mock_sce):
    acc = DataAccessor(mock_sce)
    emb = acc.get_embedding("PCA")
    assert isinstance(emb, pd.DataFrame)
    assert emb.shape[0] == mock_sce.shape[1]


def test_accessor_sce_embedding_umap(mock_sce):
    acc = DataAccessor(mock_sce)
    emb = acc.get_embedding("UMAP")
    assert emb.shape == (mock_sce.shape[1], 2)


def test_accessor_sce_expression(mock_sce):
    acc = DataAccessor(mock_sce)
    expr = acc.get_expression(["Gene0001", "Gene0002"])
    assert expr.shape == (mock_sce.shape[1], 2)


def test_accessor_sce_expression_assay(mock_sce):
    acc = DataAccessor(mock_sce)
    expr = acc.get_expression(["Gene0001"], layer="logcounts")
    assert expr.shape == (mock_sce.shape[1], 1)


def test_accessor_sce_obs(mock_sce):
    acc = DataAccessor(mock_sce)
    obs = acc.obs()
    assert isinstance(obs, pd.DataFrame)
    assert "cluster" in obs.columns
