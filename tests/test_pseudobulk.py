"""Tests for pseudobulk.py."""

import pandas as pd
import pytest
from plotnine.ggplot import ggplot as ggplot_class
from plotnine.composition import Compose

from ggnomics import plot_pseudobulk_qc, plot_pseudobulk_de


def _make_results(mock_de_results, clusters=("C0", "C1", "C2")):
    return {c: mock_de_results.copy() for c in clusters}


# ---------------------------------------------------------------------------
# plot_pseudobulk_qc — returns Compose
# ---------------------------------------------------------------------------


def test_pseudobulk_qc_adata(mock_adata):
    result = plot_pseudobulk_qc(mock_adata, sample_by="sample", group_by="cluster")
    assert isinstance(result, Compose)


def test_pseudobulk_qc_with_condition(mock_adata):
    # `batch` is assigned independently per cell in the mock generator, so it
    # is not constant within a sample — build a condition column that
    # genuinely is sample-level, since plot_pseudobulk_qc now requires that.
    adata = mock_adata.copy()
    sample_to_condition = {"S1": "ctrl", "S2": "ctrl", "S3": "treated", "S4": "treated"}
    adata.obs["condition"] = adata.obs["sample"].map(sample_to_condition)

    result = plot_pseudobulk_qc(
        adata, sample_by="sample", group_by="cluster", condition_by="condition"
    )
    assert isinstance(result, Compose)


def test_pseudobulk_qc_conflicting_condition_raises(mock_adata):
    """A condition column that varies within a sample must raise, not silently
    pick a majority vote."""
    with pytest.raises(ValueError, match="multiple"):
        plot_pseudobulk_qc(
            mock_adata, sample_by="sample", group_by="cluster", condition_by="batch"
        )


def test_pseudobulk_qc_missing_col_raises(mock_adata):
    with pytest.raises(KeyError):
        plot_pseudobulk_qc(mock_adata, sample_by="nonexistent", group_by="cluster")


def test_pseudobulk_qc_missing_condition_raises(mock_adata):
    with pytest.raises(KeyError):
        plot_pseudobulk_qc(
            mock_adata, sample_by="sample", group_by="cluster",
            condition_by="nonexistent_col"
        )


def test_pseudobulk_qc_sce(mock_sce):
    result = plot_pseudobulk_qc(mock_sce, sample_by="sample", group_by="cluster")
    assert isinstance(result, Compose)


def test_pseudobulk_qc_se(mock_se):
    result = plot_pseudobulk_qc(mock_se, sample_by="condition", group_by="batch")
    assert isinstance(result, Compose)


def test_pseudobulk_qc_dataframe_requires_features(mock_df):
    with pytest.raises(ValueError, match="features"):
        plot_pseudobulk_qc(mock_df, sample_by="sample", group_by="cluster")


def test_pseudobulk_qc_dataframe_with_features(mock_df):
    features = [f"Gene{i+1:04d}" for i in range(10)]
    result = plot_pseudobulk_qc(
        mock_df, sample_by="sample", group_by="cluster", features=features
    )
    assert isinstance(result, Compose)


def test_pseudobulk_qc_dataframe_prefers_n_counts_for_libsize(mock_df):
    # n_counts is present, so features is only required for the PCA panel,
    # not for library size.
    features = [f"Gene{i+1:04d}" for i in range(10)]
    result = plot_pseudobulk_qc(
        mock_df, sample_by="sample", group_by="cluster", features=features
    )
    assert isinstance(result, Compose)


def test_pseudobulk_qc_unsupported_type_raises():
    with pytest.raises(TypeError, match="plot_pseudobulk_qc does not support"):
        plot_pseudobulk_qc(object(), sample_by="s", group_by="g")


def test_pseudobulk_qc_sparse_dense_equivalence():
    """plot_pseudobulk_qc must not raise for sparse AnnData input, and its
    underlying pseudobulk aggregation must agree exactly with the dense path
    (checked directly against the private helper, since Compose objects
    don't expose panels by stable index)."""
    anndata = pytest.importorskip("anndata")
    import numpy as np
    from scipy import sparse
    from ggnomics.pseudobulk import _pseudobulk_aggregate, _sparse_row_sums

    rng = np.random.default_rng(3)
    n_cells, n_genes = 120, 15
    X = rng.negative_binomial(3, 0.4, size=(n_cells, n_genes)).astype(float)
    gene_names = [f"Gene{i+1:04d}" for i in range(n_genes)]
    samples = rng.choice(["S1", "S2", "S3"], n_cells)
    clusters = rng.choice(["C0", "C1"], n_cells)

    def _make(x):
        ad = anndata.AnnData(X=x.copy())
        ad.var_names = gene_names
        ad.obs["sample"] = samples
        ad.obs["cluster"] = clusters
        return ad

    ad_dense = _make(X)
    ad_sparse = _make(sparse.csr_matrix(X))

    # Full plots must build without error for both dense and sparse input.
    plot_pseudobulk_qc(ad_dense, sample_by="sample", group_by="cluster")
    plot_pseudobulk_qc(ad_sparse, sample_by="sample", group_by="cluster")

    unique_samples = pd.unique(samples)
    lib_dense = _sparse_row_sums(X)
    lib_sparse = _sparse_row_sums(sparse.csr_matrix(X))
    np.testing.assert_allclose(lib_dense, lib_sparse)

    pb_dense = _pseudobulk_aggregate(X, samples, unique_samples)
    pb_sparse = _pseudobulk_aggregate(sparse.csr_matrix(X), samples, unique_samples)
    np.testing.assert_allclose(pb_dense, pb_sparse)


# ---------------------------------------------------------------------------
# plot_pseudobulk_de
# ---------------------------------------------------------------------------


def test_pseudobulk_de_volcano_returns_compose(mock_de_results):
    results = _make_results(mock_de_results)
    comp = plot_pseudobulk_de(results, mode="volcano", ncol=2)
    assert isinstance(comp, Compose)


def test_pseudobulk_de_volcano_title(mock_de_results):
    results = _make_results(mock_de_results)
    comp = plot_pseudobulk_de(results, mode="volcano", ncol=2, title="DE Grid")
    assert isinstance(comp, Compose)


def test_pseudobulk_de_volcano_empty_raises(mock_de_results):
    with pytest.raises(ValueError, match="empty"):
        plot_pseudobulk_de({}, mode="volcano")


def test_pseudobulk_de_summary_bar(mock_de_results):
    results = _make_results(mock_de_results)
    p = plot_pseudobulk_de(results, mode="summary_bar")
    assert isinstance(p, ggplot_class)


def test_pseudobulk_de_summary_bar_title(mock_de_results):
    results = _make_results(mock_de_results)
    p = plot_pseudobulk_de(results, mode="summary_bar", title="DE Summary")
    assert isinstance(p, ggplot_class)


def test_pseudobulk_de_upset(mock_de_results):
    pytest.importorskip("upsetplot", reason="upsetplot not installed")
    results = _make_results(mock_de_results)
    fig = plot_pseudobulk_de(results, mode="upset", gene_col="gene")
    assert fig is not None


def test_pseudobulk_de_unknown_mode_raises(mock_de_results):
    results = _make_results(mock_de_results)
    with pytest.raises(ValueError, match="Unknown mode"):
        plot_pseudobulk_de(results, mode="bogus")
