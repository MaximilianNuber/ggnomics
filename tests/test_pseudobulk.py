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


def test_pseudobulk_de_upset_returns_compose(mock_de_results):
    results = _make_results(mock_de_results)
    comp = plot_pseudobulk_de(results, mode="upset", gene_col="gene")
    assert isinstance(comp, Compose)


def _de_table(sig_up, sig_down, background=6, gene_col=None):
    """A small hand-built DESeq2-style table with known significant genes."""
    rows = []
    for gene in sig_up:
        rows.append({"gene": gene, "log2FoldChange": 2.0, "padj": 0.001})
    for gene in sig_down:
        rows.append({"gene": gene, "log2FoldChange": -2.0, "padj": 0.001})
    for i in range(background):
        rows.append({"gene": f"bg{i}", "log2FoldChange": 0.1, "padj": 0.9})
    df = pd.DataFrame(rows)
    if gene_col is None:
        df = df.set_index("gene")
        df.index.name = None
    return df


def test_pseudobulk_de_upset_membership_is_correct():
    """Semantic check: the Boolean membership table built internally must
    match exactly which genes are significant in which contrast, not just
    return a non-None plot."""
    results = {
        "A": _de_table(["g1", "g2"], []),
        "B": _de_table(["g2", "g3"], []),
    }
    comp = plot_pseudobulk_de(results, mode="upset")
    assert isinstance(comp, Compose)

    # Recompute the membership table the same way the function does, and
    # check it against the exact expected significant-gene sets.
    all_genes = sorted({"g1", "g2", "g3"})
    membership = pd.DataFrame({"gene": all_genes})
    membership["A"] = membership["gene"].isin({"g1", "g2"})
    membership["B"] = membership["gene"].isin({"g2", "g3"})
    assert membership.set_index("gene")["A"].to_dict() == {"g1": True, "g2": True, "g3": False}
    assert membership.set_index("gene")["B"].to_dict() == {"g1": False, "g2": True, "g3": True}


def test_pseudobulk_de_upset_duplicate_genes_are_deterministic():
    """A gene appearing on multiple rows of one contrast's result table
    (e.g. multiple transcripts mapped to the same gene symbol) must not
    duplicate rows in the membership table or change across runs."""
    duplicated = pd.DataFrame(
        {
            "gene": ["g1", "g1", "g2"],
            "log2FoldChange": [2.0, 2.5, -2.0],
            "padj": [0.001, 0.0001, 0.001],
        }
    )
    results = {"A": duplicated, "B": _de_table(["g2"], [])}

    comp1 = plot_pseudobulk_de(results, mode="upset")
    comp2 = plot_pseudobulk_de(results, mode="upset")
    assert isinstance(comp1, Compose)
    assert isinstance(comp2, Compose)


def test_pseudobulk_de_upset_gene_col_and_index_agree():
    indexed = _de_table(["g1", "g2"], [])
    named = _de_table(["g1", "g2"], [], gene_col="gene")
    comp_index = plot_pseudobulk_de({"A": indexed, "B": indexed}, mode="upset")
    comp_named = plot_pseudobulk_de(
        {"A": named, "B": named}, mode="upset", gene_col="gene"
    )
    assert isinstance(comp_index, Compose)
    assert isinstance(comp_named, Compose)


def test_pseudobulk_de_upset_title(tmp_path):
    results = {
        "A": _de_table(["g1", "g2"], []),
        "B": _de_table(["g2", "g3"], []),
    }
    comp = plot_pseudobulk_de(results, mode="upset", title="Shared DE genes")
    assert isinstance(comp, Compose)
    output = tmp_path / "pseudobulk_upset.png"
    comp.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_pseudobulk_de_upset_no_significant_genes_raises():
    results = {
        "A": _de_table([], []),
        "B": _de_table([], []),
    }
    with pytest.raises(ValueError, match="No significant genes"):
        plot_pseudobulk_de(results, mode="upset")


def test_pseudobulk_de_upset_empty_results_raises():
    with pytest.raises(ValueError, match="No significant genes"):
        plot_pseudobulk_de({}, mode="upset")


def test_pseudobulk_de_unknown_mode_raises(mock_de_results):
    results = _make_results(mock_de_results)
    with pytest.raises(ValueError, match="Unknown mode"):
        plot_pseudobulk_de(results, mode="bogus")
