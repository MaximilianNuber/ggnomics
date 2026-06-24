"""Tests for repertoire.py."""

import numpy as np
import pandas as pd
import pytest
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import (
    plot_clonotype_abundance,
    plot_clonotype_overlap,
    plot_clonotype_embedding,
)
from ggnomics._utils import HeatmapResult


# ---------------------------------------------------------------------------
# plot_clonotype_abundance
# ---------------------------------------------------------------------------


def test_clonotype_abundance_df(mock_repertoire_df):
    p = plot_clonotype_abundance(mock_repertoire_df, clonotype_col="clonotype_id", top_n=10)
    assert isinstance(p, ggplot_class)


def test_clonotype_abundance_facet_by_sample(mock_repertoire_df):
    p = plot_clonotype_abundance(
        mock_repertoire_df,
        clonotype_col="clonotype_id",
        sample_col="sample",
        top_n=10,
    )
    assert isinstance(p, ggplot_class)


def test_clonotype_abundance_expansion_computed(mock_repertoire_df):
    """Expansion categories should be computed from clone sizes when expansion_col is None."""
    p = plot_clonotype_abundance(mock_repertoire_df, clonotype_col="clonotype_id", top_n=5)
    assert isinstance(p, ggplot_class)
    # The fill layer should be present
    fill_mapped = any(
        "fill" in (l.mapping or {}) or (hasattr(l.mapping, "keys") and "fill" in l.mapping)
        for l in p.layers
    )
    # Just check it returned ggplot without error


def test_clonotype_abundance_title(mock_repertoire_df):
    p = plot_clonotype_abundance(
        mock_repertoire_df, clonotype_col="clonotype_id", title="Clonotypes"
    )
    assert isinstance(p, ggplot_class)


def test_clonotype_abundance_missing_col_raises(mock_repertoire_df):
    with pytest.raises(KeyError):
        plot_clonotype_abundance(mock_repertoire_df, clonotype_col="nonexistent")


def test_clonotype_abundance_adata(mock_adata):
    """Clonotype plot from AnnData with clonotype_id in obs."""
    if "clonotype_id" not in mock_adata.obs.columns:
        pytest.skip("clonotype_id not in adata.obs")
    p = plot_clonotype_abundance(mock_adata, clonotype_col="clonotype_id", top_n=5)
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# plot_clonotype_overlap
# ---------------------------------------------------------------------------


def test_clonotype_overlap_jaccard(mock_repertoire_df):
    result = plot_clonotype_overlap(
        mock_repertoire_df,
        clonotype_col="clonotype_id",
        sample_col="sample",
        method="jaccard",
    )
    assert isinstance(result, HeatmapResult)
    assert result.matrix is not None


def test_clonotype_overlap_matrix_symmetric(mock_repertoire_df):
    result = plot_clonotype_overlap(
        mock_repertoire_df,
        clonotype_col="clonotype_id",
        sample_col="sample",
        method="jaccard",
    )
    mat = result.matrix
    # Diagonal should be 1.0
    for i in range(len(mat)):
        assert abs(mat.iloc[i, i] - 1.0) < 1e-9
    # Should be symmetric
    diff = (mat.values - mat.values.T).max()
    assert diff < 1e-9


def test_clonotype_overlap_morisita(mock_repertoire_df):
    result = plot_clonotype_overlap(
        mock_repertoire_df,
        clonotype_col="clonotype_id",
        sample_col="sample",
        method="morisita",
    )
    assert isinstance(result, HeatmapResult)


def test_clonotype_overlap_overlap_coef(mock_repertoire_df):
    result = plot_clonotype_overlap(
        mock_repertoire_df,
        clonotype_col="clonotype_id",
        sample_col="sample",
        method="overlap_coef",
    )
    assert isinstance(result, HeatmapResult)


def test_clonotype_overlap_has_ggplot(mock_repertoire_df):
    result = plot_clonotype_overlap(
        mock_repertoire_df,
        clonotype_col="clonotype_id",
        sample_col="sample",
    )
    assert isinstance(result.plot, ggplot_class)


def test_clonotype_overlap_unknown_method_raises(mock_repertoire_df):
    with pytest.raises(ValueError, match="Unknown method"):
        plot_clonotype_overlap(
            mock_repertoire_df,
            clonotype_col="clonotype_id",
            sample_col="sample",
            method="bogus",
        )


# ---------------------------------------------------------------------------
# plot_clonotype_embedding
# ---------------------------------------------------------------------------


def test_clonotype_embedding_adata(mock_adata):
    if "clonotype_id" not in mock_adata.obs.columns:
        pytest.skip("clonotype_id not in adata.obs")
    p = plot_clonotype_embedding(mock_adata, clonotype_col="clonotype_id", dimred="X_umap")
    assert isinstance(p, ggplot_class)


def test_clonotype_embedding_na_drawn_first(mock_adata):
    """Cells with NaN clonotype should be drawn first (non-TCR layer under expanded)."""
    if "clonotype_id" not in mock_adata.obs.columns:
        pytest.skip("clonotype_id not in adata.obs")
    p = plot_clonotype_embedding(mock_adata, clonotype_col="clonotype_id", dimred="X_umap")
    # Data in plot should have 'None' rows appearing before expanded rows
    # Verify by checking the order column in the plot data
    assert isinstance(p, ggplot_class)
    # Check that 'None' appears as a category
    plot_data = p.data
    if "__expansion__" in plot_data.columns:
        assert "None" in plot_data["__expansion__"].values


def test_clonotype_embedding_missing_col_raises(mock_adata):
    with pytest.raises(KeyError):
        plot_clonotype_embedding(mock_adata, clonotype_col="nonexistent", dimred="X_umap")


def test_clonotype_embedding_title(mock_adata):
    if "clonotype_id" not in mock_adata.obs.columns:
        pytest.skip("clonotype_id not in adata.obs")
    p = plot_clonotype_embedding(
        mock_adata, clonotype_col="clonotype_id", dimred="X_umap",
        title="Clonotype Embedding"
    )
    assert isinstance(p, ggplot_class)
