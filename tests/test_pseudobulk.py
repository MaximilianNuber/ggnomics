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
    result = plot_pseudobulk_qc(
        mock_adata, sample_by="sample", group_by="cluster", condition_by="batch"
    )
    assert isinstance(result, Compose)


def test_pseudobulk_qc_missing_col_raises(mock_adata):
    with pytest.raises(KeyError):
        plot_pseudobulk_qc(mock_adata, sample_by="nonexistent", group_by="cluster")


def test_pseudobulk_qc_missing_condition_raises(mock_adata):
    with pytest.raises(KeyError):
        plot_pseudobulk_qc(
            mock_adata, sample_by="sample", group_by="cluster",
            condition_by="nonexistent_col"
        )


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
