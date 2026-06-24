"""Tests for multimodal.py."""

import numpy as np
import pandas as pd
import pytest
from plotnine.ggplot import ggplot as ggplot_class
from plotnine.composition import Compose

from ggnomics import plot_bimodal_scatter, plot_adt_qc


# ---------------------------------------------------------------------------
# plot_bimodal_scatter — MuData
# ---------------------------------------------------------------------------


def test_bimodal_scatter_mudata(mock_mudata):
    pytest.importorskip("mudata", reason="mudata not installed")
    p = plot_bimodal_scatter(
        mock_mudata, x_feature="Gene0001", y_feature="Prot01",
        x_mod="rna", y_mod="prot",
    )
    assert isinstance(p, ggplot_class)


def test_bimodal_scatter_mudata_color(mock_mudata):
    pytest.importorskip("mudata", reason="mudata not installed")
    p = plot_bimodal_scatter(
        mock_mudata, x_feature="Gene0001", y_feature="Prot01",
        x_mod="rna", y_mod="prot", color="cluster",
    )
    assert isinstance(p, ggplot_class)


def test_bimodal_scatter_mudata_title(mock_mudata):
    pytest.importorskip("mudata", reason="mudata not installed")
    p = plot_bimodal_scatter(
        mock_mudata, x_feature="Gene0001", y_feature="Prot01",
        x_mod="rna", y_mod="prot",
        title="RNA vs Protein",
    )
    assert isinstance(p, ggplot_class)


def test_bimodal_scatter_add_marginal_returns_compose(mock_mudata):
    """add_marginal=True → delegates to plot_scatter_marginal → returns Compose."""
    pytest.importorskip("mudata", reason="mudata not installed")
    result = plot_bimodal_scatter(
        mock_mudata, x_feature="Gene0001", y_feature="Prot01",
        x_mod="rna", y_mod="prot",
        add_marginal=True,
    )
    assert isinstance(result, Compose)


def test_bimodal_scatter_missing_modality_raises(mock_mudata):
    pytest.importorskip("mudata", reason="mudata not installed")
    with pytest.raises(KeyError):
        plot_bimodal_scatter(
            mock_mudata, x_feature="Gene0001", y_feature="Prot01",
            x_mod="nonexistent", y_mod="prot",
        )


# ---------------------------------------------------------------------------
# plot_adt_qc — AnnData (protein modality)
# ---------------------------------------------------------------------------


def _make_protein_adata(n_cells=200, n_prot=10, seed=42):
    import anndata
    rng = np.random.default_rng(seed)
    prot_names = [f"Prot{i+1:02d}" for i in range(n_prot)]
    iso_names = ["IgG1", "IgG2a"]
    all_names = prot_names + iso_names
    X = np.abs(rng.normal(3, 1, (n_cells, len(all_names)))).astype(np.float32)
    adata = anndata.AnnData(X=X)
    adata.var_names = all_names
    adata.obs["sample"] = np.where(rng.random(n_cells) < 0.5, "S1", "S2")
    adata.obs["cluster"] = np.array([f"C{i}" for i in rng.integers(0, 3, n_cells)])
    return adata


@pytest.fixture(scope="module")
def protein_adata():
    return _make_protein_adata()


def test_adt_qc_returns_ggplot(protein_adata):
    p = plot_adt_qc(protein_adata, isotype_controls=["IgG1", "IgG2a"])
    assert isinstance(p, ggplot_class)


def test_adt_qc_log1p_false(protein_adata):
    p = plot_adt_qc(protein_adata, isotype_controls=["IgG1", "IgG2a"], log1p=False)
    assert isinstance(p, ggplot_class)


def test_adt_qc_group_by(protein_adata):
    p = plot_adt_qc(protein_adata, isotype_controls=["IgG1", "IgG2a"], group_by="sample")
    assert isinstance(p, ggplot_class)
    assert p.facet is not None


def test_adt_qc_title(protein_adata):
    p = plot_adt_qc(protein_adata, isotype_controls=["IgG1", "IgG2a"], title="ADT QC")
    assert isinstance(p, ggplot_class)


def test_adt_qc_empty_isotype_raises(protein_adata):
    with pytest.raises(ValueError, match="empty"):
        plot_adt_qc(protein_adata, isotype_controls=[])


def test_adt_qc_missing_isotype_raises(protein_adata):
    with pytest.raises((ValueError, KeyError)):
        plot_adt_qc(protein_adata, isotype_controls=["NonExistentIso"])
