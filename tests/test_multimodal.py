"""Tests for multimodal.py."""

import numpy as np
import pandas as pd
import pytest
from plotnine.composition import Compose
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import plot_adt_qc, plot_bimodal_scatter

# ---------------------------------------------------------------------------
# plot_bimodal_scatter — MuData
# ---------------------------------------------------------------------------


def test_bimodal_scatter_mudata(mock_mudata):
    pytest.importorskip("mudata", reason="mudata not installed")
    p = plot_bimodal_scatter(
        mock_mudata,
        x_feature="Gene0001",
        y_feature="Prot01",
        x_mod="rna",
        y_mod="prot",
    )
    assert isinstance(p, ggplot_class)


def test_bimodal_scatter_mudata_color(mock_mudata):
    pytest.importorskip("mudata", reason="mudata not installed")
    p = plot_bimodal_scatter(
        mock_mudata,
        x_feature="Gene0001",
        y_feature="Prot01",
        x_mod="rna",
        y_mod="prot",
        color="cluster",
    )
    assert isinstance(p, ggplot_class)


def test_bimodal_scatter_mudata_title(mock_mudata):
    pytest.importorskip("mudata", reason="mudata not installed")
    p = plot_bimodal_scatter(
        mock_mudata,
        x_feature="Gene0001",
        y_feature="Prot01",
        x_mod="rna",
        y_mod="prot",
        title="RNA vs Protein",
    )
    assert isinstance(p, ggplot_class)


def test_bimodal_scatter_mudata_same_feature_name_across_modalities():
    """x_feature == y_feature (e.g. a gene symbol matching an antibody name,
    like CITE-seq "CD38") must plot each modality's own values, not collapse
    to a single column and draw a spurious perfect diagonal.
    """
    mudata = pytest.importorskip("mudata", reason="mudata not installed")
    anndata = pytest.importorskip("anndata", reason="anndata not installed")

    rng = np.random.default_rng(0)
    n = 50
    rna = anndata.AnnData(
        X=rng.integers(0, 5, size=(n, 1)).astype(float),
        var=pd.DataFrame(index=["CD38"]),
    )
    prot = anndata.AnnData(
        X=rng.normal(loc=10, scale=1, size=(n, 1)),
        var=pd.DataFrame(index=["CD38"]),
    )
    mdata = mudata.MuData({"rna": rna, "prot": prot})

    p = plot_bimodal_scatter(mdata, x_feature="CD38", y_feature="CD38", x_mod="rna", y_mod="prot")

    assert isinstance(p, ggplot_class)
    x_vals, y_vals = p.data.iloc[:, 0].to_numpy(), p.data.iloc[:, 1].to_numpy()
    assert not np.allclose(x_vals, y_vals)
    assert np.allclose(np.sort(x_vals), np.sort(np.asarray(rna.X).ravel()))
    assert np.allclose(np.sort(y_vals), np.sort(np.asarray(prot.X).ravel()))
    # Axis labels still show the shared, human-meaningful feature name.
    assert p.labels.x == "CD38"
    assert p.labels.y == "CD38"


def test_bimodal_scatter_add_marginal_returns_compose(mock_mudata):
    """add_marginal=True → delegates to plot_scatter_marginal → returns Compose."""
    pytest.importorskip("mudata", reason="mudata not installed")
    result = plot_bimodal_scatter(
        mock_mudata,
        x_feature="Gene0001",
        y_feature="Prot01",
        x_mod="rna",
        y_mod="prot",
        add_marginal=True,
    )
    assert isinstance(result, Compose)


def test_bimodal_scatter_missing_modality_raises(mock_mudata):
    pytest.importorskip("mudata", reason="mudata not installed")
    with pytest.raises(KeyError):
        plot_bimodal_scatter(
            mock_mudata,
            x_feature="Gene0001",
            y_feature="Prot01",
            x_mod="nonexistent",
            y_mod="prot",
        )


def test_bimodal_scatter_missing_feature_raises(mock_mudata):
    pytest.importorskip("mudata", reason="mudata not installed")
    with pytest.raises(KeyError):
        plot_bimodal_scatter(
            mock_mudata,
            x_feature="NonexistentGene",
            y_feature="Prot01",
            x_mod="rna",
            y_mod="prot",
        )


def test_bimodal_scatter_partial_overlap_warns():
    pytest.importorskip("mudata")
    import anndata
    import mudata

    rng = np.random.default_rng(0)
    n = 50
    rna = anndata.AnnData(X=rng.normal(size=(n, 3)).astype(np.float32))
    rna.var_names = ["g1", "g2", "g3"]
    rna.obs_names = [f"cell{i}" for i in range(n)]

    # Protein modality only shares the first 30 observations.
    prot = anndata.AnnData(X=rng.normal(size=(30, 2)).astype(np.float32))
    prot.var_names = ["p1", "p2"]
    prot.obs_names = [f"cell{i}" for i in range(30)]

    mdata = mudata.MuData({"rna": rna, "prot": prot})

    with pytest.warns(UserWarning, match="dropped"):
        p = plot_bimodal_scatter(mdata, x_feature="g1", y_feature="p1", x_mod="rna", y_mod="prot")
    assert isinstance(p, ggplot_class)
    assert len(p.data) == 30


def test_bimodal_scatter_anndata_layers():
    import anndata

    rng = np.random.default_rng(0)
    n = 40
    ad = anndata.AnnData(X=rng.normal(size=(n, 3)).astype(np.float32))
    ad.var_names = ["g1", "g2", "g3"]
    ad.layers["rna"] = rng.normal(size=(n, 3)).astype(np.float32)
    ad.layers["prot"] = rng.normal(size=(n, 3)).astype(np.float32)

    p = plot_bimodal_scatter(ad, x_feature="g1", y_feature="g2", x_mod="rna", y_mod="prot")
    assert isinstance(p, ggplot_class)


def test_bimodal_scatter_unsupported_type_raises():
    with pytest.raises(TypeError, match="plot_bimodal_scatter does not support"):
        plot_bimodal_scatter(object(), x_feature="x", y_feature="y")


# ---------------------------------------------------------------------------
# plot_adt_qc — AnnData (protein modality)
# ---------------------------------------------------------------------------


def _make_protein_adata(n_cells=200, n_prot=10, seed=42):
    import anndata

    rng = np.random.default_rng(seed)
    prot_names = [f"Prot{i + 1:02d}" for i in range(n_prot)]
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


def test_adt_qc_missing_group_by_raises(protein_adata):
    with pytest.raises(KeyError):
        plot_adt_qc(protein_adata, isotype_controls=["IgG1", "IgG2a"], group_by="nonexistent")


def test_adt_qc_unsupported_type_raises():
    with pytest.raises(TypeError, match="plot_adt_qc does not support"):
        plot_adt_qc(object(), isotype_controls=["a"])


def test_adt_qc_dataframe_requires_features():
    df = pd.DataFrame({"Ab1": [1.0, 2.0], "Ab2": [2.0, 1.0], "IgG1": [0.5, 0.5]})
    with pytest.raises(ValueError, match="features"):
        plot_adt_qc(df, isotype_controls=["IgG1"])


def test_adt_qc_dataframe_with_features():
    df = pd.DataFrame(
        {
            "Ab1": [1.0, 2.0, 3.0],
            "Ab2": [2.0, 1.0, 0.5],
            "IgG1": [0.5, 0.5, 0.4],
            "sample": ["S1", "S1", "S2"],
        }
    )
    p = plot_adt_qc(df, isotype_controls=["IgG1"], features=["Ab1", "Ab2", "IgG1"])
    assert isinstance(p, ggplot_class)
    p2 = plot_adt_qc(df, isotype_controls=["IgG1"], features=["Ab1", "Ab2", "IgG1"], group_by="sample")
    assert p2.facet is not None


def test_adt_qc_sce():
    pytest.importorskip("singlecellexperiment")
    import biocframe
    from singlecellexperiment import SingleCellExperiment

    rng = np.random.default_rng(0)
    n_prot, n_cells = 5, 40
    counts = np.abs(rng.normal(3, 1, (n_prot, n_cells))).astype(np.float32)
    names = ["Ab1", "Ab2", "Ab3", "IgG1", "IgG2a"]
    sce = SingleCellExperiment(
        assays={"counts": counts},
        row_names=names,
        column_names=[f"cell{i}" for i in range(n_cells)],
        column_data=biocframe.BiocFrame({"sample": ["S1"] * 20 + ["S2"] * 20}),
    )
    p = plot_adt_qc(sce, isotype_controls=["IgG1", "IgG2a"])
    assert isinstance(p, ggplot_class)
    p.draw()


def test_adt_qc_se():
    pytest.importorskip("summarizedexperiment")
    from summarizedexperiment import SummarizedExperiment

    rng = np.random.default_rng(0)
    n_prot, n_samples = 5, 20
    counts = np.abs(rng.normal(3, 1, (n_prot, n_samples))).astype(np.float32)
    names = ["Ab1", "Ab2", "Ab3", "IgG1", "IgG2a"]
    se = SummarizedExperiment(
        assays={"counts": counts},
        row_names=names,
        column_names=[f"sample{i}" for i in range(n_samples)],
    )
    p = plot_adt_qc(se, isotype_controls=["IgG1", "IgG2a"])
    assert isinstance(p, ggplot_class)
    p.draw()


def _make_adt_mudata():
    mudata = pytest.importorskip("mudata")
    import anndata

    adt = _make_protein_adata()
    rna = anndata.AnnData(X=np.zeros((adt.n_obs, 3), dtype=np.float32))
    rna.var_names = ["g1", "g2", "g3"]
    rna.obs_names = adt.obs_names
    return mudata.MuData({"rna": rna, "prot": adt})


def test_adt_qc_mudata_default_modality():
    mdata = _make_adt_mudata()
    p = plot_adt_qc(mdata, isotype_controls=["IgG1", "IgG2a"])
    assert isinstance(p, ggplot_class)


def test_adt_qc_mudata_group_by_shared_obs():
    mdata = _make_adt_mudata()
    mdata.obs["sample"] = mdata.mod["prot"].obs["sample"].to_numpy()
    p = plot_adt_qc(mdata, isotype_controls=["IgG1", "IgG2a"], group_by="sample")
    assert p.facet is not None


def test_adt_qc_mudata_missing_modality_raises():
    mdata = _make_adt_mudata()
    with pytest.raises(KeyError):
        plot_adt_qc(mdata, isotype_controls=["IgG1"], mod="nonexistent")
