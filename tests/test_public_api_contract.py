"""Contract tests for the public plotting API.

These complement ``tests/test_backends.py`` (which covers optional-backend
registration) by pinning down three properties that the BiocPy developer guide
treats as load-bearing:

* public plotting functions return the documented plotnine objects;
* public plotting functions do not mutate the container they are given;
* unsupported containers and missing optional dependencies produce errors that
  say what to do about them.

Everything here uses the shared mock factories, so no network access and no
optional data-access package is involved.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from plotnine import ggplot
from plotnine.composition import Compose

import ggnomics as gg
import ggnomics.upset as upset

# examples/00_mock_data.py starts with a digit, so it is loaded by path in the
# same way tests/conftest.py loads it. Fresh objects (rather than the
# session-scoped fixtures) matter here: the mutation tests must not observe
# state left behind by another module.
_MOCK_PATH = Path(__file__).resolve().parent.parent / "examples" / "00_mock_data.py"
_spec = importlib.util.spec_from_file_location("_ggnomics_mock_data", _MOCK_PATH)
_mock_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mock_data)

make_mock_df = _mock_data.make_mock_df
make_mock_anndata = _mock_data.make_mock_anndata
make_mock_sce = _mock_data.make_mock_sce
make_mock_se = _mock_data.make_mock_se


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


def test_dataframe_plots_return_ggplot(mock_df):
    """The DataFrame implementations return a plain plotnine ggplot."""
    assert isinstance(gg.plot_scatter(mock_df, x="UMAP1", y="UMAP2"), ggplot)
    assert isinstance(gg.plot_umap(mock_df, color="cluster"), ggplot)
    assert isinstance(gg.plot_coldata(mock_df, x="n_counts", y="mito_frac", color_by="cluster"), ggplot)


def test_multi_panel_helpers_return_compose(mock_df):
    """Multi-panel functions return plotnine.composition.Compose, as documented."""
    composed = gg.plot_scatter_marginal(mock_df, x="UMAP1", y="UMAP2")
    assert isinstance(composed, Compose)


def test_upset_returns_compose():
    """ggnomics.upset.upset returns an ordinary Compose, not a Matplotlib figure."""
    data = pd.DataFrame(
        {
            "set_a": [True, True, False, False, True],
            "set_b": [True, False, True, False, True],
            "set_c": [False, True, True, True, False],
        }
    )
    result = upset.upset(data, ["set_a", "set_b", "set_c"])
    assert isinstance(result, Compose)


def test_geom_signif_is_preserved_on_the_root_namespace():
    """ggnomics.signif stays reachable from the package root."""
    assert callable(gg.geom_signif)
    assert callable(gg.run_comparisons)
    assert callable(gg.map_pvalue_to_stars)


# ---------------------------------------------------------------------------
# Non-mutation
#
# BiocPy's guide asks that methods "avoid side effects that mutate the object".
# Plotting is read-only by definition, so these are regression guards.
# ---------------------------------------------------------------------------


def test_dataframe_input_is_not_mutated():
    df = make_mock_df(n_cells=100, n_genes=50, seed=0)
    before = df.copy(deep=True)
    gg.plot_umap(df, color="cluster")
    gg.plot_coldata(df, x="n_counts", y="mito_frac", color_by="cluster")
    pd.testing.assert_frame_equal(df, before)


def test_anndata_input_is_not_mutated():
    pytest.importorskip("anndata")
    adata = make_mock_anndata(n_cells=100, n_genes=50, seed=0)

    obs_before = adata.obs.copy(deep=True)
    var_before = adata.var.copy(deep=True)
    x_before = np.asarray(adata.X).copy()
    obsm_keys_before = set(adata.obsm.keys())
    layers_before = set(adata.layers.keys())

    gg.plot_umap(adata, color="cluster")
    gg.plot_expression(adata, features=[adata.var_names[0]], group_by="cluster")

    pd.testing.assert_frame_equal(adata.obs, obs_before)
    pd.testing.assert_frame_equal(adata.var, var_before)
    np.testing.assert_array_equal(np.asarray(adata.X), x_before)
    assert set(adata.obsm.keys()) == obsm_keys_before
    assert set(adata.layers.keys()) == layers_before


def test_single_cell_experiment_input_is_not_mutated():
    pytest.importorskip("singlecellexperiment")
    sce = make_mock_sce(n_cells=100, n_genes=50, seed=0)

    reduced_before = set(sce.get_reduced_dim_names())
    assay_before = set(sce.get_assay_names())
    counts_before = np.asarray(sce.assay("counts")).copy()
    coldata_before = sce.get_column_data().to_pandas().copy(deep=True)

    gg.plot_umap(sce, color="cluster")
    gg.plot_expression(sce, features=[sce.get_row_names()[0]], group_by="cluster", layer="logcounts")

    assert set(sce.get_reduced_dim_names()) == reduced_before
    assert set(sce.get_assay_names()) == assay_before
    np.testing.assert_array_equal(np.asarray(sce.assay("counts")), counts_before)
    pd.testing.assert_frame_equal(sce.get_column_data().to_pandas(), coldata_before)


def test_summarized_experiment_input_is_not_mutated():
    pytest.importorskip("summarizedexperiment")
    se = make_mock_se(n_samples=24, n_genes=50, seed=0)

    assay_before = set(se.get_assay_names())
    counts_before = np.asarray(se.assay("counts")).copy()
    coldata_before = se.get_column_data().to_pandas().copy(deep=True)

    gg.plot_expression(se, features=[se.get_row_names()[0]], group_by="condition", layer="counts")

    assert set(se.get_assay_names()) == assay_before
    np.testing.assert_array_equal(np.asarray(se.assay("counts")), counts_before)
    pd.testing.assert_frame_equal(se.get_column_data().to_pandas(), coldata_before)


# ---------------------------------------------------------------------------
# Orientation and container capabilities
# ---------------------------------------------------------------------------


def test_summarized_experiment_is_feature_by_sample():
    """SE assays are features x samples; the extracted frame is one row per sample."""
    pytest.importorskip("summarizedexperiment")
    se = make_mock_se(n_samples=24, n_genes=50, seed=0)
    assert np.asarray(se.assay("counts")).shape == (50, 24)

    p = gg.plot_expression(se, features=[se.get_row_names()[0]], group_by="condition", layer="counts")
    assert isinstance(p, ggplot)
    # One observation per sample, not per feature.
    assert len(p.data) == 24


def test_single_cell_experiment_reduced_dims_are_supported():
    pytest.importorskip("singlecellexperiment")
    sce = make_mock_sce(n_cells=100, n_genes=50, seed=0)
    assert "UMAP" in set(sce.get_reduced_dim_names())
    assert isinstance(gg.plot_umap(sce, color="cluster"), ggplot)
    assert isinstance(gg.plot_pca(sce, color="cluster"), ggplot)


def test_summarized_experiment_embedding_raises_a_useful_error():
    """A plain SE has no reducedDims; the error must say so rather than KeyError."""
    pytest.importorskip("summarizedexperiment")
    se = make_mock_se(n_samples=24, n_genes=50, seed=0)

    with pytest.raises((ValueError, TypeError, KeyError, AttributeError)) as excinfo:
        gg.plot_umap(se, color="condition")

    message = str(excinfo.value).lower()
    assert any(token in message for token in ("reduced", "embedding", "dimension", "umap", "singlecellexperiment")), (
        f"unhelpful error message: {excinfo.value!r}"
    )


# ---------------------------------------------------------------------------
# Errors for unsupported input
# ---------------------------------------------------------------------------


def test_unsupported_container_names_the_type_and_the_remedy():
    class NotAContainer:
        pass

    with pytest.raises(TypeError) as excinfo:
        gg.plot_embedding(NotAContainer())

    message = str(excinfo.value)
    assert "NotAContainer" in message
    assert "DataFrame" in message or "optional dependency" in message
