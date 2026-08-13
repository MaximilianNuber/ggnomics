"""Shared pytest fixtures for ggnomics tests."""

import importlib.util
import os
import sys

import matplotlib
import pytest

# Headless rendering for the whole suite: no interactive windows.
matplotlib.use("Agg")

# Load examples/00_mock_data.py (file starts with digit, use importlib)
_MOCK_PATH = os.path.join(os.path.dirname(__file__), "..", "examples", "00_mock_data.py")
_spec = importlib.util.spec_from_file_location("mock_data", _MOCK_PATH)
_mock_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mock_data)

make_mock_df = _mock_data.make_mock_df
make_mock_anndata = _mock_data.make_mock_anndata
make_mock_sce = _mock_data.make_mock_sce
make_mock_se = _mock_data.make_mock_se
make_small_df = _mock_data.make_small_df
make_large_df = _mock_data.make_large_df
make_mock_de_results = _mock_data.make_mock_de_results
make_mock_coefs = _mock_data.make_mock_coefs
make_mock_mudata = _mock_data.make_mock_mudata
make_mock_repertoire = _mock_data.make_mock_repertoire


@pytest.fixture(scope="session")
def mock_df():
    """500-cell mock DataFrame (session-scoped for speed)."""
    return make_mock_df(n_cells=500, n_genes=50, seed=42)


@pytest.fixture(scope="session")
def small_df():
    """50-cell mock DataFrame."""
    return make_small_df()


@pytest.fixture(scope="session")
def large_df():
    """5000-cell mock DataFrame (for adaptive-size tests)."""
    return make_large_df()


@pytest.fixture(scope="session")
def mock_adata():
    """500-cell AnnData."""
    pytest.importorskip("anndata", reason="anndata not installed")
    return make_mock_anndata(n_cells=500, n_genes=50, seed=42)


@pytest.fixture(scope="session")
def mock_sce():
    """500-cell SingleCellExperiment (skipped if package not available)."""
    pytest.importorskip(
        "singlecellexperiment",
        reason="singlecellexperiment not installed",
    )
    return make_mock_sce(n_cells=500, n_genes=50, seed=42)


@pytest.fixture(scope="session")
def mock_se():
    """24-sample SummarizedExperiment (skipped if package is unavailable)."""
    pytest.importorskip(
        "summarizedexperiment",
        reason="summarizedexperiment not installed",
    )
    pytest.importorskip("biocframe", reason="biocframe not installed")
    return make_mock_se(n_samples=24, n_genes=50, seed=42)


@pytest.fixture(scope="session")
def mock_de_results():
    """DESeq2-style DE results DataFrame."""
    return make_mock_de_results(n_genes=500, seed=42)


@pytest.fixture(scope="session")
def mock_coefs():
    """Elastic net coefficient table."""
    return make_mock_coefs(n_features=50, seed=42)


@pytest.fixture(scope="session")
def mock_mudata():
    """MuData with RNA and protein modalities (skipped if mudata not installed)."""
    pytest.importorskip("mudata", reason="mudata not installed")
    return make_mock_mudata(n_cells=200, n_rna=50, n_prot=10, seed=42)


@pytest.fixture(scope="session")
def mock_repertoire_df():
    """obs DataFrame with clonotype info."""
    return make_mock_repertoire(n_cells=300, n_clonotypes=40, n_samples=3, seed=42)


# ---------------------------------------------------------------------------
# ggnomics.upset fixtures — shared across tests/test_upset_*.py
# ---------------------------------------------------------------------------


@pytest.fixture
def abc_example():
    """The ComplexUpset-distributed A/B/C example with known exact counts.

    325 rows; exclusive counts: () -> 2, (A,) -> 50, (B,) -> 50, (C,) -> 200,
    (A,B) -> 10, (A,C) -> 6, (B,C) -> 6, (A,B,C) -> 1.
    """
    import ggnomics.upset as upset

    return upset.create_upset_abc_example()


@pytest.fixture
def abc_example_with_covariates(abc_example):
    """The A/B/C example plus a numeric and a categorical covariate.

    Deterministic, no randomness: ``score`` is the row position and
    ``batch`` alternates first/second.
    """
    data = abc_example.copy(deep=True)
    data["score"] = range(len(data))
    data["batch"] = ["first", "second"] * (len(data) // 2) + ["first"]
    return data
