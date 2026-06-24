"""Shared pytest fixtures for ggnomics tests."""

import importlib.util
import os
import sys

import pytest

# Load examples/00_mock_data.py (file starts with digit, use importlib)
_MOCK_PATH = os.path.join(os.path.dirname(__file__), "..", "examples", "00_mock_data.py")
_spec = importlib.util.spec_from_file_location("mock_data", _MOCK_PATH)
_mock_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mock_data)

make_mock_df = _mock_data.make_mock_df
make_mock_anndata = _mock_data.make_mock_anndata
make_mock_sce = _mock_data.make_mock_sce
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
