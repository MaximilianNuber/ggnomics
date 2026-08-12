"""Tests for conditional backend discovery."""

import subprocess
import sys

import pandas as pd
import pytest

from ggnomics import _backends


def test_missing_optional_backend_is_not_imported(monkeypatch):
    imported: list[str] = []

    monkeypatch.setattr(_backends, "find_spec", lambda dependency: None)
    monkeypatch.setattr(_backends, "import_module", imported.append)

    _backends.register_installed_backends()

    assert imported == []


def test_only_discovered_backend_is_imported(monkeypatch):
    imported: list[str] = []

    monkeypatch.setattr(
        _backends,
        "find_spec",
        lambda dependency: object() if dependency == "anndata" else None,
    )
    monkeypatch.setattr(_backends, "import_module", imported.append)

    _backends.register_installed_backends()

    assert imported == ["ggnomics._backends.anndata"]


def test_registration_is_idempotent():
    """Calling register_installed_backends() twice must not raise or double-register."""

    _backends.register_installed_backends()
    _backends.register_installed_backends()

    import ggnomics
    import anndata

    df_impl = ggnomics.plot_scatter.dispatch(pd.DataFrame)
    anndata_impl = ggnomics.plot_scatter.dispatch(anndata.AnnData)
    assert df_impl is not anndata_impl


def test_broken_installed_backend_propagates_import_error(monkeypatch):
    """A backend whose package IS installed but whose module fails to import
    for a real reason (not "package absent") must raise, not be silently
    treated as an absent optional dependency."""

    def _broken_import(name):
        if name == "ggnomics._backends.anndata":
            raise RuntimeError("simulated broken backend module")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(
        _backends, "find_spec", lambda dependency: object() if dependency == "anndata" else None
    )
    monkeypatch.setattr(_backends, "import_module", _broken_import)

    with pytest.raises(RuntimeError, match="simulated broken backend"):
        _backends.register_installed_backends()


# ---------------------------------------------------------------------------
# Subprocess-based clean-import tests.
#
# These simulate an "optional package not installed" environment by setting
# sys.modules[name] = None before importing ggnomics (Python's import system
# treats that as authoritative absence), without needing a separate venv per
# scenario. Run in a subprocess so a crash or partial-import doesn't corrupt
# this test process's own already-imported ggnomics module. Snippets are
# written flush-left (column 0) intentionally: they are code for a
# subprocess, not part of this file's own indentation.
# ---------------------------------------------------------------------------


def _run_snippet(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )


def _hide(*names: str) -> str:
    return "\n".join(f"sys.modules[{name!r}] = None" for name in names)


CONTAINER_PACKAGES = ("anndata", "singlecellexperiment", "biocframe", "summarizedexperiment", "mudata")
STATS_PACKAGES = ("scipy", "statsmodels", "sklearn", "upsetplot")


def test_core_only_import_succeeds_without_any_optional_package():
    result = _run_snippet(f"""
import sys
{_hide(*CONTAINER_PACKAGES, *STATS_PACKAGES)}
import ggnomics
import pandas as pd
df = pd.DataFrame({{"x": [1, 2, 3], "y": [3, 2, 1]}})
p = ggnomics.plot_scatter(df, x="x", y="y")
assert type(p).__name__ == "ggplot"
print("OK")
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_anndata_only_registers_only_anndata_backend():
    result = _run_snippet(f"""
import sys
{_hide("singlecellexperiment", "summarizedexperiment", "mudata")}
import ggnomics
import anndata
assert ggnomics.plot_scatter.dispatch(anndata.AnnData) is not ggnomics.plot_scatter.dispatch(object)
assert "ggnomics._backends.singlecellexperiment" not in sys.modules
assert "ggnomics._backends.summarizedexperiment" not in sys.modules
assert "ggnomics._backends.mudata" not in sys.modules
print("OK")
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_se_only_installation_loads_se_registration():
    result = _run_snippet(f"""
import sys
{_hide("anndata", "mudata")}
import ggnomics
from summarizedexperiment import SummarizedExperiment
assert ggnomics.plot_coldata.dispatch(SummarizedExperiment) is not ggnomics.plot_coldata.dispatch(object)
# SE has no embeddings: plot_scatter/plot_embedding must not be registered for it.
assert ggnomics.plot_scatter.dispatch(SummarizedExperiment) is ggnomics.plot_scatter.dispatch(object)
print("OK")
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_sce_backend_module_does_not_import_se_backend_module():
    """SCE and SE are registered by two independent backend modules, both
    gated by the same central `_BACKENDS` loader — `_backends/singlecellexperiment.py`
    must not itself import `_backends/summarizedexperiment.py`.

    Note: `summarizedexperiment` (the *package*) is a real, unavoidable
    transitive dependency of `singlecellexperiment` (SCE extends SE
    upstream), so it will always be importable whenever SCE is — asserting
    otherwise would be a false claim about a third-party package's own
    internals. What must hold is that *ggnomics* has no direct coupling
    between the two backend modules.
    """
    import ggnomics._backends.singlecellexperiment as sce_backend

    assert "summarizedexperiment" not in sce_backend.__dict__
    assert not hasattr(sce_backend, "SummarizedExperiment")


def test_sce_only_registers_sce_but_not_ggnomics_se_backend_when_se_package_absent(monkeypatch):
    """When the `_BACKENDS` loader genuinely can't find `summarizedexperiment`
    (e.g. a hypothetical SCE distribution without that transitive dependency),
    it must not import ggnomics' SE backend module — verified directly
    against the loader rather than against real package internals."""

    imported: list[str] = []
    monkeypatch.setattr(
        _backends,
        "find_spec",
        lambda dependency: object() if dependency == "singlecellexperiment" else None,
    )
    monkeypatch.setattr(_backends, "import_module", imported.append)

    _backends.register_installed_backends()

    assert imported == ["ggnomics._backends.singlecellexperiment"]
    assert "ggnomics._backends.summarizedexperiment" not in imported


def test_mudata_only_registers_mudata_backend():
    result = _run_snippet(f"""
import sys
{_hide("singlecellexperiment", "summarizedexperiment")}
import ggnomics
from mudata import MuData
assert ggnomics.plot_bimodal_scatter.dispatch(MuData) is not ggnomics.plot_bimodal_scatter.dispatch(object)
print("OK")
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_all_backends_installed_registers_every_concrete_class():
    result = _run_snippet("""
import ggnomics
import anndata
from singlecellexperiment import SingleCellExperiment
from summarizedexperiment import SummarizedExperiment
from mudata import MuData

for cls in (anndata.AnnData, SingleCellExperiment):
    assert ggnomics.plot_scatter.dispatch(cls) is not ggnomics.plot_scatter.dispatch(object)
for cls in (anndata.AnnData, SingleCellExperiment, SummarizedExperiment):
    assert ggnomics.plot_coldata.dispatch(cls) is not ggnomics.plot_coldata.dispatch(object)
for cls in (anndata.AnnData, MuData):
    assert ggnomics.plot_bimodal_scatter.dispatch(cls) is not ggnomics.plot_bimodal_scatter.dispatch(object)
print("OK")
""")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
