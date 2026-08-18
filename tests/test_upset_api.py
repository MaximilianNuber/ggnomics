"""Public-API surface tests for ggnomics.upset.

These tests guard the two required import forms and the ComplexUpset-style
public compatibility surface. They intentionally do not exercise plotting
behavior; see test_upset_plot.py and friends for that.
"""

from __future__ import annotations

import subprocess
import sys
import types

import pytest

# ComplexUpset-style public names that must remain importable from
# ggnomics.upset, per the package's documented compatibility surface.
EXPECTED_PUBLIC_NAMES = [
    "upset",
    "upset_data",
    "intersection_size",
    "intersection_ratio",
    "upset_annotate",
    "upset_stripes",
    "get_size_mode",
    "upset_mode",
    "upset_text_percentage",
    "aes_percentage",
    "upset_query",
    "reverse_log_trans",
    "upset_set_size",
    "intersection_matrix",
    "upset_themes",
    "upset_default_themes",
    "upset_modify_themes",
    "compare_between_intersections",
    "upset_test",
    "create_upset_abc_example",
    "arrange_venn",
    "geom_venn_circle",
    "geom_venn_region",
    "geom_venn_label_region",
    "geom_venn_label_set",
    "scale_color_venn_mix",
    "scale_fill_venn_mix",
]


def test_import_ggnomics_upset_as_upset():
    import ggnomics.upset as upset

    assert isinstance(upset, types.ModuleType)


def test_import_upset_from_ggnomics_package():
    from ggnomics import upset

    assert isinstance(upset, types.ModuleType)


def test_both_import_forms_identify_the_same_module():
    import ggnomics.upset as upset_a
    from ggnomics import upset as upset_b

    assert upset_a is upset_b


def test_ggnomics_upset_is_a_module_not_the_plotting_function():
    import ggnomics

    assert isinstance(ggnomics.upset, types.ModuleType)
    assert not callable(ggnomics.upset) or isinstance(ggnomics.upset, types.ModuleType)


def test_plotting_function_is_callable_as_ggnomics_upset_upset():
    import ggnomics.upset as upset

    assert callable(upset.upset)
    assert not isinstance(upset.upset, types.ModuleType)


def test_root_all_includes_upset():
    import ggnomics

    assert "upset" in ggnomics.__all__


@pytest.mark.parametrize("name", EXPECTED_PUBLIC_NAMES)
def test_upset_all_contains_expected_public_name(name):
    import ggnomics.upset as upset

    assert name in upset.__all__
    assert hasattr(upset, name)


def test_no_reference_to_upsetplot_in_native_module_files():
    import pathlib

    import ggnomics.upset as upset

    package_dir = pathlib.Path(upset.__file__).parent
    offenders = []
    for path in package_dir.glob("*.py"):
        text = path.read_text()
        if "upsetplot" in text:
            offenders.append(str(path))
    assert offenders == []


def test_core_import_succeeds_without_scipy_statsmodels_upsetplot():
    """Simulate an environment without the optional statistics packages (and
    without upsetplot, which must never be a runtime dependency at all) by
    hiding them via sys.modules, then verify core ggnomics and
    ggnomics.upset still import successfully.

    The container backends (anndata, etc.) are hidden too, purely because
    they themselves depend on scipy internally and would otherwise fail for
    a reason unrelated to what this test is checking; that "container
    backend is optional" behavior is already covered by test_backends.py.
    """

    snippet = """
import sys
for name in ("scipy", "statsmodels", "upsetplot", "anndata", "singlecellexperiment", "summarizedexperiment", "biocframe", "mudata", "sklearn"):
    sys.modules[name] = None
import types
import ggnomics
import ggnomics.upset as upset_a
from ggnomics import upset as upset_b
assert isinstance(upset_a, types.ModuleType)
assert upset_a is upset_b
assert callable(upset_a.upset)
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout
