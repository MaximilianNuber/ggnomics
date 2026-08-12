"""Shared environment/cache setup for the ggnomics Quarto vignettes.

Every vignette starts with:

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path("_includes").resolve()))
    from setup import DATA_CACHE, SCRNASEQ_CACHE, EXPRESSIONATLAS_CACHE, EXPERIMENTHUB_CACHE

The cache root is resolved once here so all vignettes and helper modules agree on
where downloaded/processed data lives, without hardcoding any user- or
machine-specific path in the documentation source.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

#: Root cache directory for all documentation data fetches. Override with the
#: ``GGNOMICS_DATA_CACHE`` environment variable (e.g. in CI, or to share a cache
#: across repo checkouts). Defaults to a per-user cache directory, never a path
#: inside the repository itself, so multi-gigabyte datasets are never at risk of
#: being committed.
DATA_CACHE = Path(os.environ.get("GGNOMICS_DATA_CACHE", Path.home() / ".cache" / "ggnomics-docs")).expanduser()

SCRNASEQ_CACHE = DATA_CACHE / "scrnaseq"
EXPRESSIONATLAS_CACHE = DATA_CACHE / "expressionatlas"
EXPERIMENTHUB_CACHE = DATA_CACHE / "experimenthub"
PROCESSED_CACHE = DATA_CACHE / "processed"

for _dir in (SCRNASEQ_CACHE, EXPRESSIONATLAS_CACHE, EXPERIMENTHUB_CACHE, PROCESSED_CACHE):
    _dir.mkdir(parents=True, exist_ok=True)


def ensure_matplotlib_initialized() -> None:
    """Fully initialize matplotlib's Agg backend (including the `ft2font`
    C-extension) before anything else runs.

    Confirmed live, reproducibly: importing `pyexpressionatlas` or
    `experimenthub` (both pull in `rds2py`) *before* matplotlib's backend has
    been fully loaded leaves `matplotlib.ft2font` in a state where a later,
    unrelated `import matplotlib.pyplot` (e.g. triggered lazily by
    `upsetplot` or the first `ggplot.save()`/`_repr_mimebundle_`) raises
    `rds2py.lib_rds_parser.RdsParserError: module 'matplotlib.ft2font' has no
    attribute '__path__'` - a real cross-package import-order interaction,
    not a ggnomics or pydeseq2 bug. Calling this first avoids it entirely.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.close("all")


def configure_warnings() -> None:
    """Silence a small, explicit set of benign warnings so rendered vignettes stay
    readable. Deliberately narrow (module + message match) rather than a blanket
    ``ignore`` filter, so genuine problems still surface.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*FigureCanvasAgg is non-interactive.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        message=r".*`__version__` is deprecated.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"mudata.*",
        message=r".*pull_on_update.*",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r".*Variable names are not unique.*",
    )


def require_network(action: str) -> None:
    """Raise a clear, actionable error if a documentation dataset fetch fails,
    rather than letting a low-level connection traceback surface in rendered docs.
    """
    raise RuntimeError(
        f"ggnomics docs: could not {action}. This vignette requires network access "
        f"to Bioconductor/BiocPy public data services on first run (results are then "
        f"cached under {DATA_CACHE}). If you are offline, set GGNOMICS_DATA_CACHE to "
        f"a directory with a pre-populated cache, or run this vignette elsewhere and "
        f"copy the cache over."
    )
