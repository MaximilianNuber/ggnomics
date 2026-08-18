"""Registration of optional data-container backends.

Backend modules are imported only when their corresponding third-party
package is importable. Importing a backend registers its concrete classes on
the public ``singledispatch`` plotting functions.
"""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

_BACKENDS = {
    "anndata": "ggnomics._backends.anndata",
    "singlecellexperiment": "ggnomics._backends.singlecellexperiment",
    "summarizedexperiment": "ggnomics._backends.summarizedexperiment",
    "mudata": "ggnomics._backends.mudata",
}


def register_installed_backends() -> None:
    """Register plotting implementations for installed optional packages.

    A backend that is not installed is skipped. If it is installed but its
    backend fails to import, the import error is intentionally allowed to
    propagate: a broken installation should not be mistaken for an absent
    optional dependency.
    """

    for dependency, backend in _BACKENDS.items():
        if find_spec(dependency) is not None:
            import_module(backend)


__all__ = ["register_installed_backends"]
