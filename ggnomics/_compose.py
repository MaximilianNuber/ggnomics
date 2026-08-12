"""Plotnine composition helpers.

Abstracts over plotnine 0.15 (stable operators only) and 0.16+
(``plot_layout`` / ``plot_annotation`` available).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List, Optional

import plotnine as _pn

# Robust version parse: "0.16.0a7" → (0, 16)
_PLOTNINE_VERSION = tuple(
    int(re.match(r"\d+", x).group())  # type: ignore[union-attr]
    for x in _pn.__version__.split(".")[:2]
)
HAS_LAYOUT = _PLOTNINE_VERSION >= (0, 16)

if TYPE_CHECKING:
    from plotnine.composition import Compose


def hstack(*plots, spacer: bool = False) -> "Compose":
    """Place plots side by side (``|`` operator).

    Parameters
    ----------
    *plots:
        Any number of ``ggplot`` or ``Compose`` objects.
    spacer:
        When ``True``, insert a ``plot_spacer()`` between each plot.
    """
    from plotnine.composition import plot_spacer as _sp

    parts: list = []
    for i, p in enumerate(plots):
        if spacer and i > 0:
            parts.append(_sp())
        parts.append(p)

    result = parts[0]
    for p in parts[1:]:
        result = result | p
    return result


def vstack(*plots) -> "Compose":
    """Stack plots vertically (``/`` operator)."""
    result = plots[0]
    for p in plots[1:]:
        result = result / p
    return result


def grid(
    plots: List,
    ncol: int = 2,
    widths: Optional[List] = None,
    heights: Optional[List] = None,
) -> "Compose":
    """Arrange *plots* in a grid.

    On plotnine ≥ 0.16 uses ``plot_layout(ncol=…)`` (``Wrap`` composition).
    On plotnine 0.15 falls back to manual row/column composition with ``|``
    and ``/``, padding the last row with spacers.

    Parameters
    ----------
    plots:
        List of ``ggplot`` / ``Compose`` objects.
    ncol:
        Number of columns.
    widths:
        Column width ratios (plotnine ≥ 0.16 only).
    heights:
        Row height ratios (plotnine ≥ 0.16 only).
    """
    if not plots:
        raise ValueError("plots list is empty.")

    # A single plot can't be wrapped into a Compose without adding phantom panels.
    # Return it unchanged — callers that always need Compose should add a spacer.
    if len(plots) == 1:
        return plots[0]  # type: ignore[return-value]

    if HAS_LAYOUT:
        from plotnine.composition import plot_layout

        result = plots[0]
        for p in plots[1:]:
            result = result + p
        kw: dict = {"ncol": ncol}
        if widths is not None:
            kw["widths"] = widths
        if heights is not None:
            kw["heights"] = heights
        return result + plot_layout(**kw)

    # Plotnine 0.15 fallback: manual rows/cols
    from plotnine.composition import plot_spacer as _sp

    rows = [list(plots[i : i + ncol]) for i in range(0, len(plots), ncol)]
    last = rows[-1]
    while len(last) < ncol:
        last.append(_sp())

    row_comps = [hstack(*row) for row in rows]
    return vstack(*row_comps)


def annotate_composition(
    composition: "Compose",
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    caption: Optional[str] = None,
) -> "Compose":
    """Add title / subtitle / caption to an entire composition.

    Uses ``plot_annotation`` on plotnine ≥ 0.16.  On 0.15 falls back to
    appending ``labs()`` which applies to the last plot.
    """
    if HAS_LAYOUT:
        from plotnine.composition import plot_annotation

        kw: dict = {}
        if title:
            kw["title"] = title
        if subtitle:
            kw["subtitle"] = subtitle
        if caption:
            kw["caption"] = caption
        if kw:
            return composition + plot_annotation(**kw)
        return composition

    # 0.15 fallback
    from plotnine import labs

    kw_labs: dict = {}
    if title:
        kw_labs["title"] = title
    if caption:
        kw_labs["caption"] = caption
    if kw_labs:
        return composition + labs(**kw_labs)
    return composition


def save_composition(composition: "Compose", path: str, dpi: int = 150, **kwargs) -> None:
    """Save a composition to *path*.  Same API as ``ggplot.save()``."""
    composition.save(path, dpi=dpi, **kwargs)
