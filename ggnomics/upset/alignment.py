"""Measure and correct cross-row panel-margin misalignment in a composition.

plotnine's composition layout (as of the ``0.16.0a`` series this package
targets) aligns the outermost left/right/top/bottom edges of a composition,
but does not equalize the internal column boundary between panels that sit
in different rows of a nested nested ``Beside``-under-``Stack`` layout, such
as the "spacer / set-size" column and the "annotation / matrix" column built
by :func:`~ggnomics.upset.compose_upset`. A panel with wide y-axis tick text
(e.g. long set names) ends up with a different left edge than the panel
stacked above or below it, which has little or no such text.

This module measures that gap by rendering the composition once (a cheap,
throwaway render, not the caller's eventual output) and once more with a
probe margin applied, then computes and applies an explicit
``plot_margin_left`` correction so the panels named by a shared role line
up. It is a workaround for a real limitation of the pinned plotnine alpha,
not a general-purpose layout tool.

Chain: :func:`tag_panel_role` (while building panels) →
:func:`compute_panel_edges` → :func:`compute_margin_sensitivity` →
:func:`suggest_left_margin_corrections` → :func:`apply_left_margin_corrections`.
"""

from __future__ import annotations

import io
from copy import deepcopy
from typing import Sequence

import matplotlib.pyplot as plt
from plotnine import ggplot, theme
from plotnine.composition import Compose

from ._types import MarginCorrections, MarginSensitivity, PanelEdges

_ROLE_ATTRIBUTE = "_ggnomics_panel_role"
_DEFAULT_PROBE_MARGIN = 0.05


def tag_panel_role(plot: ggplot, *, role: str) -> ggplot:
    """Mark a freshly built leaf panel with the composition role it fills.

    ``plot`` must be a panel this call's caller just constructed (never a
    caller-supplied plot passed through unchanged), since this mutates it in
    place by setting a private attribute; mutating a just-built object is
    cheaper than copying it and does not affect anything else holding a
    reference. The tag survives ``ggplot.__deepcopy__`` and
    ``Compose.draw()``, so :func:`compute_panel_edges` can recover it from
    the rendered figure's axes. Returns ``plot`` unchanged otherwise, so the
    call composes into the expression that built the panel.
    """

    setattr(plot, _ROLE_ATTRIBUTE, role)
    return plot


def _role_positions(composition: Compose) -> dict[str, tuple[float, float]]:
    """Render a throwaway copy of `composition` and read back tagged panel edges."""

    rendered = deepcopy(composition)
    figure = rendered.draw()
    try:
        figure.savefig(io.BytesIO(), format="png")
        positions: dict[str, tuple[float, float]] = {}
        for plot in rendered.iter_plots_all():
            role = getattr(plot, _ROLE_ATTRIBUTE, None)
            if role is not None:
                for axes in plot.axs:
                    box = axes.get_position()
                    positions.setdefault(role, (float(box.x0), float(box.x1)))
        return positions
    finally:
        plt.close(figure)


def compute_panel_edges(composition: Compose) -> PanelEdges:
    """Render `composition` once and read back each tagged panel's left/right edge.

    `composition` must have had its leaf panels marked with
    :func:`tag_panel_role`; untagged leaves are ignored, and only the first
    panel seen for a repeated role (e.g. several stacked annotations sharing
    role ``"annotation"``) is kept, since those are expected to already
    share an edge. Rendering uses `composition`'s own current theme and
    figure size, exactly as ``Compose.draw()``/``Compose.save()`` would
    without an explicit ``width``/``height`` override, so a correction
    computed from it is calibrated for that size specifically.

    Returns a :class:`~ggnomics.upset.PanelEdges` whose ``by_role`` maps
    each tag to ``(x0, x1)`` in matplotlib figure-fraction units. Consumed
    by :func:`compute_margin_sensitivity` and
    :func:`suggest_left_margin_corrections`.
    """

    return PanelEdges(by_role=_role_positions(composition))


def _non_max_roles(edges: PanelEdges, columns: Sequence[Sequence[str]]) -> set[str]:
    """Roles that are not the widest-left-margin member of their column."""

    candidates: set[str] = set()
    for column in columns:
        present = [role for role in column if role in edges.by_role]
        if len(present) < 2:
            continue
        target = max(edges.by_role[role][0] for role in present)
        candidates.update(role for role in present if edges.by_role[role][0] < target)
    return candidates


def compute_margin_sensitivity(
    composition: Compose,
    *,
    baseline: PanelEdges,
    columns: Sequence[Sequence[str]],
    probe: float = _DEFAULT_PROBE_MARGIN,
) -> MarginSensitivity:
    """Measure how far each under-aligned panel's left edge moves per unit of margin.

    `composition` must have had its leaf panels marked with
    :func:`tag_panel_role`, and `baseline` must be
    :func:`compute_panel_edges` applied to that same `composition` before
    any correction. `columns` groups role names that are meant to share a
    left edge, exactly as in :func:`suggest_left_margin_corrections`. Only
    the roles that are *not* already the widest-left-margin member of their
    column are probed -- bumping a role that will not be corrected would
    change how much room its row's other panel gets, contaminating that
    other panel's own measured slope. `probe` (a ``plot_margin_left`` value,
    in the units plotnine's ``theme(plot_margin_left=...)`` takes) is added
    to every such role at once and the composition is re-rendered; probing
    them together is safe because panels in different rows of the layout do
    not affect each other's margins, and this module never asks two roles
    in the *same* row to be probed together. The left-edge response to
    ``plot_margin_left`` is locally linear but its slope depends on the
    panel's row (its width relative to its neighbour), so it must be
    measured, not assumed.

    Returns a :class:`~ggnomics.upset.MarginSensitivity` whose
    ``slope_by_role`` maps each probed role to
    ``(probed_x0 - baseline_x0) / probe``. Consumed by
    :func:`suggest_left_margin_corrections`.
    """

    if probe <= 0:
        raise ValueError(f"probe must be positive; got {probe}")
    to_probe = _non_max_roles(baseline, columns)
    if not to_probe:
        return MarginSensitivity(slope_by_role={})
    probed = deepcopy(composition)
    _apply_corrections(probed, dict.fromkeys(to_probe, probe))
    probed_positions = _role_positions(probed)
    slopes = {
        role: (probed_positions[role][0] - baseline.by_role[role][0]) / probe
        for role in to_probe
        if role in probed_positions
    }
    return MarginSensitivity(slope_by_role=slopes)


def suggest_left_margin_corrections(
    edges: PanelEdges,
    sensitivity: MarginSensitivity,
    *,
    columns: Sequence[Sequence[str]],
) -> MarginCorrections:
    """Decide the ``plot_margin_left`` addition that aligns each column of roles.

    `edges` and `sensitivity` must come from :func:`compute_panel_edges` and
    :func:`compute_margin_sensitivity` for the same composition. `columns`
    groups role names that are meant to share a left edge (e.g.
    ``[("spacer", "set_size"), ("annotation", "matrix")]``); roles absent
    from `edges` are skipped, and a column with fewer than two present roles
    needs no correction.

    Within each column, every present role except the one with the largest
    measured ``x0`` gets a correction so that
    ``edges.x0 + correction * sensitivity.slope == max(x0 in the column)``;
    a role with a non-positive or missing slope is left uncorrected rather
    than divided by, since its margin does not reliably move its edge.

    Returns a :class:`~ggnomics.upset.MarginCorrections` whose
    ``plot_margin_left`` maps role names needing a correction to the extra
    margin to add; roles needing none are omitted. Consumed by
    :func:`apply_left_margin_corrections`.
    """

    corrections: dict[str, float] = {}
    for column in columns:
        present = [role for role in column if role in edges.by_role]
        if len(present) < 2:
            continue
        target = max(edges.by_role[role][0] for role in present)
        for role in _non_max_roles(edges, [column]):
            slope = sensitivity.slope_by_role.get(role, 0.0)
            if slope > 0:
                corrections[role] = (target - edges.by_role[role][0]) / slope
    return MarginCorrections(plot_margin_left=corrections)


def apply_left_margin_corrections(
    composition: Compose,
    corrections: MarginCorrections,
) -> Compose:
    """Apply a decided margin correction to a composition's tagged panels.

    `composition` must have had its leaf panels marked with
    :func:`tag_panel_role`. Returns a new composition, structurally
    identical to `composition`, where every tagged panel whose role appears
    in `corrections.plot_margin_left` has that amount added to its
    ``plot_margin_left`` theme setting; `composition` itself is not
    mutated. A panel whose role is not in `corrections.plot_margin_left` is
    returned unchanged.
    """

    if not corrections.plot_margin_left:
        return composition
    result = deepcopy(composition)
    _apply_corrections(result, corrections.plot_margin_left)
    return result


def _apply_corrections(node, plot_margin_left) -> None:
    for index, item in enumerate(node):
        if isinstance(item, ggplot):
            role = getattr(item, _ROLE_ATTRIBUTE, None)
            margin = plot_margin_left.get(role)
            if margin is not None:
                node[index] = item + theme(plot_margin_left=margin)
        else:
            _apply_corrections(item, plot_margin_left)
