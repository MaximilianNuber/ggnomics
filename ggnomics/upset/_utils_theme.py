"""Component-theme application shared by every panel builder."""

from __future__ import annotations

from ._types import ThemeCollection


def apply_component_theme(plot, *, component: str, themes: ThemeCollection):
    """Add one named component's theme layers to a panel plot.

    ``themes`` must be a :data:`~ggnomics.upset.ThemeCollection` such as
    :data:`~ggnomics.upset.upset_themes`. Looks up ``themes[component]``,
    falling back to ``themes["default"]`` when the component has no entry,
    and adds each layer in order. Returns a plot of the same type as ``plot``
    with the theme layers appended; does not mutate ``plot``. Consumed by
    every panel builder as its last step before returning.
    """

    components = themes.get(component, themes.get("default", ()))
    for addition in components:
        plot = plot + addition
    return plot
