"""Default component themes for native Plotnine UpSet compositions."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

from plotnine import element_blank, theme, theme_minimal

from ._types import ThemeCollection


def _new_upset_themes() -> dict[str, tuple[Any, ...]]:
    return {
        "intersections_matrix": (
            theme_minimal(),
            theme(
                axis_text_x=element_blank(),
                axis_ticks_major_x=element_blank(),
                axis_text_y=element_blank(),
                axis_ticks_major_y=element_blank(),
                axis_title_y=element_blank(),
                panel_grid=element_blank(),
            ),
        ),
        "Intersection size": (
            theme_minimal(),
            theme(
                axis_text_x=element_blank(),
                axis_title_x=element_blank(),
                panel_grid_major_x=element_blank(),
                panel_grid_minor_x=element_blank(),
            ),
        ),
        "overall_sizes": (
            theme_minimal(),
            theme(
                axis_title_y=element_blank(),
                panel_grid_major_y=element_blank(),
                panel_grid_minor_y=element_blank(),
            ),
        ),
        "default": (
            theme_minimal(),
            theme(
                axis_text_x=element_blank(),
                axis_title_x=element_blank(),
                panel_grid_major_x=element_blank(),
                panel_grid_minor_x=element_blank(),
            ),
        ),
    }


upset_themes: ThemeCollection = MappingProxyType(_new_upset_themes())


def upset_default_themes(**theme_kwargs: Any) -> ThemeCollection:
    """Return fresh defaults with one theme modification applied everywhere."""

    addition = theme(**theme_kwargs)
    return MappingProxyType({name: (*components, addition) for name, components in _new_upset_themes().items()})


def upset_modify_themes(
    to_update: Mapping[str, Any | Sequence[Any]],
) -> ThemeCollection:
    """Return fresh defaults with component-specific theme additions."""

    if not isinstance(to_update, Mapping):
        raise TypeError("to_update must be a mapping from component names to themes")
    result = _new_upset_themes()
    for name, additions in to_update.items():
        if name not in result:
            raise KeyError(f"unknown theme component {name!r}; expected one of {sorted(result)}")
        if isinstance(additions, (list, tuple)):
            result[name] = (*result[name], *additions)
        else:
            result[name] = (*result[name], additions)
    return MappingProxyType(result)
