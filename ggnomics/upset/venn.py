"""Plotnine-native Venn helpers mirroring ComplexUpset's exported names."""

from __future__ import annotations

from math import tau
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from plotnine import (
    aes,
    geom_label,
    geom_path,
    geom_tile,
    scale_color_manual,
    scale_fill_manual,
)

from ._types import VennLayout


def _membership_frame(
    data: pd.DataFrame,
    sets: Sequence[str] | None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas.DataFrame")
    selected = tuple(sets) if sets is not None else tuple(
        column for column in data.columns if pd.api.types.is_bool_dtype(data[column])
    )
    if len(selected) not in {2, 3, 4}:
        raise ValueError(
            "Venn geometry supports exactly two, three, or four sets; "
            f"selected {len(selected)}"
        )
    missing = [name for name in selected if name not in data.columns]
    if missing:
        raise KeyError(f"sets missing from data: {missing}")
    frame = pd.DataFrame(index=data.index)
    for name in selected:
        values = data[name]
        if values.isna().any():
            raise ValueError(f"data[{name!r}] contains missing memberships")
        unique = set(pd.unique(values))
        if not unique.issubset({False, True, 0, 1, 0.0, 1.0}):
            raise TypeError(f"data[{name!r}] must be Boolean or 0/1")
        frame[name] = values.astype(bool)
    return frame, selected


def _circle_centers(count: int, radius: float) -> list[tuple[float, float]]:
    if count == 2:
        return [(-0.45 * radius, 0.0), (0.45 * radius, 0.0)]
    if count == 3:
        return [
            (-0.48 * radius, 0.28 * radius),
            (0.48 * radius, 0.28 * radius),
            (0.0, -0.50 * radius),
        ]
    return [
        (-0.48 * radius, 0.28 * radius),
        (0.48 * radius, 0.28 * radius),
        (-0.48 * radius, -0.42 * radius),
        (0.48 * radius, -0.42 * radius),
    ]


def _region_name(members: Sequence[str]) -> str:
    return "&".join(members) if members else "(none)"


def compute_venn_layout(
    data: pd.DataFrame,
    *,
    sets: Sequence[str] | None = None,
    radius: float = 1.5,
    max_iterations: int = 10,
    outwards_adjust: float = 1.3,
    repeat_in_intersections: bool = False,
    starting_grid_size: int | str = "auto",
) -> VennLayout:
    """Compute circle, raster-region, and label data for two to four sets."""

    del max_iterations, repeat_in_intersections
    if radius <= 0:
        raise ValueError("radius must be positive")
    memberships, selected = _membership_frame(data, sets)
    resolution = 180 if starting_grid_size == "auto" else int(starting_grid_size)
    if resolution < 25:
        raise ValueError("starting_grid_size must be 'auto' or an integer >= 25")
    centers = _circle_centers(len(selected), radius)
    angles = np.linspace(0, tau, 240, endpoint=True)
    circle_rows: list[dict[str, Any]] = []
    for name, (x0, y0) in zip(selected, centers, strict=True):
        circle_rows.extend(
            {
                "set": name,
                "x": x0 + radius * np.cos(angle),
                "y": y0 + radius * np.sin(angle),
                "radius": radius,
            }
            for angle in angles
        )
    circles = pd.DataFrame(circle_rows)

    extent = radius * 1.65
    axis = np.linspace(-extent, extent, resolution)
    xx, yy = np.meshgrid(axis, axis)
    membership_grid = np.column_stack(
        [
            ((xx - x0) ** 2 + (yy - y0) ** 2 <= radius**2).ravel()
            for x0, y0 in centers
        ]
    )
    inside = membership_grid.any(axis=1)
    grid_members = [
        tuple(name for name, present in zip(selected, row, strict=True) if present)
        for row in membership_grid[inside]
    ]
    regions = pd.DataFrame(
        {
            "x": xx.ravel()[inside],
            "y": yy.ravel()[inside],
            "region": [_region_name(members) for members in grid_members],
            "members": grid_members,
        }
    )
    cell_width = float(axis[1] - axis[0])
    regions["cell_width"] = cell_width

    observed_members = [
        tuple(name for name, present in zip(selected, row, strict=True) if present)
        for row in memberships.to_numpy(dtype=bool, copy=False)
    ]
    counts = pd.Series(
        [_region_name(members) for members in observed_members]
    ).value_counts()
    labels = (
        regions.groupby("region", observed=True)[["x", "y"]]
        .mean()
        .reset_index()
    )
    labels["count"] = labels["region"].map(counts).fillna(0).astype(int)

    center_x = np.mean([center[0] for center in centers])
    center_y = np.mean([center[1] for center in centers])
    set_label_rows = []
    for name, (x0, y0) in zip(selected, centers, strict=True):
        dx, dy = x0 - center_x, y0 - center_y
        norm = np.hypot(dx, dy) or 1.0
        set_label_rows.append(
            {
                "set": name,
                "x": x0 + outwards_adjust * radius * dx / norm,
                "y": y0 + outwards_adjust * radius * dy / norm,
            }
        )
    return VennLayout(
        memberships=memberships,
        circles=circles,
        regions=regions,
        region_labels=labels,
        set_labels=pd.DataFrame(set_label_rows),
        sets=selected,
        resolution=resolution,
    )


def arrange_venn(
    data: pd.DataFrame,
    *,
    sets: Sequence[str] | None = None,
    radius: float = 1.5,
    max_iterations: int = 10,
    verbose: bool = False,
    outwards_adjust: float = 1.3,
    extract_sets: bool = False,
    extract_regions: bool = False,
    repeat_in_intersections: bool = False,
    starting_grid_size: int | str = "auto",
) -> VennLayout:
    """Prepare a stable VennLayout using ComplexUpset-compatible arguments."""

    del verbose, extract_sets, extract_regions
    return compute_venn_layout(
        data,
        sets=sets,
        radius=radius,
        max_iterations=max_iterations,
        outwards_adjust=outwards_adjust,
        repeat_in_intersections=repeat_in_intersections,
        starting_grid_size=starting_grid_size,
    )


def _layout(data: pd.DataFrame | VennLayout, sets=None, **kwargs) -> VennLayout:
    if isinstance(data, VennLayout):
        return data
    return compute_venn_layout(data, sets=sets, **kwargs)


def _merge_aes(base: Mapping[str, Any], mapping) -> Any:
    values = dict(base)
    if mapping is not None:
        values.update(dict(mapping))
    return aes(**values)


def geom_venn_circle(
    data: pd.DataFrame | VennLayout,
    *,
    mapping=None,
    sets: Sequence[str] | None = None,
    radius: float = 1.5,
    resolution: int = 100,
    size: float = 0.8,
    color: str = "black",
    **geom_kwargs: Any,
):
    """Construct circle-outline layers for a Venn diagram."""

    del resolution
    prepared = _layout(data, sets=sets, radius=radius)
    return geom_path(
        data=prepared.circles,
        mapping=_merge_aes({"x": "x", "y": "y", "group": "set"}, mapping),
        inherit_aes=False,
        size=size,
        color=color,
        **geom_kwargs,
    )


def geom_venn_region(
    data: pd.DataFrame | VennLayout,
    *,
    mapping=None,
    sets: Sequence[str] | None = None,
    resolution: int = 250,
    **geom_kwargs: Any,
):
    """Construct rasterized region layers for a Venn diagram."""

    prepared = _layout(data, sets=sets, starting_grid_size=resolution)
    width = float(prepared.regions["cell_width"].iloc[0])
    return geom_tile(
        data=prepared.regions,
        mapping=_merge_aes({"x": "x", "y": "y", "fill": "region"}, mapping),
        inherit_aes=False,
        width=width,
        height=width,
        **geom_kwargs,
    )


def geom_venn_label_region(
    data: pd.DataFrame | VennLayout,
    *,
    mapping=None,
    sets: Sequence[str] | None = None,
    outwards_adjust: float = 1.3,
    fill: str = "white",
    size: float = 5,
    label_size: float = 0,
    **geom_kwargs: Any,
):
    """Construct count-label layers for Venn regions."""

    prepared = _layout(data, sets=sets, outwards_adjust=outwards_adjust)
    return geom_label(
        data=prepared.region_labels,
        mapping=_merge_aes({"x": "x", "y": "y", "label": "count"}, mapping),
        inherit_aes=False,
        fill=fill,
        size=size,
        label_size=label_size,
        **geom_kwargs,
    )


def geom_venn_label_set(
    data: pd.DataFrame | VennLayout,
    *,
    mapping=None,
    sets: Sequence[str] | None = None,
    outwards_adjust: float = 2.5,
    fill: str = "white",
    size: float = 5,
    label_size: float = 0,
    **geom_kwargs: Any,
):
    """Construct name-label layers for Venn sets."""

    prepared = _layout(data, sets=sets, outwards_adjust=outwards_adjust)
    return geom_label(
        data=prepared.set_labels,
        mapping=_merge_aes({"x": "x", "y": "y", "label": "set"}, mapping),
        inherit_aes=False,
        fill=fill,
        size=size,
        label_size=label_size,
        **geom_kwargs,
    )


def _mixed_region_colors(
    prepared: VennLayout,
    colors: Sequence[str] | Mapping[str, str],
    *,
    highlight: Sequence[Sequence[str]] | None,
    active_color: str,
    inactive_color: str | None,
) -> dict[str, str | None]:
    if isinstance(colors, Mapping):
        set_colors = dict(colors)
    else:
        palette = tuple(colors)
        if len(palette) < len(prepared.sets):
            raise ValueError("colors must contain at least one color per set")
        set_colors = dict(zip(prepared.sets, palette, strict=False))
    highlighted = None if highlight is None else {
        tuple(name for name in prepared.sets if name in set(members))
        for members in highlight
    }
    values: dict[str, str | None] = {}
    for members in dict.fromkeys(prepared.regions["members"]):
        name = _region_name(members)
        if highlighted is not None:
            values[name] = active_color if members in highlighted else inactive_color
        elif not members:
            values[name] = None
        else:
            rgb = np.mean([to_rgb(set_colors[member]) for member in members], axis=0)
            values[name] = to_hex(rgb)
    return values


def scale_color_venn_mix(
    data: pd.DataFrame | VennLayout,
    *,
    sets: Sequence[str] | None = None,
    colors: Sequence[str] | Mapping[str, str] = ("red", "blue", "green"),
    na_value: str = "#666666",
    highlight: Sequence[Sequence[str]] | None = None,
    active_color: str = "orange",
    inactive_color: str | None = None,
    **scale_kwargs: Any,
):
    """Construct a mixed-region manual color scale."""

    prepared = _layout(data, sets=sets)
    values = _mixed_region_colors(
        prepared,
        colors,
        highlight=highlight,
        active_color=active_color,
        inactive_color=inactive_color,
    )
    return scale_color_manual(values=values, na_value=na_value, **scale_kwargs)


def scale_fill_venn_mix(
    data: pd.DataFrame | VennLayout,
    *,
    sets: Sequence[str] | None = None,
    colors: Sequence[str] | Mapping[str, str] = ("red", "blue", "green"),
    na_value: str | None = None,
    highlight: Sequence[Sequence[str]] | None = None,
    active_color: str = "orange",
    inactive_color: str | None = None,
    **scale_kwargs: Any,
):
    """Construct a mixed-region manual fill scale."""

    prepared = _layout(data, sets=sets)
    values = _mixed_region_colors(
        prepared,
        colors,
        highlight=highlight,
        active_color=active_color,
        inactive_color=inactive_color,
    )
    return scale_fill_manual(values=values, na_value=na_value, **scale_kwargs)
