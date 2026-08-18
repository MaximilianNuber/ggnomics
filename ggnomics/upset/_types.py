"""Shared, immutable result vocabulary for :mod:`ggnomics.upset`."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class IntersectionStatistics:
    """Unfiltered statistics computed from a Boolean membership frame."""

    data: pd.DataFrame
    sets: tuple[str, ...]
    memberships: pd.DataFrame
    element_intersections: pd.Series
    intersection_members: Mapping[str, tuple[str, ...]]
    sizes: pd.DataFrame
    degrees: pd.Series
    set_sizes: pd.Series
    converted_columns: tuple[str, ...] = ()
    size_columns_suffix: str = "_size"


@dataclass(frozen=True)
class IntersectionSelection:
    """Serializable filtering and ordering decision for intersection statistics."""

    intersections: tuple[str, ...]
    sets: tuple[str, ...]
    dropped_sets: tuple[str, ...]
    min_size: int
    max_size: float
    min_degree: int
    max_degree: float
    n_intersections: int | None
    sort_sets: str | bool
    sort_intersections: str | bool
    sort_intersections_by: tuple[str, ...]
    sort_ratio_numerator: str
    sort_ratio_denominator: str
    group_by: str
    mode: str
    origin_id: int


@dataclass(frozen=True)
class UpSetData:
    """Plot-ready UpSet representations with aligned categorical axes."""

    statistics: IntersectionStatistics
    selection: IntersectionSelection
    with_sizes: pd.DataFrame
    presence: pd.DataFrame
    matrix: pd.DataFrame
    matrix_frame: pd.DataFrame
    sizes: pd.DataFrame
    sorted_sets: tuple[str, ...]
    sorted_intersections: tuple[str, ...]
    plot_sets_subset: tuple[str, ...]
    plot_intersections_subset: tuple[str, ...]
    intersection_members: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class ModeSpec:
    """Canonical region-selection mode used by an annotation."""

    mode: str


@dataclass(frozen=True)
class UpSetQuery:
    """One validated set, intersection, or group highlighting query."""

    set: str | None = None
    intersect: tuple[str, ...] | None = None
    group: str | None = None
    only_components: tuple[str, ...] | None = None
    aesthetics: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class UpSetAnnotation:
    """A Plotnine annotation plus metadata needed by the UpSet composer."""

    plot: Any
    kind: str
    mode: str
    options: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    default_y: str | None = None

    def __add__(self, component: Any) -> UpSetAnnotation:
        """Return a copy with a Plotnine component added to its plot."""

        if isinstance(component, ModeSpec):
            return replace(self, mode=component.mode)
        return replace(self, plot=self.plot + component)


@dataclass(frozen=True)
class IntersectionMatrixSpec:
    """Layers and colors for the intersection-membership matrix."""

    geom: Any
    segment: Any
    outline_color: Mapping[str, str]


@dataclass(frozen=True)
class SetSizeSpec:
    """Plot specification for marginal set sizes."""

    mapping: Any
    geom: Any
    position: str
    filter_intersections: bool


@dataclass(frozen=True)
class UpSetStripes:
    """Plot specification for alternating or metadata-mapped matrix stripes."""

    mapping: Any
    geom: Any
    colors: tuple[str, ...] | Mapping[Any, str] | None
    data: pd.DataFrame | None


@dataclass(frozen=True)
class PanelEdges:
    """Measured left/right figure-fraction positions of tagged composition panels."""

    by_role: Mapping[str, tuple[float, float]]


@dataclass(frozen=True)
class MarginSensitivity:
    """Measured figure-fraction left-shift per unit of ``plot_margin_left``, by role."""

    slope_by_role: Mapping[str, float]


@dataclass(frozen=True)
class MarginCorrections:
    """Extra ``plot_margin_left`` to add to each role, in that panel's own theme units."""

    plot_margin_left: Mapping[str, float]


@dataclass(frozen=True)
class VennLayout:
    """Prepared circle, raster-region, and label data for two or three sets."""

    memberships: pd.DataFrame
    circles: pd.DataFrame
    regions: pd.DataFrame
    region_labels: pd.DataFrame
    set_labels: pd.DataFrame
    sets: tuple[str, ...]
    resolution: int


def freeze_mapping(values: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Copy a mapping and expose it through a read-only proxy."""

    return MappingProxyType(dict(values or {}))


def copy_plot(plot: Any, *, data: pd.DataFrame | None = None) -> Any:
    """Deep-copy a Plotnine plot and optionally replace its data."""

    result = deepcopy(plot)
    if data is not None:
        result.data = data
    return result


ThemeCollection = Mapping[str, tuple[Any, ...]]
Labeller = Callable[[str], str]
LayerSequence = Sequence[Any]
