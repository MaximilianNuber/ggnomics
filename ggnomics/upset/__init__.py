"""Plotnine-native ComplexUpset-style set visualization.

The public API mirrors the exported names of the R package ComplexUpset while
also exposing the granular ``compute_intersections`` → ``select_intersections``
→ ``apply_intersection_selection`` workflow.
"""

from ._types import (
    IntersectionMatrixSpec,
    IntersectionSelection,
    IntersectionStatistics,
    MarginCorrections,
    MarginSensitivity,
    ModeSpec,
    PanelEdges,
    SetSizeSpec,
    UpSetAnnotation,
    UpSetData,
    UpSetQuery,
    UpSetStripes,
    VennLayout,
)
from .alignment import (
    apply_left_margin_corrections,
    compute_margin_sensitivity,
    compute_panel_edges,
    suggest_left_margin_corrections,
    tag_panel_role,
)
from .annotations import intersection_ratio, intersection_size, upset_annotate
from .examples import create_upset_abc_example
from .intersections import (
    apply_intersection_selection,
    compute_intersections,
    report_dropped_sets,
    select_intersections,
    select_mode_observations,
    upset_data,
)
from .matrix import intersection_matrix
from .modes import (
    aes_percentage,
    get_size_mode,
    normalize_mode,
    reverse_log_trans,
    upset_mode,
    upset_text_percentage,
)
from .plot import compose_upset, upset
from .queries import upset_query
from .set_size import upset_set_size
from .stripes import upset_stripes
from .testing import (
    adjust_intersection_tests,
    compare_between_intersections,
    compute_intersection_tests,
    upset_test,
)
from .themes import upset_default_themes, upset_modify_themes, upset_themes
from .venn import (
    arrange_venn,
    compute_venn_layout,
    geom_venn_circle,
    geom_venn_label_region,
    geom_venn_label_set,
    geom_venn_region,
    scale_color_venn_mix,
    scale_fill_venn_mix,
)

__all__ = [
    "IntersectionMatrixSpec",
    "IntersectionSelection",
    "IntersectionStatistics",
    "MarginCorrections",
    "MarginSensitivity",
    "ModeSpec",
    "PanelEdges",
    "SetSizeSpec",
    "UpSetAnnotation",
    "UpSetData",
    "UpSetQuery",
    "UpSetStripes",
    "VennLayout",
    "compute_intersections",
    "select_intersections",
    "apply_intersection_selection",
    "report_dropped_sets",
    "select_mode_observations",
    "upset_data",
    "aes_percentage",
    "get_size_mode",
    "normalize_mode",
    "reverse_log_trans",
    "upset_mode",
    "upset_text_percentage",
    "intersection_size",
    "intersection_ratio",
    "upset_annotate",
    "intersection_matrix",
    "upset_set_size",
    "upset_query",
    "upset_stripes",
    "upset_themes",
    "upset_default_themes",
    "upset_modify_themes",
    "tag_panel_role",
    "compute_panel_edges",
    "compute_margin_sensitivity",
    "suggest_left_margin_corrections",
    "apply_left_margin_corrections",
    "compose_upset",
    "upset",
    "compute_intersection_tests",
    "adjust_intersection_tests",
    "compare_between_intersections",
    "upset_test",
    "create_upset_abc_example",
    "compute_venn_layout",
    "arrange_venn",
    "geom_venn_circle",
    "geom_venn_region",
    "geom_venn_label_region",
    "geom_venn_label_set",
    "scale_color_venn_mix",
    "scale_fill_venn_mix",
]
