"""Plotnine-native ComplexUpset-style set visualization.

The public API mirrors the exported names of the R package ComplexUpset while
also exposing the granular ``compute_intersections`` → ``select_intersections``
→ ``apply_intersection_selection`` workflow.
"""

from ._types import IntersectionMatrixSpec
from ._types import IntersectionSelection
from ._types import IntersectionStatistics
from ._types import ModeSpec
from ._types import SetSizeSpec
from ._types import UpSetAnnotation
from ._types import UpSetData
from ._types import UpSetQuery
from ._types import UpSetStripes
from ._types import VennLayout

from .intersections import compute_intersections
from .intersections import select_intersections
from .intersections import apply_intersection_selection
from .intersections import report_dropped_sets
from .intersections import upset_data

from .modes import aes_percentage
from .modes import get_size_mode
from .modes import normalize_mode
from .modes import reverse_log_trans
from .modes import upset_mode
from .modes import upset_text_percentage

from .annotations import intersection_size
from .annotations import intersection_ratio
from .annotations import upset_annotate

from .matrix import intersection_matrix
from .set_size import upset_set_size
from .queries import upset_query
from .stripes import upset_stripes

from .themes import upset_themes
from .themes import upset_default_themes
from .themes import upset_modify_themes

from .plot import compose_upset
from .plot import upset

from .testing import compute_intersection_tests
from .testing import adjust_intersection_tests
from .testing import compare_between_intersections
from .testing import upset_test

from .examples import create_upset_abc_example

from .venn import compute_venn_layout
from .venn import arrange_venn
from .venn import geom_venn_circle
from .venn import geom_venn_region
from .venn import geom_venn_label_region
from .venn import geom_venn_label_set
from .venn import scale_color_venn_mix
from .venn import scale_fill_venn_mix

__all__ = [
    "IntersectionMatrixSpec",
    "IntersectionSelection",
    "IntersectionStatistics",
    "ModeSpec",
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
