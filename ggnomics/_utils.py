"""Shared utility functions for ggnomics plotting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from plotnine import ggplot


@dataclass
class HeatmapResult:
    """Container returned by heatmap-style functions that also carry the matrix.

    Attributes:
        plot: The underlying plotnine ggplot object.
        matrix: Optional DataFrame of the values shown in the heatmap
                (e.g. an overlap-coefficient matrix or mean-expression matrix).
    """

    plot: "ggplot"
    matrix: Optional[pd.DataFrame] = field(default=None)

    def _repr_html_(self) -> str:
        return self.plot._repr_html_()

    def savefig(self, filename: str, **kwargs) -> None:
        self.plot.save(filename, **kwargs)
from plotnine import (
    scale_color_manual,
    scale_fill_manual,
    scale_color_brewer,
    scale_fill_brewer,
    scale_color_cmap,
    scale_fill_cmap,
)


def adaptive_size(
    n: int,
    size_max: float = 1.5,
    size_min: float = 0.2,
    ref: int = 5_000,
) -> float:
    """Seurat-style adaptive point size: more cells → smaller points.

    Args:
        n: Number of cells/points to render.
        size_max: Maximum point size (used for very small datasets).
        size_min: Minimum point size (floor for very large datasets).
        ref: Reference cell count at which ``size_max`` is returned.

    Returns:
        Clipped point size as a float.
    """
    if n <= 0:
        return float(size_max)
    computed = size_max * np.sqrt(ref / float(n))
    return float(np.clip(computed, size_min, size_max))


def adaptive_stroke(
    n: int,
    stroke_max: float = 0.3,
    ref: int = 5_000,
) -> float:
    """Stroke width that shrinks proportionally to cell count.

    Args:
        n: Number of cells/points.
        stroke_max: Maximum stroke width.
        ref: Reference count at which ``stroke_max`` is returned.

    Returns:
        Clipped stroke width.
    """
    if n <= 0:
        return float(stroke_max)
    computed = stroke_max * np.sqrt(ref / float(n))
    return float(np.clip(computed, 0.0, stroke_max))


def color_scale(
    col: pd.Series,
    palette=None,
    type_: str = "color",
):
    """Return the most appropriate plotnine color scale for ``col``.

    Decision logic:
    - Numeric column → continuous viridis scale.
    - Categorical/object column + ``palette`` dict → ``scale_*_manual``.
    - Categorical/object column without palette → qualitative Brewer "Set2".

    Args:
        col: The Series being mapped to color.
        palette: Optional ``{category: hex_color}`` dict.
        type_: ``"color"`` or ``"fill"`` — selects which scale family to use.

    Returns:
        A plotnine scale object.
    """
    is_numeric = pd.api.types.is_numeric_dtype(col)

    if is_numeric:
        if type_ == "fill":
            return scale_fill_cmap(cmap_name="viridis")
        return scale_color_cmap(cmap_name="viridis")

    # Categorical path
    if palette is not None:
        breaks = list(palette.keys())
        values = list(palette.values())
        if type_ == "fill":
            return scale_fill_manual(breaks=breaks, values=values)
        return scale_color_manual(breaks=breaks, values=values)

    # Default qualitative palette
    if type_ == "fill":
        return scale_fill_brewer(type="qual", palette="Set2")
    return scale_color_brewer(type="qual", palette="Set2")


def to_long(
    df: pd.DataFrame,
    id_vars: List[str],
    value_vars: List[str],
    var_name: str = "feature",
    value_name: str = "expression",
) -> pd.DataFrame:
    """Melt a wide DataFrame to long format for expression plotting.

    Args:
        df: Wide-format DataFrame.
        id_vars: Columns to keep as identifier variables.
        value_vars: Columns to melt (feature expression columns).
        var_name: Name for the new variable column.
        value_name: Name for the new value column.

    Returns:
        Long-format DataFrame.
    """
    return df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
    )
