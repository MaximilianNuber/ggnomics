"""Shared utility functions for ggnomics plotting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pandas as pd
from plotnine import (
    scale_color_manual,
    scale_fill_manual,
)

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

    plot: ggplot
    matrix: Optional[pd.DataFrame] = field(default=None)

    def _repr_html_(self) -> str:
        return self.plot._repr_html_()

    def savefig(self, filename: str, **kwargs) -> None:
        self.plot.save(filename, **kwargs)


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


def display_dtype(categories) -> pd.CategoricalDtype:
    """Categorical dtype that pins display order without declaring an ordinal variable.

    plotnine chooses a default scale from the column's dtype: an **ordered**
    Categorical gets ``scale_*_ordinal`` (a viridis ramp), an unordered one
    gets ``scale_*_discrete`` (plotnine's own hue palette). ggnomics builds
    categoricals only to fix the order categories appear in on axes, facets
    and legends -- never to claim the variable is genuinely ordinal -- so
    using ``ordered=True`` here would silently replace the user's default
    palette with viridis. An unordered Categorical preserves the declared
    ``categories`` order in every place plotnine uses it, so ordering costs
    nothing.

    Args:
        categories: Categories in the order they should be displayed.

    Returns:
        An unordered :class:`pandas.CategoricalDtype`.
    """
    return pd.CategoricalDtype(list(categories), ordered=False)


def display_categorical(values, categories) -> pd.Categorical:
    """Build a display-ordered :class:`pandas.Categorical`.

    The :class:`pandas.Categorical` counterpart of :func:`display_dtype`;
    see there for why the result is deliberately unordered.

    Args:
        values: Values to encode.
        categories: Categories in the order they should be displayed.

    Returns:
        An unordered :class:`pandas.Categorical`.
    """
    return pd.Categorical(values, dtype=display_dtype(categories))


def add_scale(plot: ggplot, scale) -> ggplot:
    """Add ``scale`` to ``plot`` if it is not ``None``, else return ``plot`` unchanged.

    Pairs with :func:`color_scale` and other helpers that return ``None`` to
    mean "impose nothing here" rather than a scale object.
    """
    return plot if scale is None else plot + scale


def color_scale(
    col: pd.Series,
    palette=None,
    type_: str = "color",
):
    """Return an explicit plotnine color scale for ``col``, or ``None``.

    ggnomics never imposes its own color scheme: this only returns a scale
    when the caller supplied a ``palette``, resolved against the column's
    own observed categories (declared order for a pandas Categorical,
    first-appearance order otherwise). In every other case — a numeric
    column, or a categorical column with no ``palette`` — it returns
    ``None`` so the caller adds no scale at all and plotnine's own default
    (continuous or discrete) applies untouched.

    Args:
        col: The Series being mapped to color.
        palette: Optional ``{category: hex_color}`` dict.
        type_: ``"color"`` or ``"fill"`` — selects which scale family to use.

    Returns:
        A plotnine scale object, or ``None`` if there is nothing to impose.
    """
    is_numeric = pd.api.types.is_numeric_dtype(col)

    if is_numeric or palette is None:
        return None

    # An explicit palette is resolved against the column's own observed
    # categories, filling any gaps rather than silently producing a broken
    # legend for uncovered categories.
    from .palettes import resolve_palette

    resolved = resolve_palette(col, palette=palette)
    breaks = list(resolved.keys())
    values = list(resolved.values())
    if type_ == "fill":
        return scale_fill_manual(breaks=breaks, values=values)
    return scale_color_manual(breaks=breaks, values=values)


def resolve_manual_colors(
    categories: List,
    overrides: Optional[Dict] = None,
) -> Optional[Dict]:
    """Resolve a ``{category: color}`` mapping for a fixed set of categories.

    Used where a plot has semantic per-category colors (e.g. up/down/ns)
    exposed as individual keyword arguments rather than a single ``palette``
    dict. Categories the caller didn't override fall back to the same hue
    colors plotnine's default discrete scale would assign them, so a partial
    override doesn't degrade the rest into flat grey placeholders.

    Args:
        categories: Ordered, complete list of category values.
        overrides: ``{category: color}`` for any subset of ``categories``.
            Entries with a ``None`` value are treated as not overridden
            (this lets callers pass a fixed set of keys defaulting to
            ``None`` without pre-filtering).

    Returns:
        ``None`` if no override value is set — the caller should skip
        adding an explicit manual scale and let plotnine's default apply.
        Otherwise a complete ``{category: color}`` mapping.
    """
    if not overrides or not any(value is not None for value in overrides.values()):
        return None

    from mizani.palettes import hue_pal

    defaults = hue_pal()(len(categories))
    return {
        category: (overrides.get(category) if overrides.get(category) is not None else default)
        for category, default in zip(categories, defaults)
    }


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
