from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_histogram,
    theme_classic,
    ggtitle,
    scale_color_gradient,
)


def _autosize(n: int, max_size: float = 1.5, min_size: float = 0.2, ref: int = 5000) -> float:
    if n <= 0:
        return max_size
    s = max_size * np.sqrt(ref / float(n))
    return float(np.clip(s, min_size, max_size))


def qc_scatter(
    df: pd.DataFrame,
    x: str = "n_counts",
    y: str = "n_genes",
    color: Optional[str] = None,
    size: Optional[float] = None,
    alpha: float = 0.6,
    title: Optional[str] = None,
):
    """
    QC scatter (e.g., n_counts vs n_genes), optionally colored by numeric metric (e.g., mito_fraction).
    """
    aes_kwargs = {"x": x, "y": y}
    is_numeric_color = False
    if color is not None and color in df.columns:
        aes_kwargs["color"] = color
        is_numeric_color = pd.api.types.is_numeric_dtype(df[color])

    if size is None:
        size = _autosize(len(df))

    p = ggplot(df) + aes(**aes_kwargs) + geom_point(size=size, alpha=alpha) + theme_classic()
    if title:
        p = p + ggtitle(title)
    if color is not None and is_numeric_color:
        p = p + scale_color_gradient(low="#f7f7f7", high="#08306B")
    return p


def qc_histogram(
    df: pd.DataFrame,
    value: str,
    bins: int = 50,
    alpha: float = 0.8,
    title: Optional[str] = None,
):
    p = ggplot(df) + aes(x=value) + geom_histogram(bins=bins, alpha=alpha) + theme_classic()
    if title:
        p = p + ggtitle(title)
    return p
