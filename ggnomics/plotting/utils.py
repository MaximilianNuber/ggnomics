from typing import Optional, Dict
import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_point,
    theme_classic,
    scale_color_manual,
    scale_fill_manual,
    ggtitle,
)

def plot_reduced_dim(
    df: pd.DataFrame,
    x: str,
    y: str,
    color: Optional[str] = None,
    fill: Optional[str] = None,
    size: Optional[float] = None,
    stroke: float = 0.0,
    alpha: float = 1.0,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
    autosize_min: float = 0.2,
    autosize_max: float = 1.5,
    autosize_ref: int = 5_000,
):
    """
    Generic 2D embedding plot for a DataFrame.

    Parameters
    ----------
    df : DataFrame
        Contains columns for x, y, and (optionally) color/fill.
    x, y : str
        Column names for the reduced dimensions.
    color : str, optional
        Column name to map to color aesthetic.
    fill : str, optional
        Column name to map to fill aesthetic.
    size : float
        Point size.
    stroke : float
        Point stroke.
    alpha : float
        Point alpha.
    palette : dict, optional
        Mapping {category: color} to use with scale_color_manual/scale_fill_manual.
    title : str, optional
        Plot title.

    Returns
    -------
    plotnine.ggplot
    """
    aes_kwargs = {"x": x, "y": y}
    if color is not None:
        aes_kwargs["color"] = color
    if fill is not None:
        aes_kwargs["fill"] = fill

    # Auto-size points based on number of rows when size not provided.
    n_points = len(df)
    if size is None:
        if n_points <= 0:
            computed_size = autosize_max
        else:
            computed_size = autosize_max * np.sqrt(autosize_ref / float(n_points))
        size = float(np.clip(computed_size, autosize_min, autosize_max))

    p = (
        ggplot(df)
        + aes(**aes_kwargs)
        + geom_point(size=size, stroke=stroke, alpha=alpha)
        + theme_classic()
    )

    if palette is not None and color is not None:
        p = p + scale_color_manual(
            breaks=list(palette.keys()),
            values=list(palette.values()),
        )

    if palette is not None and fill is not None:
        p = p + scale_fill_manual(
            breaks=list(palette.keys()),
            values=list(palette.values()),
        )

    if title is not None:
        p = p + ggtitle(title)

    return p
