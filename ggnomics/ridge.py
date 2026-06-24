from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_ribbon,
    theme_classic,
    ggtitle,
    scale_fill_brewer,
)
from sklearn.neighbors import KernelDensity


def _kde_density(values: np.ndarray, bandwidth: float, grid: np.ndarray) -> np.ndarray:
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(values[:, None])
    log_dens = kde.score_samples(grid[:, None])
    dens = np.exp(log_dens)
    return dens


def ridge_density(
    df: pd.DataFrame,
    *,
    value: str,
    group: str,
    bandwidth: Optional[float] = None,
    gridsize: int = 200,
    scale: float = 1.0,
    palette: Optional[str] = "Set3",
    title: Optional[str] = None,
):
    """
    Ridge-style density plot built from KDE per group using KernelDensity (scikit-learn).

    This constructs ribbons per group with vertical offsets, avoiding external ridge geoms.
    """
    d = df[[value, group]].dropna().copy()
    groups = d[group].unique()

    # Grid across global range
    vals = d[value].to_numpy()
    vmin, vmax = float(vals.min()), float(vals.max())
    pad = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
    grid = np.linspace(vmin - pad, vmax + pad, gridsize)

    if bandwidth is None:
        # Scott-like rule of thumb
        std = vals.std(ddof=1) if len(vals) > 1 else 1.0
        bandwidth = 1.06 * std * (len(vals) ** (-1 / 5)) if len(vals) > 1 else 0.1
        bandwidth = max(bandwidth, 1e-3)

    records = []
    for idx, g in enumerate(sorted(groups)):
        v = d.loc[d[group] == g, value].to_numpy()
        dens = _kde_density(v, bandwidth=bandwidth, grid=grid)
        dens_scaled = dens * scale
        base = float(idx)
        rec = pd.DataFrame({
            "x": grid,
            "ymin": base,
            "ymax": base + dens_scaled,
            group: g,
        })
        records.append(rec)

    plot_df = pd.concat(records, ignore_index=True)

    p = (
        ggplot(plot_df)
        + aes(x="x", ymin="ymin", ymax="ymax", fill=group)
        + geom_ribbon(alpha=0.8)
        + theme_classic()
    )
    if palette:
        p = p + scale_fill_brewer(type="qual", palette=palette)
    if title:
        p = p + ggtitle(title)
    return p
