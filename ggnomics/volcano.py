from typing import Optional, Dict

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_vline,
    geom_hline,
    theme_classic,
    ggtitle,
    scale_color_manual,
)


def volcano_plot(
    df: pd.DataFrame,
    *,
    log2fc: str = "log2fc",
    pval: Optional[str] = None,
    padj: Optional[str] = None,
    fdr_thresh: float = 0.05,
    lfc_thresh: float = 1.0,
    alpha: float = 0.6,
    palette: Optional[Dict[str, str]] = None,
    title: Optional[str] = None,
):
    """
    Volcano plot: log2FC vs -log10(p-value/FDR) with highlight by thresholds.
    Prefers `padj` if provided, else uses `pval`.
    """
    if padj is None and pval is None:
        raise ValueError("Provide either padj or pval column name.")

    ycol = padj if (padj is not None and padj in df.columns) else pval
    d = df.copy()
    d["__neglog10p__"] = -np.log10(d[ycol].astype(float).replace(0, np.nextafter(0, 1)))

    def _status(row):
        sig = (row[ycol] <= fdr_thresh) if ycol else False
        if sig and row[log2fc] >= lfc_thresh:
            return "up"
        if sig and row[log2fc] <= -lfc_thresh:
            return "down"
        return "ns"

    d["__status__"] = d.apply(_status, axis=1)

    if palette is None:
        palette = {"ns": "#bdbdbd", "up": "#e41a1c", "down": "#377eb8"}

    # Auto point size similar to reduced-dim plots
    n = len(d)
    size = float(np.clip(1.5 * np.sqrt(5000.0 / max(n, 1)), 0.2, 1.5))

    p = (
        ggplot(d)
        + aes(x=log2fc, y="__neglog10p__", color="__status__")
        + geom_point(alpha=alpha, size=size)
        + geom_vline(xintercept=[-lfc_thresh, lfc_thresh], linetype="dashed", alpha=0.6)
        + geom_hline(yintercept=-np.log10(fdr_thresh), linetype="dashed", alpha=0.6)
        + scale_color_manual(values=palette)
        + theme_classic()
    )

    if title:
        p = p + ggtitle(title)
    return p
