from typing import Optional

import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_bar,
    position_stack,
    theme_classic,
    ggtitle,
    labs,
)


def cluster_composition_barplot(
    df: pd.DataFrame,
    *,
    cluster_col: str = "cluster",
    group_col: str = "condition",
    normalize: bool = True,
    title: Optional[str] = None,
):
    """
    Stacked bar plot of sample/condition composition across clusters.
    If normalize=True, shows fraction within each cluster.
    """
    d = df[[cluster_col, group_col]].copy()
    counts = d.value_counts([cluster_col, group_col]).rename("n").reset_index()
    if normalize:
        counts["frac"] = counts.groupby(cluster_col)["n"].transform(lambda x: x / x.sum())
        y = "frac"
    else:
        y = "n"

    p = (
        ggplot(counts)
        + aes(x=cluster_col, y=y, fill=group_col)
        + geom_bar(stat="identity", position=position_stack())
        + theme_classic()
    )
    if normalize:
        p = p + labs(y="fraction")
    if title:
        p = p + ggtitle(title)
    return p
