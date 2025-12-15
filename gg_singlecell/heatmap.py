from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_tile,
    theme_classic,
    ggtitle,
    scale_fill_gradient,
    theme,
    element_text,
)


def heatmap_long(
    df_long: pd.DataFrame,
    *,
    x: str,
    y: str,
    value: str,
    low: str = "#F7FBFF",
    high: str = "#082B6B",
    title: Optional[str] = None,
    x_text_angle: int = 90,
):
    """
    Heatmap from a tidy (long) DataFrame with x, y and value columns.
    """
    p = (
        ggplot(df_long)
        + aes(x=x, y=y, fill=value)
        + geom_tile()
        + scale_fill_gradient(low=low, high=high)
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=x_text_angle, ha="right"))
    )
    if title:
        p = p + ggtitle(title)
    return p


def heatmap_from_matrix(
    mat: np.ndarray | pd.DataFrame,
    row_names: Optional[Sequence[str]] = None,
    col_names: Optional[Sequence[str]] = None,
    *,
    zscore_rows: bool = False,
    title: Optional[str] = None,
):
    """
    Convenience: convert (rows × cols) matrix to long format and plot.
    """
    if isinstance(mat, pd.DataFrame):
        data = mat.values
        row_names = list(mat.index) if row_names is None else row_names
        col_names = list(mat.columns) if col_names is None else col_names
    else:
        data = np.asarray(mat)
        if row_names is None:
            row_names = [f"r{i}" for i in range(data.shape[0])]
        if col_names is None:
            col_names = [f"c{j}" for j in range(data.shape[1])]

    if zscore_rows:
        mu = data.mean(axis=1, keepdims=True)
        sd = data.std(axis=1, keepdims=True) + 1e-9
        data = (data - mu) / sd

    df_long = (
        pd.DataFrame(data, index=row_names, columns=col_names)
        .reset_index(names="row")
        .melt(id_vars="row", var_name="col", value_name="value")
    )
    return heatmap_long(df_long, x="col", y="row", value="value", title=title)
