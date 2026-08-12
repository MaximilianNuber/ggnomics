"""plot_pairs — scatterplot matrix of embedding components."""

from __future__ import annotations

from functools import singledispatch
from typing import Dict, Optional

import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_point,
    facet_grid,
    theme_classic,
    theme,
    element_text,
    element_blank,
    ggtitle,
)

from ._utils import adaptive_size, color_scale
from .scatter import _embedding_frame_from_dataframe


def _unsupported_type(data: object) -> TypeError:
    return TypeError(
        f"plot_pairs does not support {type(data).__module__}."
        f"{type(data).__qualname__}. Pass a pandas.DataFrame or install the "
        "optional dependency for a supported genomics container."
    )


def _pairs_plot_from_embedding(
    emb_df: pd.DataFrame,
    n_components: int,
    color_series: Optional[pd.Series],
    color_by: Optional[str],
    size: Optional[float],
    alpha: float,
    palette: Optional[Dict],
    title: Optional[str],
) -> ggplot:
    """Shared plot construction, reused by every container adapter."""

    max_comps = emb_df.shape[1]
    k = min(n_components, max_comps)
    if k < 2:
        raise ValueError(
            f"Need at least 2 components for a pairs plot; embedding has {max_comps}."
        )

    comp_cols = list(emb_df.columns[:k])
    emb_k = emb_df[comp_cols].reset_index(drop=True)
    n = len(emb_k)

    records = []
    for row_dim in comp_cols:
        for col_dim in comp_cols:
            tmp = pd.DataFrame(
                {
                    "x_val": emb_k[col_dim].to_numpy(),
                    "y_val": emb_k[row_dim].to_numpy(),
                    "row_dim": row_dim,
                    "col_dim": col_dim,
                }
            )
            if color_series is not None:
                tmp[color_by] = color_series.to_numpy()
            records.append(tmp)

    long_df = pd.concat(records, ignore_index=True)
    long_df["row_dim"] = pd.Categorical(long_df["row_dim"], categories=comp_cols, ordered=True)
    long_df["col_dim"] = pd.Categorical(long_df["col_dim"], categories=comp_cols, ordered=True)

    n_panels = k * k
    effective_n = n * n_panels
    if size is None:
        size = adaptive_size(effective_n, size_max=0.8, size_min=0.1)

    aes_kwargs: dict = {"x": "x_val", "y": "y_val"}
    if color_by is not None:
        aes_kwargs["color"] = color_by

    p = (
        ggplot(long_df)
        + aes(**aes_kwargs)
        + geom_point(size=size, alpha=alpha)
        + facet_grid("row_dim ~ col_dim", scales="free")
        + theme_classic()
        + theme(
            axis_text=element_text(size=6),
            strip_text=element_text(size=7),
            axis_title=element_blank(),
        )
    )

    if color_by is not None:
        p = p + color_scale(long_df[color_by], palette=palette, type_="color")

    if title is not None:
        p = p + ggtitle(title)

    return p


@singledispatch
def plot_pairs(
    data: pd.DataFrame,
    dimred: str = "X_pca",
    n_components: int = 4,
    color_by: Optional[str] = None,
    size: Optional[float] = None,
    alpha: float = 0.6,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
) -> ggplot:
    """Scatterplot matrix of the first ``n_components`` embedding components.

    Each panel ``(row, col)`` shows a scatter of component ``row`` (y-axis)
    against component ``col`` (x-axis). All panels share the same color
    mapping so they can be read as a pairwise comparison. Diagonal panels
    (where row == col) show the same component on both axes and appear as a
    perfect 45-degree line; they serve as dimensional dividers.

    Args:
        data: DataFrame with embedding columns matched against ``dimred``
            the same way as :func:`ggnomics.plot_embedding` (e.g.
            ``PCA1``, ``PCA2``, ... or ``X_pca_1``, ``X_pca_2``, ...).
        dimred: Embedding key (e.g. ``"X_pca"``, ``"X_umap"``).
        n_components: Number of components to include (creates an
            ``n_components x n_components`` grid). Clipped to the number of
            available components.
        color_by: Column mapped to point color.
        size: Point size. ``None`` -> adaptive (based on n_cells x n_panels).
        alpha: Point transparency.
        palette: ``{category: hex}`` color mapping.
        title: Plot title (added as a label above the grid).

    Returns:
        A ``plotnine.ggplot`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        ValueError: If fewer than 2 components are available.
        KeyError: If ``color_by`` or the embedding is not found.
    """
    raise _unsupported_type(data)


@plot_pairs.register(pd.DataFrame)
def _plot_pairs_dataframe(
    data: pd.DataFrame,
    dimred: str = "X_pca",
    n_components: int = 4,
    color_by: Optional[str] = None,
    size: Optional[float] = None,
    alpha: float = 0.6,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
) -> ggplot:
    emb_df = _embedding_frame_from_dataframe(data, dimred)

    if color_by is not None and color_by not in data.columns:
        raise KeyError(
            f"color_by {color_by!r} not found in the DataFrame. "
            f"Available: {list(data.columns)[:20]}"
        )
    color_series = data[color_by].reset_index(drop=True) if color_by is not None else None

    return _pairs_plot_from_embedding(
        emb_df, n_components, color_series, color_by, size, alpha, palette, title
    )


__all__ = ["plot_pairs"]
