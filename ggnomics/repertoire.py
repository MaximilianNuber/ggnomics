"""Immune repertoire plots: clonotype abundance, overlap heatmap, and embedding."""

from __future__ import annotations

from functools import singledispatch
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_bar,
    ggplot,
    ggtitle,
    labs,
    scale_fill_manual,
    theme,
    theme_classic,
)

from ._utils import HeatmapResult
from .scatter import _embedding_frame_from_dataframe


def _unsupported_type(function_name: str, data: object) -> TypeError:
    return TypeError(
        f"{function_name} does not support {type(data).__module__}."
        f"{type(data).__qualname__}. Pass a pandas.DataFrame or install the "
        "optional dependency for a supported genomics container."
    )


def _require_columns(data: pd.DataFrame, columns: List[str], *, location: str) -> None:
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise KeyError(f"Column(s) {missing} not found in {location}. Available: {list(data.columns)}")


# Default expansion thresholds (scRepertoire-style)
_DEFAULT_THRESHOLDS: Dict[int, str] = {
    1: "Single",
    2: "Small (2-5)",
    6: "Medium (6-20)",
    21: "Large (21-100)",
    101: "Hyperexpanded (>100)",
}

_EXPANSION_PALETTE: Dict[str, str] = {
    "Single": "#BDD7EE",
    "Small (2-5)": "#9DC3E6",
    "Medium (6-20)": "#2E75B6",
    "Large (21-100)": "#1F4E79",
    "Hyperexpanded (>100)": "#FF0000",
    "None": "#DDDDDD",
}

_EXPANSION_ORDER = [
    "None",
    "Single",
    "Small (2-5)",
    "Medium (6-20)",
    "Large (21-100)",
    "Hyperexpanded (>100)",
]

_OVERLAP_METHODS = ("jaccard", "morisita", "overlap_coef")


def _compute_expansion_category(clone_size: int, thresholds: Dict[int, str]) -> str:
    """Map a clone size to an expansion category."""
    sorted_thresholds = sorted(thresholds.keys())
    for i, cutoff in enumerate(sorted_thresholds):
        if clone_size < cutoff:
            return "None"
        next_cutoff = sorted_thresholds[i + 1] if i + 1 < len(sorted_thresholds) else None
        if next_cutoff is None or clone_size < next_cutoff:
            return thresholds[cutoff]
    return thresholds[sorted_thresholds[-1]]


def expansion_column(
    clonotype_series: pd.Series,
    thresholds: Optional[Dict[int, str]] = None,
) -> pd.Series:
    """Compute an expansion-category Series from clone sizes.

    Cells with a missing (``NaN``) clonotype map to ``"None"``. This never
    mutates ``clonotype_series``.
    """
    thresholds = thresholds or _DEFAULT_THRESHOLDS
    clone_counts = clonotype_series.dropna().value_counts()

    def _cat(cid) -> str:
        if pd.isna(cid):
            return "None"
        return _compute_expansion_category(int(clone_counts.get(cid, 0)), thresholds)

    return clonotype_series.apply(_cat)


# ---------------------------------------------------------------------------
# plot_clonotype_abundance
# ---------------------------------------------------------------------------


def _abundance_table(
    obs_df: pd.DataFrame,
    clonotype_col: str,
    top_n: int,
    expansion_col: Optional[str],
    thresholds: Dict[int, str],
    facet_col: Optional[str],
) -> pd.DataFrame:
    """Rank clonotypes within each facet group (or globally when ``facet_col`` is None)."""

    if facet_col is None:
        groups: List[Tuple[Optional[object], pd.DataFrame]] = [(None, obs_df)]
    else:
        groups = list(obs_df.groupby(facet_col, observed=True))

    rows = []
    for group_name, sub in groups:
        clone_counts = sub[clonotype_col].dropna().value_counts()
        top_clones = clone_counts.head(top_n)
        for rank, (clone_id, count) in enumerate(top_clones.items(), start=1):
            if expansion_col is not None and expansion_col in sub.columns:
                exp_cat = sub.loc[sub[clonotype_col] == clone_id, expansion_col].mode()
                cat = exp_cat.iloc[0] if len(exp_cat) > 0 else "Unknown"
            else:
                cat = _compute_expansion_category(int(count), thresholds)

            row = {
                "rank": rank,
                "clone_id": str(clone_id),
                "n_cells": int(count),
                "expansion": cat,
            }
            if facet_col is not None:
                row[facet_col] = group_name
            rows.append(row)

    return pd.DataFrame(rows)


@singledispatch
def plot_clonotype_abundance(
    data: pd.DataFrame,
    clonotype_col: str,
    sample_col: Optional[str] = None,
    top_n: int = 20,
    expansion_col: Optional[str] = None,
    expansion_thresholds: Optional[Dict[int, str]] = None,
    palette: Optional[Dict] = None,
    facet_by: Optional[str] = None,
    title: Optional[str] = None,
) -> ggplot:
    """Barplot of clonotype abundance colored by expansion category.

    x-axis: clonotype rank (1 = most abundant, computed within each facet
    group when ``facet_by``/``sample_col`` is given). y-axis: cell count.
    Fill: expansion category (computed from clone sizes if ``expansion_col``
    is ``None``).

    Args:
        data: DataFrame whose rows are cells. ``clonotype_col`` and any of
            ``sample_col``/``expansion_col``/``facet_by`` are columns.
        clonotype_col: Column holding clonotype identifiers (``NaN`` for
            cells without a clonotype).
        sample_col: Column used to facet the plot, and to rank/count
            clonotypes *within* each sample rather than globally. Ignored
            if ``facet_by`` is also given (``facet_by`` takes precedence).
        top_n: Number of top clonotypes to show per facet. Must be ``>= 1``.
        expansion_col: Column with a precomputed expansion category. When
            ``None``, categories are computed from clone sizes.
        expansion_thresholds: ``{min_clone_size: category_label}`` mapping.
            Defaults to the scRepertoire-style thresholds.
        palette: ``{category: hex}`` fill color mapping. Defaults to the
            built-in expansion palette.
        facet_by: Column used to facet the plot (and to rank within each
            facet). Defaults to ``sample_col`` when omitted.
        title: Plot title.

    Returns:
        A ``plotnine.ggplot`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        ValueError: If ``top_n < 1``.
        KeyError: If a requested column is absent.
    """
    raise _unsupported_type("plot_clonotype_abundance", data)


@plot_clonotype_abundance.register(pd.DataFrame)
def _plot_clonotype_abundance_dataframe(
    data: pd.DataFrame,
    clonotype_col: str,
    sample_col: Optional[str] = None,
    top_n: int = 20,
    expansion_col: Optional[str] = None,
    expansion_thresholds: Optional[Dict[int, str]] = None,
    palette: Optional[Dict] = None,
    facet_by: Optional[str] = None,
    title: Optional[str] = None,
) -> ggplot:
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}.")

    required = [clonotype_col]
    for column in (sample_col, expansion_col, facet_by):
        if column is not None:
            required.append(column)
    _require_columns(data, required, location="the DataFrame")

    obs_df = data.reset_index(drop=True)
    thresholds = expansion_thresholds or _DEFAULT_THRESHOLDS
    facet_col = facet_by if facet_by is not None else sample_col

    plot_df = _abundance_table(obs_df, clonotype_col, top_n, expansion_col, thresholds, facet_col)
    plot_df["expansion"] = pd.Categorical(
        plot_df["expansion"],
        categories=[c for c in _EXPANSION_ORDER if c in plot_df["expansion"].values],
        ordered=True,
    )

    pal = palette or _EXPANSION_PALETTE

    p = (
        ggplot(plot_df)
        + aes(x="rank", y="n_cells", fill="expansion")
        + geom_bar(stat="identity")
        + scale_fill_manual(
            breaks=[k for k in _EXPANSION_ORDER if k in pal],
            values=[pal.get(k, "#AAAAAA") for k in _EXPANSION_ORDER if k in pal],
        )
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + labs(x="Clonotype rank", y="Cell count", fill="Expansion")
    )

    if facet_col is not None:
        p = p + facet_wrap(facet_col)

    if title is not None:
        p = p + ggtitle(title)

    return p


# ---------------------------------------------------------------------------
# plot_clonotype_overlap
# ---------------------------------------------------------------------------


def _overlap_matrix(
    obs_df: pd.DataFrame,
    clonotype_col: str,
    sample_col: str,
    method: str,
) -> pd.DataFrame:
    samples = sorted(obs_df[sample_col].unique().tolist())
    n_samples = len(samples)

    clone_sets: Dict[str, set] = {}
    clone_counts: Dict[str, Dict[str, int]] = {}
    for smp in samples:
        sub = obs_df.loc[obs_df[sample_col] == smp, clonotype_col].dropna()
        clone_sets[smp] = set(sub.unique())
        clone_counts[smp] = sub.value_counts().to_dict()

    mat = np.zeros((n_samples, n_samples))
    for i, s1 in enumerate(samples):
        for j, s2 in enumerate(samples):
            if i == j:
                mat[i, j] = 1.0
                continue
            if method == "jaccard":
                inter = len(clone_sets[s1] & clone_sets[s2])
                union = len(clone_sets[s1] | clone_sets[s2])
                mat[i, j] = inter / union if union > 0 else 0.0
            elif method == "overlap_coef":
                inter = len(clone_sets[s1] & clone_sets[s2])
                denom = min(len(clone_sets[s1]), len(clone_sets[s2]))
                mat[i, j] = inter / denom if denom > 0 else 0.0
            else:  # morisita
                shared = clone_sets[s1] & clone_sets[s2]
                if not shared:
                    mat[i, j] = 0.0
                    continue
                c1, c2 = clone_counts[s1], clone_counts[s2]
                n1, n2 = sum(c1.values()), sum(c2.values())
                if n1 == 0 or n2 == 0:
                    mat[i, j] = 0.0
                    continue
                num = 2.0 * sum((c1.get(cl, 0) / n1) * (c2.get(cl, 0) / n2) for cl in shared)
                d1 = sum((v / n1) ** 2 for v in c1.values())
                d2 = sum((v / n2) ** 2 for v in c2.values())
                denom = d1 + d2
                mat[i, j] = num / denom if denom > 0 else 0.0

    return pd.DataFrame(mat, index=samples, columns=samples)


@singledispatch
def plot_clonotype_overlap(
    data: pd.DataFrame,
    clonotype_col: str,
    sample_col: str,
    method: str = "jaccard",
    title: Optional[str] = None,
) -> HeatmapResult:
    """Heatmap of pairwise clonotype overlap between samples.

    Args:
        data: DataFrame whose rows are cells. ``clonotype_col`` and
            ``sample_col`` are columns.
        clonotype_col: Column holding clonotype identifiers.
        sample_col: Column defining the samples to compare.
        method: One of ``"jaccard"``, ``"morisita"``, or ``"overlap_coef"``.
        title: Plot title.

    Returns:
        A :class:`ggnomics._utils.HeatmapResult` with ``.plot`` (a
        ``plotnine.ggplot``) and ``.matrix`` (a square DataFrame).

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        ValueError: If ``method`` is not supported.
        KeyError: If a requested column is absent.
    """
    raise _unsupported_type("plot_clonotype_overlap", data)


@plot_clonotype_overlap.register(pd.DataFrame)
def _plot_clonotype_overlap_dataframe(
    data: pd.DataFrame,
    clonotype_col: str,
    sample_col: str,
    method: str = "jaccard",
    title: Optional[str] = None,
) -> HeatmapResult:
    if method not in _OVERLAP_METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {_OVERLAP_METHODS}.")
    _require_columns(data, [clonotype_col, sample_col], location="the DataFrame")

    obs_df = data.reset_index(drop=True)
    matrix_df = _overlap_matrix(obs_df, clonotype_col, sample_col, method)

    from .heatmap import heatmap_from_matrix

    p = heatmap_from_matrix(matrix_df, title=title or f"Clonotype overlap ({method})")

    return HeatmapResult(plot=p, matrix=matrix_df)


# ---------------------------------------------------------------------------
# plot_clonotype_embedding
# ---------------------------------------------------------------------------


def _clonotype_embedding_plot(
    emb_df: pd.DataFrame,
    components: Tuple[int, int],
    clonotype_series: pd.Series,
    dimred: str,
    expansion_thresholds: Optional[Dict[int, str]],
    non_tcell_color: str,
    palette: Optional[Dict],
    size: Optional[float],
    stroke: Optional[float],
    alpha: float,
    title: Optional[str],
) -> ggplot:
    """Shared plot construction, reused by every container adapter."""

    from .scatter import _strip_x_prefix, plot_scatter

    comp_cols = list(emb_df.columns)
    ci, cj = components[0] - 1, components[1] - 1
    if ci < 0 or cj < 0 or ci >= len(comp_cols) or cj >= len(comp_cols):
        raise IndexError(f"components={components} out of range for embedding with {len(comp_cols)} dimensions.")
    x_col, y_col = comp_cols[ci], comp_cols[cj]
    x_safe, y_safe = f"__emb_{x_col}__", f"__emb_{y_col}__"

    emb_2d = emb_df[[x_col, y_col]].rename(columns={x_col: x_safe, y_col: y_safe})
    work_df = emb_2d.reset_index(drop=True)
    work_df["__expansion__"] = expansion_column(
        clonotype_series.reset_index(drop=True), expansion_thresholds
    ).to_numpy()

    order_cats = [c for c in _EXPANSION_ORDER if c in work_df["__expansion__"].values]

    pal = dict(palette) if palette is not None else {}
    pal.setdefault("None", non_tcell_color)
    for cat, col in _EXPANSION_PALETTE.items():
        pal.setdefault(cat, col)

    base = _strip_x_prefix(dimred).upper()
    return plot_scatter(
        work_df,
        x=x_safe,
        y=y_safe,
        color="__expansion__",
        size=size,
        stroke=stroke,
        alpha=alpha,
        palette={k: pal[k] for k in order_cats if k in pal},
        order=order_cats,
        color_label="Expansion",
        x_label=f"{base} {components[0]}",
        y_label=f"{base} {components[1]}",
        title=title,
    )


@singledispatch
def plot_clonotype_embedding(
    data: pd.DataFrame,
    clonotype_col: str,
    dimred: str = "X_umap",
    components: Tuple[int, int] = (1, 2),
    expansion_thresholds: Optional[Dict[int, str]] = None,
    non_tcell_color: str = "#DDDDDD",
    palette: Optional[Dict] = None,
    size: Optional[float] = None,
    stroke: Optional[float] = None,
    alpha: float = 0.8,
    title: Optional[str] = None,
) -> ggplot:
    """Embedding plot colored by clonotype expansion category.

    Cells without a clonotype (``NaN`` in ``clonotype_col``) are drawn first
    (underneath expanded clones) in ``non_tcell_color``.

    Args:
        data: DataFrame with embedding columns matched against ``dimred``
            (see :func:`ggnomics.plot_embedding`) plus ``clonotype_col``.
        clonotype_col: Column holding clonotype identifiers.
        dimred: Embedding key (e.g. ``"X_pca"``, ``"X_umap"``).
        components: One-indexed ``(x, y)`` component pair to plot.
        expansion_thresholds: ``{min_clone_size: category_label}`` mapping.
        non_tcell_color: Fill color for cells without a clonotype.
        palette: ``{category: hex}`` mapping overriding the default
            expansion palette.
        size: Point size (``None`` -> adaptive).
        stroke: Point stroke width (``None`` -> adaptive).
        alpha: Point transparency.
        title: Plot title.

    Returns:
        A ``plotnine.ggplot`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        KeyError: If ``clonotype_col`` or the embedding is not found.
    """
    raise _unsupported_type("plot_clonotype_embedding", data)


@plot_clonotype_embedding.register(pd.DataFrame)
def _plot_clonotype_embedding_dataframe(
    data: pd.DataFrame,
    clonotype_col: str,
    dimred: str = "X_umap",
    components: Tuple[int, int] = (1, 2),
    expansion_thresholds: Optional[Dict[int, str]] = None,
    non_tcell_color: str = "#DDDDDD",
    palette: Optional[Dict] = None,
    size: Optional[float] = None,
    stroke: Optional[float] = None,
    alpha: float = 0.8,
    title: Optional[str] = None,
) -> ggplot:
    if clonotype_col not in data.columns:
        raise KeyError(
            f"clonotype_col {clonotype_col!r} not found in the DataFrame. Available: {list(data.columns)[:20]}"
        )

    emb_df = _embedding_frame_from_dataframe(data, dimred)
    return _clonotype_embedding_plot(
        emb_df,
        components,
        data[clonotype_col],
        dimred,
        expansion_thresholds,
        non_tcell_color,
        palette,
        size,
        stroke,
        alpha,
        title,
    )


__all__ = [
    "plot_clonotype_abundance",
    "plot_clonotype_overlap",
    "plot_clonotype_embedding",
]
