"""Immune repertoire plots: clonotype abundance, overlap heatmap, and embedding."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_bar,
    theme_classic,
    theme,
    element_text,
    ggtitle,
    labs,
    facet_wrap,
    scale_fill_manual,
    scale_fill_brewer,
)

from ._accessor import DataAccessor
from ._utils import HeatmapResult


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


def _compute_expansion_category(
    clone_size: int,
    thresholds: Dict[int, str],
) -> str:
    """Map a clone size to an expansion category."""
    sorted_thresholds = sorted(thresholds.keys())
    for i, cutoff in enumerate(sorted_thresholds):
        if clone_size < cutoff:
            return "None"
        next_cutoff = sorted_thresholds[i + 1] if i + 1 < len(sorted_thresholds) else None
        if next_cutoff is None or clone_size < next_cutoff:
            return thresholds[cutoff]
    return thresholds[sorted_thresholds[-1]]


def _get_obs_with_clonotype(data, clonotype_col: str) -> pd.DataFrame:
    """Extract obs DataFrame (always includes clonotype_col)."""
    if isinstance(data, pd.DataFrame):
        return data.reset_index(drop=True)
    acc = DataAccessor(data)
    obs_df = acc.obs().reset_index(drop=True)
    return obs_df


# ---------------------------------------------------------------------------
# plot_clonotype_abundance
# ---------------------------------------------------------------------------


def plot_clonotype_abundance(
    data,
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

    x-axis: clonotype rank (1 = most abundant).
    y-axis: cell count.
    Fill: expansion category (computed from clone sizes if *expansion_col* is None).
    """
    obs_df = _get_obs_with_clonotype(data, clonotype_col)

    if clonotype_col not in obs_df.columns:
        raise KeyError(f"clonotype_col '{clonotype_col}' not found in obs.")

    thresholds = expansion_thresholds or _DEFAULT_THRESHOLDS

    # Compute clone sizes
    clone_counts = obs_df[clonotype_col].dropna().value_counts()
    top_clones = clone_counts.head(top_n)

    plot_rows = []
    for rank, (clone_id, count) in enumerate(top_clones.items(), start=1):
        if expansion_col is not None and expansion_col in obs_df.columns:
            exp_cat = obs_df.loc[obs_df[clonotype_col] == clone_id, expansion_col].mode()
            cat = exp_cat.iloc[0] if len(exp_cat) > 0 else "Unknown"
        else:
            cat = _compute_expansion_category(int(count), thresholds)

        plot_rows.append({
            "rank": rank,
            "clone_id": str(clone_id),
            "n_cells": int(count),
            "expansion": cat,
        })

    plot_df = pd.DataFrame(plot_rows)
    plot_df["expansion"] = pd.Categorical(
        plot_df["expansion"],
        categories=[c for c in _EXPANSION_ORDER if c in plot_df["expansion"].values],
        ordered=True,
    )

    if sample_col is not None and sample_col in obs_df.columns and facet_by is None:
        facet_by = sample_col

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

    if facet_by is not None and facet_by in obs_df.columns:
        p = p + facet_wrap(facet_by)

    if title is not None:
        p = p + ggtitle(title)

    return p


# ---------------------------------------------------------------------------
# plot_clonotype_overlap
# ---------------------------------------------------------------------------


def plot_clonotype_overlap(
    data,
    clonotype_col: str,
    sample_col: str,
    method: str = "jaccard",
    title: Optional[str] = None,
) -> HeatmapResult:
    """Heatmap of clonotype overlap between samples.

    Computes pairwise overlap (Jaccard, Morisita, or overlap coefficient)
    and returns a ``HeatmapResult`` with ``.plot`` (ggplot) and ``.matrix``
    (square DataFrame).
    """
    obs_df = _get_obs_with_clonotype(data, clonotype_col)

    for col in (clonotype_col, sample_col):
        if col not in obs_df.columns:
            raise KeyError(f"'{col}' not found in obs columns.")

    samples = sorted(obs_df[sample_col].unique().tolist())
    n_samples = len(samples)

    # Build per-sample clone sets / counts
    clone_sets: Dict[str, set] = {}
    clone_counts: Dict[str, Dict[str, int]] = {}

    for smp in samples:
        mask = obs_df[sample_col] == smp
        sub = obs_df.loc[mask, clonotype_col].dropna()
        clone_sets[smp] = set(sub.unique())
        vc = sub.value_counts()
        clone_counts[smp] = vc.to_dict()

    # Compute pairwise matrix
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
            elif method == "morisita":
                shared = clone_sets[s1] & clone_sets[s2]
                if not shared:
                    mat[i, j] = 0.0
                    continue
                c1 = clone_counts[s1]
                c2 = clone_counts[s2]
                n1 = sum(c1.values())
                n2 = sum(c2.values())
                if n1 == 0 or n2 == 0:
                    mat[i, j] = 0.0
                    continue
                num = 2.0 * sum(
                    (c1.get(cl, 0) / n1) * (c2.get(cl, 0) / n2) for cl in shared
                )
                d1 = sum((v / n1) ** 2 for v in c1.values())
                d2 = sum((v / n2) ** 2 for v in c2.values())
                denom = d1 + d2
                mat[i, j] = num / denom if denom > 0 else 0.0
            else:
                raise ValueError(f"Unknown method '{method}'. Choose 'jaccard', 'morisita', 'overlap_coef'.")

    matrix_df = pd.DataFrame(mat, index=samples, columns=samples)

    # Build ggplot via heatmap_from_matrix
    from .heatmap import heatmap_from_matrix
    p = heatmap_from_matrix(matrix_df, title=title or f"Clonotype overlap ({method})")

    return HeatmapResult(plot=p, matrix=matrix_df)


# ---------------------------------------------------------------------------
# plot_clonotype_embedding
# ---------------------------------------------------------------------------


def plot_clonotype_embedding(
    data,
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

    Cells without a clonotype (NaN in *clonotype_col*) are drawn first
    (underneath expanded clones) in *non_tcell_color*.
    """
    from .scatter import plot_reduced_dim
    from ._accessor import DataAccessor

    acc = DataAccessor(data)
    obs_df = acc.obs().reset_index(drop=True)

    if clonotype_col not in obs_df.columns:
        raise KeyError(f"clonotype_col '{clonotype_col}' not found in obs columns.")

    thresholds = expansion_thresholds or _DEFAULT_THRESHOLDS

    # Compute clone sizes
    clone_counts = obs_df[clonotype_col].dropna().value_counts()

    def _cat(cid) -> str:
        if pd.isna(cid):
            return "None"
        return _compute_expansion_category(int(clone_counts.get(cid, 0)), thresholds)

    obs_df["__expansion__"] = obs_df[clonotype_col].apply(_cat)

    # Inject the expansion column back into the data object
    # Build a temporary DataFrame from embedding + obs + expansion
    emb_df = acc.get_embedding(dimred)
    comp_cols = list(emb_df.columns)
    ci, cj = components[0] - 1, components[1] - 1
    x_col = comp_cols[ci]
    y_col = comp_cols[cj]
    x_safe = f"__emb_{x_col}__"
    y_safe = f"__emb_{y_col}__"

    emb_2d = emb_df[[x_col, y_col]].rename(columns={x_col: x_safe, y_col: y_safe})
    work_df = pd.concat([emb_2d.reset_index(drop=True), obs_df], axis=1)

    # Order: None (non-TCR cells) drawn first, Hyperexpanded last
    order_cats = [c for c in _EXPANSION_ORDER if c in work_df["__expansion__"].values]

    pal = dict(palette) if palette is not None else {}
    pal.setdefault("None", non_tcell_color)
    for cat, col in _EXPANSION_PALETTE.items():
        pal.setdefault(cat, col)

    from .scatter import plot_scatter
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
        x_label=f"{dimred.lstrip('X_').upper()} {components[0]}",
        y_label=f"{dimred.lstrip('X_').upper()} {components[1]}",
        title=title,
    )
