"""Differential expression plot functions: volcano, MA, coefficient lollipop/expression."""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_point,
    geom_segment,
    geom_text,
    geom_hline,
    geom_vline,
    theme_classic,
    theme,
    element_text,
    ggtitle,
    labs,
    scale_color_manual,
    scale_y_continuous,
    coord_flip,
)

from ._utils import HeatmapResult


# ---------------------------------------------------------------------------
# Helpers shared by volcano/MA
# ---------------------------------------------------------------------------


def _assign_status(
    df: pd.DataFrame,
    logfc_col: str,
    pval_col: str,
    logfc_threshold: float,
    pval_threshold: float,
) -> pd.Series:
    sig = df[pval_col] < pval_threshold
    up = sig & (df[logfc_col] >= logfc_threshold)
    down = sig & (df[logfc_col] <= -logfc_threshold)
    status = pd.Series("ns", index=df.index)
    status[up] = "up"
    status[down] = "down"
    return status


def _get_gene_labels(
    df: pd.DataFrame,
    gene_col: Optional[str],
    pval_col: str,
    status_col: str,
    label_top_n: int,
    label_genes: Optional[List[str]],
) -> pd.Series:
    """Return a label series (empty string for unlabelled rows)."""
    neg_log_pval = -np.log10(df[pval_col].replace(0, np.finfo(float).tiny))
    labels = pd.Series("", index=df.index)

    # Always-labelled genes
    always: List[str] = list(label_genes) if label_genes else []
    if gene_col is not None:
        gene_names = df[gene_col]
    else:
        gene_names = df.index.astype(str)

    for g in always:
        mask = gene_names == g
        labels[mask] = g

    # Top-n significant by -log10(pval)
    sig_mask = df[status_col] != "ns"
    sig_df = df[sig_mask].copy()
    if not sig_df.empty and label_top_n > 0:
        sig_nlp = neg_log_pval[sig_mask]
        top_idx = sig_nlp.nlargest(label_top_n).index
        for idx in top_idx:
            lbl = gene_names[idx] if gene_col is not None else str(idx)
            labels[idx] = lbl

    return labels


# ---------------------------------------------------------------------------
# plot_volcano
# ---------------------------------------------------------------------------


def plot_volcano(
    data: pd.DataFrame,
    logfc_col: str = "log2FoldChange",
    pval_col: str = "padj",
    gene_col: Optional[str] = None,
    logfc_threshold: float = 1.0,
    pval_threshold: float = 0.05,
    label_top_n: int = 10,
    label_genes: Optional[List[str]] = None,
    label_col: Optional[str] = None,
    up_color: str = "#E41A1C",
    down_color: str = "#377EB8",
    ns_color: str = "#AAAAAA",
    point_size: float = 1.2,
    alpha: float = 0.7,
    x_label: str = "log2 Fold Change",
    y_label: str = "-log10(adj. p-value)",
    title: Optional[str] = None,
    xlim: Optional[tuple] = None,
    ylim: Optional[tuple] = None,
) -> ggplot:
    """Volcano plot from a DE results DataFrame.

    Points are colored by significance + direction. Top-n significant genes
    by -log10(pval) are labeled with nudged ``geom_text``.
    """
    df = data.copy()
    df = df.dropna(subset=[logfc_col, pval_col]).copy()
    df[pval_col] = df[pval_col].clip(lower=np.finfo(float).tiny)
    df["__neg_log_p__"] = -np.log10(df[pval_col])
    df["__status__"] = _assign_status(df, logfc_col, pval_col, logfc_threshold, pval_threshold)

    color_pal = {"ns": ns_color, "up": up_color, "down": down_color}

    df["__label__"] = _get_gene_labels(
        df, gene_col, pval_col, "__status__", label_top_n, label_genes
    )

    p = (
        ggplot(df)
        + aes(x=logfc_col, y="__neg_log_p__", color="__status__")
        + geom_point(size=point_size, alpha=alpha)
        + geom_vline(xintercept=[-logfc_threshold, logfc_threshold], linetype="dashed", alpha=0.5)
        + geom_hline(yintercept=-np.log10(pval_threshold), linetype="dashed", alpha=0.5)
        + scale_color_manual(values=color_pal)
        + theme_classic()
        + labs(x=x_label, y=y_label, color="")
    )

    # Label layer (only rows with non-empty label)
    label_df = df[df["__label__"] != ""].copy()
    if not label_df.empty:
        # Simple nudge: alternate up/down
        label_df = label_df.sort_values("__neg_log_p__", ascending=False)
        label_df["__nudge_y__"] = [0.3 * (-1) ** i for i in range(len(label_df))]
        label_df["__nudge_x__"] = label_df[logfc_col].apply(lambda v: 0.3 if v > 0 else -0.3)

        p = p + geom_text(
            data=label_df,
            mapping=aes(x=logfc_col, y="__neg_log_p__", label="__label__"),
            inherit_aes=False,
            size=7,
            ha="left",
            va="bottom",
        )

    if xlim is not None:
        from plotnine import xlim as _xlim
        p = p + _xlim(*xlim)
    if ylim is not None:
        from plotnine import ylim as _ylim
        p = p + _ylim(*ylim)

    if title is not None:
        p = p + ggtitle(title)

    return p


# ---------------------------------------------------------------------------
# plot_ma
# ---------------------------------------------------------------------------


def plot_ma(
    data: pd.DataFrame,
    mean_col: str = "baseMean",
    logfc_col: str = "log2FoldChange",
    pval_col: str = "padj",
    gene_col: Optional[str] = None,
    pval_threshold: float = 0.05,
    logfc_threshold: float = 0.0,
    label_top_n: int = 10,
    label_genes: Optional[List[str]] = None,
    up_color: str = "#E41A1C",
    down_color: str = "#377EB8",
    ns_color: str = "#AAAAAA",
    point_size: float = 1.0,
    alpha: float = 0.7,
    title: Optional[str] = None,
) -> ggplot:
    """MA plot (mean expression vs log fold change).

    Points colored by significance + direction. Horizontal reference line at
    logFC = 0.
    """
    df = data.copy()
    df = df.dropna(subset=[mean_col, logfc_col, pval_col]).copy()
    df = df[df[mean_col] > 0].copy()
    df["__log_mean__"] = np.log2(df[mean_col].clip(lower=1e-9))
    df[pval_col] = df[pval_col].clip(lower=np.finfo(float).tiny)
    df["__status__"] = _assign_status(df, logfc_col, pval_col, logfc_threshold, pval_threshold)

    color_pal = {"ns": ns_color, "up": up_color, "down": down_color}

    df["__label__"] = _get_gene_labels(
        df, gene_col, pval_col, "__status__", label_top_n, label_genes
    )

    p = (
        ggplot(df)
        + aes(x="__log_mean__", y=logfc_col, color="__status__")
        + geom_point(size=point_size, alpha=alpha)
        + geom_hline(yintercept=0, linetype="dashed", alpha=0.5)
        + scale_color_manual(values=color_pal)
        + theme_classic()
        + labs(x="log2(Mean Expression)", y="log2 Fold Change", color="")
    )

    label_df = df[df["__label__"] != ""]
    if not label_df.empty:
        p = p + geom_text(
            data=label_df,
            mapping=aes(x="__log_mean__", y=logfc_col, label="__label__"),
            inherit_aes=False,
            size=7,
        )

    if title is not None:
        p = p + ggtitle(title)

    return p


# ---------------------------------------------------------------------------
# plot_coef_lollipop
# ---------------------------------------------------------------------------


def plot_coef_lollipop(
    coefs: Union[pd.Series, pd.DataFrame],
    top_n: int = 20,
    coef_col: str = "coefficient",
    se_col: Optional[str] = None,
    pos_color: str = "#E41A1C",
    neg_color: str = "#377EB8",
    zero_line: bool = True,
    title: Optional[str] = None,
    x_label: str = "Coefficient",
    feature_label: str = "Feature",
) -> ggplot:
    """Horizontal lollipop plot of model coefficients.

    Features are sorted by absolute coefficient magnitude; top-n are shown.
    Positive coefficients use *pos_color*, negative *neg_color*.
    If *se_col* is present, horizontal error bars are added.
    """
    if isinstance(coefs, pd.Series):
        df = coefs.rename("coefficient").to_frame()
        df.index.name = "__feature__"
        df = df.reset_index()
        df.columns = ["__feature__", "coefficient"]
        coef_col = "coefficient"
    else:
        df = coefs.copy().reset_index()
        if "__feature__" not in df.columns:
            df = df.rename(columns={df.columns[0]: "__feature__"})

    df = df.dropna(subset=[coef_col]).copy()
    df["__abs_coef__"] = df[coef_col].abs()
    df = df.nlargest(top_n, "__abs_coef__").copy()
    df = df.sort_values("__abs_coef__", ascending=True).copy()
    df["__feature__"] = pd.Categorical(df["__feature__"], categories=df["__feature__"].tolist())
    df["__direction__"] = df[coef_col].apply(lambda v: "positive" if v >= 0 else "negative")

    p = (
        ggplot(df)
        + aes(x="__feature__", y=coef_col, color="__direction__")
        + geom_segment(aes(xend="__feature__", yend=0), size=0.8)
        + geom_point(size=2.5)
        + scale_color_manual(values={"positive": pos_color, "negative": neg_color})
        + coord_flip()
        + theme_classic()
        + labs(x=feature_label, y=x_label, color="")
    )

    if zero_line:
        p = p + geom_vline(xintercept=0, linetype="dashed", alpha=0.5)

    if se_col is not None and se_col in df.columns:
        from plotnine import geom_errorbarh
        p = p + geom_errorbarh(
            aes(xmin=coef_col + " - " + se_col, xmax=coef_col + " + " + se_col),
            height=0.3,
        )

    if title is not None:
        p = p + ggtitle(title)

    return p


# ---------------------------------------------------------------------------
# plot_coef_expression
# ---------------------------------------------------------------------------


def plot_coef_expression(
    data,
    coefs: Union[pd.Series, pd.DataFrame],
    group_by: str,
    top_n: int = 20,
    coef_col: str = "coefficient",
    layer: Optional[str] = None,
    plot_type: str = "dot",
    order_by_coef: bool = True,
    title: Optional[str] = None,
) -> Union[ggplot, HeatmapResult]:
    """Plot expression of top-n features from a coefficient table.

    Selects top_n features by absolute coefficient value, then plots their
    expression using ``plot_dot``, ``plot_expression``, or ``plot_heatmap``
    depending on *plot_type*. Returns a ``HeatmapResult`` for *plot_type="heatmap"*.
    """
    # Extract top-n feature names
    if isinstance(coefs, pd.Series):
        coef_series = coefs.rename("coefficient")
    else:
        coef_series = coefs[coef_col].copy()
        coef_series.index = (
            coefs.index if not isinstance(coefs.index, pd.RangeIndex)
            else coefs.iloc[:, 0]
        )

    coef_abs = coef_series.abs().nlargest(top_n)
    if order_by_coef:
        features = coef_series.loc[coef_abs.index].sort_values(ascending=False).index.tolist()
    else:
        features = coef_abs.index.tolist()

    features = [str(f) for f in features]

    if plot_type == "dot":
        from .expression import plot_dot
        return plot_dot(data, features=features, group_by=group_by, layer=layer, title=title)

    if plot_type == "violin":
        from .expression import plot_expression
        return plot_expression(data, features=features, group_by=group_by, layer=layer, title=title)

    if plot_type == "heatmap":
        from .expression import plot_heatmap
        p = plot_heatmap(data, features=features, group_by=group_by, layer=layer, title=title)
        return HeatmapResult(plot=p)

    raise ValueError(f"Unknown plot_type '{plot_type}'. Choose from 'dot', 'violin', 'heatmap'.")
