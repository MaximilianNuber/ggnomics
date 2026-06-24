from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import Optional, Sequence, Any

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_violin,
    geom_boxplot,
    geom_jitter,
    theme_classic,
    ggtitle,
    labs,
)


@singledispatch
def expression_violin(data: Any, *args, **kwargs):  # pragma: no cover - dispatch entry
    raise TypeError(
        "Unsupported data type for expression_violin. Pass a pandas DataFrame, or use the sce/se wrappers."
    )


@expression_violin.register(pd.DataFrame)
def _(df: pd.DataFrame,
      value: str,
      group: str,
      *,
      fill: Optional[str] = None,
      log1p: bool = False,
      add_box: bool = False,
      add_points: bool = True,
      point_alpha: float = 0.25,
      point_size: float = 0.2,
      jitter_width: float = 0.15,
      palette: Optional[dict] = None,
      title: Optional[str] = None,
      y_label: Optional[str] = None):
    d = df.copy()
    ycol = "__expr__"
    d[ycol] = np.log1p(d[value]) if log1p else d[value]

    aes_kwargs = {"x": group, "y": ycol}
    if fill is not None:
        aes_kwargs["fill"] = fill

    p = (
        ggplot(d)
        + aes(**aes_kwargs)
        + geom_violin(scale="width", trim=True)
        + theme_classic()
    )

    if add_box:
        p = p + geom_boxplot(width=0.12, outlier_alpha=0.0)

    if add_points:
        p = p + geom_jitter(width=jitter_width, height=0.0, size=point_size, alpha=point_alpha)

    if y_label:
        p = p + labs(y=y_label)
    if title:
        p = p + ggtitle(title)

    # Palette handling left to caller via scale_* if needed.
    return p


# -------------------------------
# Optional BiocPy adapters (dynamic)
# -------------------------------

def _get_assay_matrix(obj, assay: str):
    if hasattr(obj, "assay"):
        try:
            return obj.assay(assay)
        except Exception:
            pass
    if hasattr(obj, "assays"):
        try:
            return obj.assays[assay]
        except Exception:
            pass
    if hasattr(obj, "X"):
        return obj.X
    raise AttributeError("Could not access assay matrix from object.")


def _get_row_names(obj):
    for attr in ("row_names", "feature_names", "gene_names"):
        if getattr(obj, attr, None) is not None:
            return getattr(obj, attr)
    if hasattr(obj, "row_data") and hasattr(obj.row_data, "to_pandas"):
        rd = obj.row_data.to_pandas()
        if rd is not None and rd.index is not None:
            return rd.index
    return None


def _get_col_data(obj) -> pd.DataFrame:
    if hasattr(obj, "col_data") and hasattr(obj.col_data, "to_pandas"):
        return obj.col_data.to_pandas().copy()
    raise AttributeError("Object has no col_data.to_pandas().")


def _extract_gene_vector(mat, genes, gene: str):
    if genes is None:
        raise ValueError("Cannot locate gene names to select expression vector.")
    try:
        if isinstance(genes, (pd.Index, pd.Series, list, np.ndarray)):
            idx = int(np.where(np.asarray(genes) == gene)[0][0])
        else:
            idx = genes.index(gene)  # fallback
    except Exception:
        raise KeyError(f"Gene '{gene}' not found in row names.")
    vec = mat[idx, :] if mat.shape[0] == len(genes) else mat[:, idx]
    if hasattr(vec, "toarray"):
        vec = vec.toarray()
    vec = np.ravel(np.asarray(vec))
    return vec


def expression_violin_sce(
    sce,
    gene: str,
    group_col: str,
    *,
    assay: str = "logcounts",
    log1p: bool = False,
    title: Optional[str] = None,
):
    mat = _get_assay_matrix(sce, assay)
    genes = _get_row_names(sce)
    expr = _extract_gene_vector(mat, genes, gene)
    meta = _get_col_data(sce)
    df = pd.DataFrame({"expr": expr, group_col: meta[group_col].values})
    return expression_violin(df, value="expr", group=group_col, log1p=log1p, title=title or f"{gene} expression")


def expression_violin_se(
    se,
    gene: str,
    group_col: str,
    *,
    assay: str = "logcounts",
    log1p: bool = False,
    title: Optional[str] = None,
):
    mat = _get_assay_matrix(se, assay)
    genes = _get_row_names(se)
    expr = _extract_gene_vector(mat, genes, gene)
    meta = _get_col_data(se)
    df = pd.DataFrame({"expr": expr, group_col: meta[group_col].values})
    return expression_violin(df, value="expr", group=group_col, log1p=log1p, title=title or f"{gene} expression")


# Try dynamic registration for BiocPy types if available
try:  # pragma: no cover - optional dependency
    from singlecellexperiment import SingleCellExperiment  # type: ignore

    @expression_violin.register(SingleCellExperiment)
    def _(sce, gene: str, group_col: str, *, assay: str = "logcounts", log1p: bool = False, title: Optional[str] = None):
        return expression_violin_sce(sce, gene, group_col, assay=assay, log1p=log1p, title=title)
except Exception:
    pass

try:  # pragma: no cover - optional dependency
    from summarizedexperiment import SummarizedExperiment  # type: ignore

    @expression_violin.register(SummarizedExperiment)
    def _(se, gene: str, group_col: str, *, assay: str = "logcounts", log1p: bool = False, title: Optional[str] = None):
        return expression_violin_se(se, gene, group_col, assay=assay, log1p=log1p, title=title)
except Exception:
    pass
