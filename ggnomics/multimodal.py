"""Multimodal plots: bimodal scatter (RNA vs protein) and ADT QC."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from plotnine import (
    ggplot,
    aes,
    geom_density,
    theme_classic,
    theme,
    element_text,
    ggtitle,
    labs,
    facet_wrap,
    scale_color_brewer,
    scale_fill_brewer,
    scale_color_manual,
    scale_fill_manual,
)

from ._accessor import DataAccessor


def _is_mudata(data) -> bool:
    return type(data).__name__ == "MuData" or "mudata" in (type(data).__module__ or "")


# ---------------------------------------------------------------------------
# plot_bimodal_scatter
# ---------------------------------------------------------------------------


def plot_bimodal_scatter(
    data,
    x_feature: str,
    y_feature: str,
    x_mod: str = "rna",
    y_mod: str = "prot",
    color: Optional[str] = None,
    size: Optional[float] = None,
    stroke: Optional[float] = None,
    alpha: float = 0.7,
    layer_x: Optional[str] = None,
    layer_y: Optional[str] = None,
    add_marginal: bool = False,
    palette: Optional[Dict] = None,
    cmap: str = "viridis",
    title: Optional[str] = None,
    x_label: Optional[str] = None,
    y_label: Optional[str] = None,
):
    """Scatter plot of two features from different modalities (e.g. RNA vs protein).

    Designed for CITEseq data. Resolves x_feature from *x_mod* and y_feature
    from *y_mod*. Supports MuData or AnnData with multiple layers.

    If *add_marginal=True*, delegates to ``plot_scatter_marginal`` and returns
    a ``plotnine.composition.Compose`` object instead of a ggplot.
    """
    from .scatter import plot_scatter
    from ._utils import adaptive_size, adaptive_stroke

    # Resolve x and y expression values
    if _is_mudata(data):
        try:
            x_adata = data[x_mod]
            y_adata = data[y_mod]
        except KeyError as e:
            raise KeyError(
                f"Modality '{e}' not found in MuData. Available: {list(data.mod.keys())}"
            ) from e

        x_acc = DataAccessor(x_adata)
        y_acc = DataAccessor(y_adata)

        x_series = x_acc.get_expression([x_feature], layer=layer_x)[x_feature].reset_index(drop=True)
        y_series = y_acc.get_expression([y_feature], layer=layer_y)[y_feature].reset_index(drop=True)
        obs_df = x_acc.obs().reset_index(drop=True)
        n = x_acc.n_obs()
    else:
        # AnnData with layers
        acc = DataAccessor(data)
        x_series = acc.get_expression([x_feature], layer=layer_x or x_mod)[x_feature].reset_index(drop=True)
        y_series = acc.get_expression([y_feature], layer=layer_y or y_mod)[y_feature].reset_index(drop=True)
        obs_df = acc.obs().reset_index(drop=True)
        n = acc.n_obs()

    # Build flat DataFrame
    work_df = pd.DataFrame({
        "__bm_x__": x_series.values,
        "__bm_y__": y_series.values,
    })
    # Add obs columns for color resolution
    for col in obs_df.columns:
        work_df[col] = obs_df[col].values

    if add_marginal:
        from .stats_plots import plot_scatter_marginal
        return plot_scatter_marginal(
            work_df,
            x="__bm_x__",
            y="__bm_y__",
            color=color,
            alpha=alpha,
            palette=palette,
            cmap=cmap,
            title=title,
            x_label=x_label or x_feature,
            y_label=y_label or y_feature,
        )

    return plot_scatter(
        work_df,
        x="__bm_x__",
        y="__bm_y__",
        color=color,
        size=size,
        stroke=stroke,
        alpha=alpha,
        palette=palette,
        cmap=cmap,
        x_label=x_label or x_feature,
        y_label=y_label or y_feature,
        title=title,
    )


# ---------------------------------------------------------------------------
# plot_adt_qc
# ---------------------------------------------------------------------------


def plot_adt_qc(
    data,
    isotype_controls: List[str],
    layer: Optional[str] = None,
    group_by: Optional[str] = None,
    log1p: bool = True,
    palette: Optional[Dict] = None,
    ncol: int = 3,
    title: Optional[str] = None,
) -> ggplot:
    """ADT / protein QC: overlay isotype control distributions on real antibodies.

    For each real antibody, shows a density curve (geom_density with color).
    Overlays the pooled isotype control distribution as a shaded fill.

    If *group_by* is provided, facets by that obs column.
    """
    acc = DataAccessor(data)
    obs_df = acc.obs().reset_index(drop=True)

    all_features = acc.var_names()
    real_features = [f for f in all_features if f not in isotype_controls]

    if not real_features:
        raise ValueError("No real antibody features found (all features listed as isotype controls).")
    if not isotype_controls:
        raise ValueError("isotype_controls list is empty.")

    # Fetch expression
    fetch_features = real_features + [f for f in isotype_controls if f in all_features]
    expr_df = acc.get_expression(fetch_features, layer=layer)

    if log1p:
        expr_df = np.log1p(expr_df)

    # Build long-format for real antibodies
    id_vars = []
    if group_by is not None and group_by in obs_df.columns:
        expr_df[group_by] = obs_df[group_by].values
        id_vars.append(group_by)

    real_long = expr_df[id_vars + real_features].melt(
        id_vars=id_vars, var_name="antibody", value_name="expression"
    )

    # Isotype combined
    iso_cols = [f for f in isotype_controls if f in expr_df.columns]
    if not iso_cols:
        raise ValueError(f"None of the isotype_controls found in data: {isotype_controls}")

    iso_vals = expr_df[iso_cols].values.ravel()
    iso_df = pd.DataFrame({"expression": iso_vals})
    if group_by is not None and group_by in obs_df.columns:
        iso_df[group_by] = np.tile(obs_df[group_by].values, len(iso_cols))

    p = (
        ggplot(real_long)
        + aes(x="expression", color="antibody")
        + geom_density(alpha=0.0, size=0.6)
        + geom_density(data=iso_df, mapping=aes(x="expression"), inherit_aes=False,
                       fill="#CCCCCC", alpha=0.5, color="grey", size=0.4)
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + labs(x="log1p(expression)" if log1p else "Expression",
               y="Density", color="Antibody")
    )

    if palette is not None:
        p = p + scale_color_manual(breaks=list(palette.keys()), values=list(palette.values()))
    else:
        p = p + scale_color_brewer(type="qual", palette="Set3")

    if group_by is not None and group_by in obs_df.columns:
        p = p + facet_wrap(group_by, ncol=ncol)

    if title is not None:
        p = p + ggtitle(title)

    return p
