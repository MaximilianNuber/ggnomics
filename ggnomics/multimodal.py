"""Multimodal plots: bimodal scatter (e.g. RNA vs protein) and ADT QC."""

from __future__ import annotations

from functools import singledispatch
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_density,
    ggplot,
    ggtitle,
    labs,
    scale_color_brewer,
    scale_color_manual,
    theme,
    theme_classic,
)

if TYPE_CHECKING:
    from plotnine.composition import Compose


def _unsupported_type(function_name: str, data: object) -> TypeError:
    return TypeError(
        f"{function_name} does not support {type(data).__module__}."
        f"{type(data).__qualname__}. Pass a pandas.DataFrame or install the "
        "optional dependency for a supported genomics container."
    )


# ---------------------------------------------------------------------------
# plot_bimodal_scatter
# ---------------------------------------------------------------------------


@singledispatch
def plot_bimodal_scatter(
    data: pd.DataFrame,
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
) -> Union[ggplot, Compose]:
    """Scatter plot of two features, typically from different modalities.

    Designed for CITE-seq-style data (e.g. RNA vs surface protein). If
    ``add_marginal=True``, delegates to :func:`ggnomics.plot_scatter_marginal`
    and returns a ``plotnine.composition.Compose`` instead of a ``ggplot``.

    Args:
        data: DataFrame whose rows are cells. ``x_feature``, ``y_feature``,
            and ``color`` are ordinary columns in the same DataFrame.
        x_feature: Column mapped to the x-axis.
        y_feature: Column mapped to the y-axis.
        x_mod: Present for cross-container signature consistency (selects
            the modality/layer for ``x_feature`` in AnnData/MuData). Has no
            effect for a plain DataFrame.
        y_mod: Same as ``x_mod`` for ``y_feature``.
        color: Optional column mapped to point color.
        size: Point size (``None`` -> adaptive).
        stroke: Point stroke width (``None`` -> adaptive).
        alpha: Point transparency.
        layer_x: Present for cross-container signature consistency. Has no
            effect for a plain DataFrame.
        layer_y: Same as ``layer_x`` for ``y_feature``.
        add_marginal: Return a scatter-with-marginals composition instead of
            a bare scatter plot.
        palette: ``{category: hex}`` color mapping.
        cmap: Matplotlib colormap for a continuous ``color``.
        title: Plot title.
        x_label: Override x-axis label (defaults to ``x_feature``).
        y_label: Override y-axis label (defaults to ``y_feature``).

    Returns:
        A ``plotnine.ggplot`` object, or a ``plotnine.composition.Compose`` when ``add_marginal=True``.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        KeyError: If a requested column is absent.
    """
    raise _unsupported_type("plot_bimodal_scatter", data)


@plot_bimodal_scatter.register(pd.DataFrame)
def _plot_bimodal_scatter_dataframe(
    data: pd.DataFrame,
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
    del x_mod, y_mod, layer_x, layer_y  # meaningful only for container backends

    required = [x_feature, y_feature] + ([color] if color is not None else [])
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(f"Column(s) {missing} not found in the DataFrame. Available: {list(data.columns)[:20]}")

    if add_marginal:
        from .stats_plots import plot_scatter_marginal

        return plot_scatter_marginal(
            data,
            x=x_feature,
            y=y_feature,
            color=color,
            layer=None,
            alpha=alpha,
            palette=palette,
            cmap=cmap,
            title=title,
            x_label=x_label or x_feature,
            y_label=y_label or y_feature,
        )

    from .scatter import plot_scatter

    return plot_scatter(
        data,
        x=x_feature,
        y=y_feature,
        color=color,
        size=size,
        stroke=stroke,
        alpha=alpha,
        palette=palette,
        cmap=cmap,
        layer=None,
        x_label=x_label or x_feature,
        y_label=y_label or y_feature,
        title=title,
    )


# ---------------------------------------------------------------------------
# plot_adt_qc
# ---------------------------------------------------------------------------


def _build_adt_qc_plot(
    real_long: pd.DataFrame,
    iso_df: pd.DataFrame,
    group_by: Optional[str],
    palette: Optional[Dict],
    ncol: int,
    title: Optional[str],
    log1p: bool,
) -> ggplot:
    """Shared plot construction, reused by every container adapter."""

    p = (
        ggplot(real_long)
        + aes(x="expression", color="antibody")
        + geom_density(alpha=0.0, size=0.6)
        + geom_density(
            data=iso_df,
            mapping=aes(x="expression"),
            inherit_aes=False,
            fill="#CCCCCC",
            alpha=0.5,
            color="grey",
            size=0.4,
        )
        + theme_classic()
        + theme(axis_text_x=element_text(rotation=45, ha="right"))
        + labs(
            x="log1p(expression)" if log1p else "Expression",
            y="Density",
            color="Antibody",
        )
    )

    if palette is not None:
        p = p + scale_color_manual(breaks=list(palette.keys()), values=list(palette.values()))
    else:
        p = p + scale_color_brewer(type="qual", palette="Set3")

    if group_by is not None:
        p = p + facet_wrap(group_by, ncol=ncol)

    if title is not None:
        p = p + ggtitle(title)

    return p


@singledispatch
def plot_adt_qc(
    data: pd.DataFrame,
    isotype_controls: List[str],
    layer: Optional[str] = None,
    group_by: Optional[str] = None,
    log1p: bool = True,
    palette: Optional[Dict] = None,
    ncol: int = 3,
    title: Optional[str] = None,
    features: Optional[List[str]] = None,
    *,
    mod: str = "prot",
) -> ggplot:
    """ADT / protein QC: overlay isotype-control distributions on real antibodies.

    For each real antibody, shows a density curve. Overlays the pooled
    isotype-control distribution as a shaded fill. If ``group_by`` is
    provided, facets by that column.

    Args:
        data: DataFrame whose rows are cells. ``features`` are the antibody
            expression columns; ``group_by`` is a metadata column.
        isotype_controls: Non-empty list of isotype-control feature names.
            Must be a subset of ``features``.
        layer: Present for cross-container signature consistency. Has no
            effect for a plain DataFrame, which has no concept of layers.
        group_by: Optional column used to facet the plot. Raises
            ``KeyError`` if supplied but absent (never silently ignored).
        log1p: Log1p-transform expression before plotting.
        palette: ``{antibody: hex}`` color mapping.
        ncol: Facet columns when ``group_by`` is given.
        title: Plot title.
        features: The antibody feature columns (real antibodies plus
            isotype controls). Required for a DataFrame — metadata columns
            are never guessed as antibodies.
        mod: Keyword-only. MuData modality to use (default ``"prot"``).
            Ignored for a plain DataFrame.

    Returns:
        A ``plotnine.ggplot`` object.

    Raises:
        TypeError: If no implementation is registered for ``type(data)``.
        ValueError: If ``isotype_controls`` is empty, ``features`` is
            omitted for a DataFrame, or no real antibody features remain.
        KeyError: If a requested column/feature is absent.
    """
    raise _unsupported_type("plot_adt_qc", data)


@plot_adt_qc.register(pd.DataFrame)
def _plot_adt_qc_dataframe(
    data: pd.DataFrame,
    isotype_controls: List[str],
    layer: Optional[str] = None,
    group_by: Optional[str] = None,
    log1p: bool = True,
    palette: Optional[Dict] = None,
    ncol: int = 3,
    title: Optional[str] = None,
    features: Optional[List[str]] = None,
    *,
    mod: str = "prot",
) -> ggplot:
    del layer, mod

    if not isotype_controls:
        raise ValueError("isotype_controls must be a non-empty list.")
    if features is None:
        raise ValueError(
            "`features` must be given for a DataFrame to identify the antibody "
            "feature columns; metadata columns are never guessed as antibodies."
        )

    missing_features = [f for f in features if f not in data.columns]
    if missing_features:
        raise KeyError(
            f"features not found in the DataFrame: {missing_features}. Available (first 20): {list(data.columns)[:20]}"
        )
    missing_iso = [f for f in isotype_controls if f not in features]
    if missing_iso:
        raise KeyError(f"isotype_controls not found among `features`: {missing_iso}.")
    if group_by is not None and group_by not in data.columns:
        raise KeyError(f"group_by {group_by!r} not found in the DataFrame. Available: {list(data.columns)[:20]}")

    real_features = [f for f in features if f not in isotype_controls]
    if not real_features:
        raise ValueError("No real antibody features found (all `features` listed as isotype_controls).")

    obs_df = data.reset_index(drop=True)
    expr_df = obs_df[features].copy()
    if log1p:
        expr_df = np.log1p(expr_df)

    id_vars = []
    if group_by is not None:
        expr_df[group_by] = obs_df[group_by].to_numpy()
        id_vars.append(group_by)

    real_long = expr_df[id_vars + real_features].melt(id_vars=id_vars, var_name="antibody", value_name="expression")

    iso_vals = expr_df[isotype_controls].to_numpy().ravel()
    iso_df = pd.DataFrame({"expression": iso_vals})
    if group_by is not None:
        iso_df[group_by] = np.tile(obs_df[group_by].to_numpy(), len(isotype_controls))

    return _build_adt_qc_plot(real_long, iso_df, group_by, palette, ncol, title, log1p)


__all__ = ["plot_bimodal_scatter", "plot_adt_qc"]
