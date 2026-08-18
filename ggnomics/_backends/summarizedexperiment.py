"""SummarizedExperiment registrations for metadata and expression plots.

SummarizedExperiment does not store cell/sample embeddings, so
embedding-dependent generics (``plot_scatter``, ``plot_embedding``,
``plot_pairs``, ``plot_embedding_panel``, ``plot_clonotype_embedding``) are
intentionally not registered here. Calling those with an SE raises the
normal unsupported-type ``TypeError``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from summarizedexperiment import SummarizedExperiment

from ..abundance import plot_abundance
from ..coldata import plot_coldata, plot_rowdata
from ..expression import plot_dot, plot_expression, plot_heatmap
from ..highest_exprs import (
    _build_highest_exprs_plot,
    _long_fraction_frame,
    _sparse_aware_top_n_medians,
    plot_highest_exprs,
)
from ..multimodal import plot_adt_qc
from ..pseudobulk import (
    _counts_panel_data,
    _counts_panel_plot,
    _libsize_panel_data,
    _libsize_panel_plot,
    _pca_panel_data,
    _pca_panel_plot,
    _pseudobulk_aggregate,
    _sample_condition_map,
    _sparse_row_sums,
    plot_pseudobulk_qc,
)
from ..repertoire import plot_clonotype_abundance, plot_clonotype_overlap
from ..stats_plots import plot_box_stats, plot_scatter_marginal, plot_violin_stats
from ..violin import expression_violin, expression_violin_se
from ._bioc import column_data_frame, row_data_frame


def _row_names(se: SummarizedExperiment) -> list[str]:
    for attribute in ("row_names", "feature_names", "gene_names"):
        names = getattr(se, attribute, None)
        if names is not None:
            return list(names)
    return [str(index) for index in range(se.shape[0])]


def _resolve_assay(se: SummarizedExperiment, layer: Optional[str]):
    assay_name = layer if layer is not None else "counts"
    try:
        return se.assay(assay_name)
    except (KeyError, ValueError, AttributeError) as exc:
        available = list(getattr(se, "assay_names", []))
        raise KeyError(f"Assay {assay_name!r} not found in the SE. Available: {available}") from exc


def _expression_frame(
    se: SummarizedExperiment,
    features: List[str],
    layer: Optional[str],
) -> pd.DataFrame:
    """Slice only the requested features, densifying just that slice."""

    row_names = _row_names(se)
    name_to_index = {name: index for index, name in enumerate(row_names)}
    missing = [f for f in features if f not in name_to_index]
    if missing:
        raise KeyError(f"Feature(s) {missing} not found in se.row_names. Available (first 20): {row_names[:20]}")

    indices = [name_to_index[f] for f in features]
    matrix = _resolve_assay(se, layer)
    sliced = matrix[indices, :]
    if hasattr(sliced, "toarray"):
        sliced = sliced.toarray()
    sliced = np.asarray(sliced).T  # features x samples -> samples x features
    return pd.DataFrame(sliced, columns=features)


def _expression_vector(
    se: SummarizedExperiment,
    feature: str,
    layer: Optional[str],
) -> np.ndarray:
    row_names = _row_names(se)
    try:
        feature_index = row_names.index(feature)
    except ValueError as exc:
        raise KeyError(f"Feature {feature!r} not found in se.row_names.") from exc

    matrix = _resolve_assay(se, layer)
    vector = matrix[feature_index, :]
    if hasattr(vector, "toarray"):
        vector = vector.toarray()
    return np.asarray(vector).reshape(-1)


def _resolve_column(
    se: SummarizedExperiment,
    key: str,
    layer: Optional[str] = None,
) -> tuple[pd.Series, bool]:
    obs = column_data_frame(se)
    if key in obs.columns:
        series = obs[key].reset_index(drop=True)
        return series, bool(pd.api.types.is_numeric_dtype(series))

    row_names = _row_names(se)
    if key in row_names:
        return pd.Series(_expression_vector(se, key, layer)), True

    raise KeyError(
        f"{key!r} not found in SE column_data or row_names. "
        f"Column-data names (first 20): {list(obs.columns)[:20]}. "
        f"Feature names (first 20): {row_names[:20]}"
    )


def _obs_expression_frame(
    se: SummarizedExperiment,
    features: List[str],
    layer: Optional[str],
    metadata_columns: List[str],
) -> pd.DataFrame:
    collisions = sorted(set(features) & set(metadata_columns))
    if collisions:
        raise ValueError(
            f"Name(s) {collisions} are both requested feature(s) and SE "
            "column_data metadata column(s), which is ambiguous."
        )
    obs = column_data_frame(se)
    missing_meta = [c for c in metadata_columns if c not in obs.columns]
    if missing_meta:
        raise KeyError(f"Column(s) {missing_meta} not found in SE column_data. Available: {list(obs.columns)[:20]}")

    frame = _expression_frame(se, features, layer)
    for column in metadata_columns:
        frame[column] = obs[column].to_numpy()
    return frame


@plot_expression.register(SummarizedExperiment)
def _plot_expression_se(
    data: SummarizedExperiment,
    features: List[str],
    group_by: str,
    layer: Optional[str] = None,
    color_by: Optional[str] = None,
    **kwargs,
):
    """SummarizedExperiment adapter for :func:`ggnomics.plot_expression`."""

    fill_col = color_by if color_by is not None else group_by
    metadata_columns = list(dict.fromkeys([group_by, fill_col]))
    frame = _obs_expression_frame(data, features, layer, metadata_columns)
    return plot_expression(frame, features=features, group_by=group_by, layer=None, color_by=color_by, **kwargs)


@plot_dot.register(SummarizedExperiment)
def _plot_dot_se(
    data: SummarizedExperiment,
    features: List[str],
    group_by: str,
    layer: Optional[str] = None,
    **kwargs,
):
    """SummarizedExperiment adapter for :func:`ggnomics.plot_dot`."""

    frame = _obs_expression_frame(data, features, layer, [group_by])
    return plot_dot(frame, features=features, group_by=group_by, layer=None, **kwargs)


@plot_heatmap.register(SummarizedExperiment)
def _plot_heatmap_se(
    data: SummarizedExperiment,
    features: List[str],
    group_by: Optional[str] = None,
    layer: Optional[str] = None,
    **kwargs,
):
    """SummarizedExperiment adapter for :func:`ggnomics.plot_heatmap`."""

    metadata_columns = [group_by] if group_by is not None else []
    frame = _obs_expression_frame(data, features, layer, metadata_columns)
    return plot_heatmap(frame, features=features, group_by=group_by, layer=None, **kwargs)


@plot_highest_exprs.register(SummarizedExperiment)
def _plot_highest_exprs_se(
    data: SummarizedExperiment,
    n: int = 50,
    layer: Optional[str] = None,
    features: Optional[List[str]] = None,
    color_cells_by: Optional[str] = None,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
):
    """SummarizedExperiment adapter for :func:`ggnomics.plot_highest_exprs`.

    Library size, per-feature medians, and ranking are computed on the
    sparse assay without densifying it; only the selected top-``n`` columns
    are ever converted to a dense array.
    """

    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    obs = column_data_frame(data)
    if color_cells_by is not None and color_cells_by not in obs.columns:
        raise KeyError(
            f"color_cells_by {color_cells_by!r} not found in SE column_data. Available: {list(obs.columns)[:20]}"
        )

    row_names = _row_names(data)
    matrix = _resolve_assay(data, layer)
    if features is not None:
        name_to_index = {name: index for index, name in enumerate(row_names)}
        missing = [f for f in features if f not in name_to_index]
        if missing:
            raise KeyError(f"features not found in se.row_names: {missing}. Available (first 20): {row_names[:20]}")
        indices = [name_to_index[f] for f in features]
        matrix = matrix[indices, :]
        candidate_names = list(features)
    else:
        candidate_names = row_names

    matrix = matrix.T  # features x samples -> samples x features

    top_local_idx, top_fraction = _sparse_aware_top_n_medians(matrix, n)
    top_genes = [candidate_names[i] for i in top_local_idx]

    long_df = _long_fraction_frame(top_fraction, top_genes, obs.reset_index(drop=True), color_cells_by)
    return _build_highest_exprs_plot(long_df, top_genes, color_cells_by, palette, title)


@plot_violin_stats.register(SummarizedExperiment)
def _plot_violin_stats_se(
    data: SummarizedExperiment,
    feature: str,
    group_by: str,
    layer: Optional[str] = None,
    **kwargs,
):
    """SummarizedExperiment adapter for :func:`ggnomics.plot_violin_stats`."""

    feature_values, _ = _resolve_column(data, feature, layer=layer)
    obs = column_data_frame(data)
    if group_by not in obs.columns:
        raise KeyError(f"group_by {group_by!r} not found in SE column_data. Available: {list(obs.columns)[:20]}")
    frame = pd.DataFrame(
        {
            feature: feature_values.to_numpy(),
            group_by: obs[group_by].to_numpy(),
        }
    )
    return plot_violin_stats(frame, feature=feature, group_by=group_by, layer=None, **kwargs)


@plot_box_stats.register(SummarizedExperiment)
def _plot_box_stats_se(
    data: SummarizedExperiment,
    feature: str,
    group_by: str,
    layer: Optional[str] = None,
    **kwargs,
):
    """SummarizedExperiment adapter for :func:`ggnomics.plot_box_stats`."""

    feature_values, _ = _resolve_column(data, feature, layer=layer)
    obs = column_data_frame(data)
    if group_by not in obs.columns:
        raise KeyError(f"group_by {group_by!r} not found in SE column_data. Available: {list(obs.columns)[:20]}")
    frame = pd.DataFrame(
        {
            feature: feature_values.to_numpy(),
            group_by: obs[group_by].to_numpy(),
        }
    )
    return plot_box_stats(frame, feature=feature, group_by=group_by, layer=None, **kwargs)


@plot_scatter_marginal.register(SummarizedExperiment)
def _plot_scatter_marginal_se(
    data: SummarizedExperiment,
    x: str,
    y: str,
    color: Optional[str] = None,
    layer: Optional[str] = None,
    **kwargs,
):
    """SummarizedExperiment adapter for :func:`ggnomics.plot_scatter_marginal`."""

    x_values, _ = _resolve_column(data, x)
    y_values, _ = _resolve_column(data, y)
    frame = pd.DataFrame({x: x_values.to_numpy(), y: y_values.to_numpy()})
    if color is not None:
        color_values, _ = _resolve_column(data, color, layer=layer)
        frame[color] = color_values.to_numpy()
    return plot_scatter_marginal(frame, x=x, y=y, color=color, layer=None, **kwargs)


@plot_clonotype_abundance.register(SummarizedExperiment)
def _plot_clonotype_abundance_se(data: SummarizedExperiment, *args, **kwargs):
    """SummarizedExperiment adapter for :func:`ggnomics.plot_clonotype_abundance`."""

    return plot_clonotype_abundance(column_data_frame(data), *args, **kwargs)


@plot_clonotype_overlap.register(SummarizedExperiment)
def _plot_clonotype_overlap_se(data: SummarizedExperiment, *args, **kwargs):
    """SummarizedExperiment adapter for :func:`ggnomics.plot_clonotype_overlap`."""

    return plot_clonotype_overlap(column_data_frame(data), *args, **kwargs)


@plot_pseudobulk_qc.register(SummarizedExperiment)
def _plot_pseudobulk_qc_se(
    data: SummarizedExperiment,
    sample_by: str,
    group_by: str,
    condition_by: Optional[str] = None,
    min_cells: int = 10,
    palette=None,
    ncol: int = 2,
):
    """SummarizedExperiment adapter for :func:`ggnomics.plot_pseudobulk_qc`.

    Library size and pseudobulk aggregation are computed via sparse-aware
    row sums / indicator-matrix products; the complete assay is never
    densified.
    """

    obs = column_data_frame(data)
    required = [sample_by, group_by] + ([condition_by] if condition_by is not None else [])
    missing = [c for c in required if c not in obs.columns]
    if missing:
        raise KeyError(f"Column(s) {missing} not found in SE column_data. Available: {list(obs.columns)[:20]}")

    obs_df = obs.reset_index(drop=True)
    counts_df = _counts_panel_data(obs_df, sample_by, group_by)
    p1 = _counts_panel_plot(counts_df, sample_by, group_by, min_cells, palette)

    matrix = _resolve_assay(data, None).T  # features x samples -> samples x features
    library_sizes = _sparse_row_sums(matrix)
    condition_values = obs_df[condition_by].to_numpy() if condition_by is not None else None
    libsize_df = _libsize_panel_data(
        obs_df[sample_by].to_numpy(), library_sizes, sample_by, condition_by, condition_values
    )
    p2 = _libsize_panel_plot(libsize_df, sample_by, condition_by, palette)

    sample_ids = obs_df[sample_by].to_numpy()
    unique_samples = pd.unique(sample_ids)
    pb_mat = _pseudobulk_aggregate(matrix, sample_ids, unique_samples)
    condition_map = (
        _sample_condition_map(obs_df, sample_by, condition_by, unique_samples) if condition_by is not None else None
    )
    pca_df = _pca_panel_data(pb_mat, unique_samples, sample_by, condition_by, condition_map)
    p3 = _pca_panel_plot(pca_df, sample_by, condition_by)

    return (p1 | p2) / p3


@plot_adt_qc.register(SummarizedExperiment)
def _plot_adt_qc_se(
    data: SummarizedExperiment,
    isotype_controls: List[str],
    layer: Optional[str] = None,
    group_by: Optional[str] = None,
    log1p: bool = True,
    palette=None,
    ncol: int = 3,
    title: Optional[str] = None,
    features: Optional[List[str]] = None,
    *,
    mod: str = "prot",
):
    """SummarizedExperiment adapter for :func:`ggnomics.plot_adt_qc`."""

    del mod
    if not isotype_controls:
        raise ValueError("isotype_controls must be a non-empty list.")

    row_names = _row_names(data)
    candidate_features = list(features) if features is not None else row_names
    missing_features = [f for f in candidate_features if f not in row_names]
    if missing_features:
        raise KeyError(
            f"features not found in se.row_names: {missing_features}. Available (first 20): {row_names[:20]}"
        )
    missing_iso = [f for f in isotype_controls if f not in candidate_features]
    if missing_iso:
        raise KeyError(f"isotype_controls not found among candidate features: {missing_iso}.")
    obs = column_data_frame(data)
    if group_by is not None and group_by not in obs.columns:
        raise KeyError(f"group_by {group_by!r} not found in SE column_data. Available: {list(obs.columns)[:20]}")

    frame = _expression_frame(data, candidate_features, layer)
    if group_by is not None:
        frame[group_by] = obs[group_by].to_numpy()

    return plot_adt_qc(
        frame,
        isotype_controls=isotype_controls,
        layer=None,
        group_by=group_by,
        log1p=log1p,
        palette=palette,
        ncol=ncol,
        title=title,
        features=candidate_features,
    )


@expression_violin.register(SummarizedExperiment)
def _expression_violin_se_dispatch(
    se: SummarizedExperiment,
    gene: str,
    group_col: str,
    *,
    assay: str = "logcounts",
    log1p: bool = False,
    title: Optional[str] = None,
):
    """SE dispatch entry for the legacy :func:`ggnomics.expression_violin`."""

    return expression_violin_se(se, gene, group_col, assay=assay, log1p=log1p, title=title)


@plot_coldata.register(SummarizedExperiment)
def _plot_coldata_se(data: SummarizedExperiment, *args, **kwargs):
    """Delegate SE column metadata to the DataFrame implementation."""

    return plot_coldata(column_data_frame(data), *args, **kwargs)


@plot_rowdata.register(SummarizedExperiment)
def _plot_rowdata_se(data: SummarizedExperiment, *args, **kwargs):
    """Delegate SE row metadata to the DataFrame implementation."""

    return plot_rowdata(row_data_frame(data), *args, **kwargs)


@plot_abundance.register(SummarizedExperiment)
def _plot_abundance_se(data: SummarizedExperiment, *args, **kwargs):
    """Delegate SE column metadata to the DataFrame implementation."""

    return plot_abundance(column_data_frame(data), *args, **kwargs)


__all__: list[str] = []
