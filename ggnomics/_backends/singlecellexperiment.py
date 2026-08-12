"""SingleCellExperiment registrations for scatter, embedding, and expression plots."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from singlecellexperiment import SingleCellExperiment

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
from ..pairs import _pairs_plot_from_embedding, plot_pairs
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
from ..repertoire import (
    _clonotype_embedding_plot,
    plot_clonotype_abundance,
    plot_clonotype_embedding,
    plot_clonotype_overlap,
)
from ..scatter import (
    _embedding_key_candidates,
    _strip_x_prefix,
    plot_embedding,
    plot_scatter,
)
from ..stats_plots import (
    plot_box_stats,
    plot_embedding_panel,
    plot_scatter_marginal,
    plot_violin_stats,
)
from ..violin import expression_violin, expression_violin_sce
from ._bioc import column_data_frame, row_data_frame


def _column_data(sce: SingleCellExperiment) -> pd.DataFrame:
    column_data = getattr(sce, "column_data", None)
    if column_data is None:
        column_data = getattr(sce, "col_data", None)
    if column_data is None:
        return pd.DataFrame(index=range(sce.shape[1]))

    if hasattr(column_data, "to_pandas"):
        frame = column_data.to_pandas().copy()
    else:
        frame = pd.DataFrame(column_data).copy()

    if "rownames" in frame.columns:
        frame = frame.drop(columns="rownames")
    return frame.reset_index(drop=True)


def _row_names(sce: SingleCellExperiment) -> list[str]:
    for attribute in ("row_names", "feature_names", "gene_names"):
        names = getattr(sce, attribute, None)
        if names is not None:
            return list(names)
    return [str(index) for index in range(sce.shape[0])]


def _expression_vector(
    sce: SingleCellExperiment,
    feature: str,
    layer: Optional[str],
) -> np.ndarray:
    row_names = _row_names(sce)
    try:
        feature_index = row_names.index(feature)
    except ValueError as exc:
        raise KeyError(f"Feature {feature!r} not found in sce.row_names.") from exc

    matrix = _resolve_assay(sce, layer)

    vector = matrix[feature_index, :]
    if hasattr(vector, "toarray"):
        vector = vector.toarray()
    return np.asarray(vector).reshape(-1)


def _resolve_column(
    sce: SingleCellExperiment,
    key: str,
    layer: Optional[str] = None,
) -> tuple[pd.Series, bool]:
    obs = _column_data(sce)
    if key in obs.columns:
        series = obs[key].reset_index(drop=True)
        return series, bool(pd.api.types.is_numeric_dtype(series))

    row_names = _row_names(sce)
    if key in row_names:
        return pd.Series(_expression_vector(sce, key, layer)), True

    raise KeyError(
        f"{key!r} not found in SCE column_data or row_names. "
        f"Column-data names (first 20): {list(obs.columns)[:20]}. "
        f"Feature names (first 20): {row_names[:20]}"
    )


def _mapping_embedding(
    reduced_dimensions,
    candidates: list[str],
) -> np.ndarray | None:
    if not isinstance(reduced_dimensions, Mapping):
        return None

    keys_by_lower = {str(key).lower(): key for key in reduced_dimensions.keys()}
    for candidate in candidates:
        if candidate in reduced_dimensions:
            return np.asarray(reduced_dimensions[candidate])
        if candidate.lower() in keys_by_lower:
            return np.asarray(reduced_dimensions[keys_by_lower[candidate.lower()]])
    return None


def _embedding_frame(sce: SingleCellExperiment, key: str) -> pd.DataFrame:
    candidates = _embedding_key_candidates(key)
    values = None

    for attribute in ("reduced_dimensions", "reduced_dims"):
        reduced_dimensions = getattr(sce, attribute, None)
        values = _mapping_embedding(reduced_dimensions, candidates)
        if values is not None:
            break

    if values is None:
        for getter_name in ("get_reduced_dimension", "reduced_dim"):
            getter = getattr(sce, getter_name, None)
            if getter is None:
                continue
            for candidate in candidates:
                try:
                    candidate_values = getter(candidate)
                except (KeyError, ValueError, AttributeError):
                    continue
                if candidate_values is not None:
                    values = np.asarray(candidate_values)
                    break
            if values is not None:
                break

    if values is None:
        available = []
        for attribute in ("reduced_dimensions", "reduced_dims"):
            reduced_dimensions = getattr(sce, attribute, None)
            if isinstance(reduced_dimensions, Mapping):
                available.extend(map(str, reduced_dimensions.keys()))
        raise KeyError(
            f"Embedding {key!r} not found in SCE reduced dimensions. "
            f"Available: {sorted(set(available))}"
        )

    if values.ndim != 2:
        raise ValueError(f"SCE reduced dimension {key!r} must be two-dimensional.")

    base = _strip_x_prefix(key).lower()
    return pd.DataFrame(
        values,
        columns=[f"{base}_{index + 1}" for index in range(values.shape[1])],
    )


def _resolve_assay(sce: SingleCellExperiment, layer: Optional[str]):
    assay_name = layer if layer is not None else "counts"
    try:
        return sce.assay(assay_name)
    except (KeyError, ValueError, AttributeError) as exc:
        available = list(getattr(sce, "assay_names", []))
        raise KeyError(
            f"Assay {assay_name!r} not found in the SCE. Available: {available}"
        ) from exc


def _expression_frame(
    sce: SingleCellExperiment,
    features: List[str],
    layer: Optional[str],
) -> pd.DataFrame:
    """Slice only the requested features, densifying just that slice."""

    row_names = _row_names(sce)
    name_to_index = {name: index for index, name in enumerate(row_names)}
    missing = [f for f in features if f not in name_to_index]
    if missing:
        raise KeyError(
            f"Feature(s) {missing} not found in sce.row_names. "
            f"Available (first 20): {row_names[:20]}"
        )

    indices = [name_to_index[f] for f in features]
    matrix = _resolve_assay(sce, layer)
    sliced = matrix[indices, :]
    if hasattr(sliced, "toarray"):
        sliced = sliced.toarray()
    sliced = np.asarray(sliced).T  # features x obs -> obs x features
    return pd.DataFrame(sliced, columns=features)


def _obs_expression_frame(
    sce: SingleCellExperiment,
    features: List[str],
    layer: Optional[str],
    metadata_columns: List[str],
) -> pd.DataFrame:
    collisions = sorted(set(features) & set(metadata_columns))
    if collisions:
        raise ValueError(
            f"Name(s) {collisions} are both requested feature(s) and SCE "
            "column_data metadata column(s), which is ambiguous."
        )
    obs = _column_data(sce)
    missing_meta = [c for c in metadata_columns if c not in obs.columns]
    if missing_meta:
        raise KeyError(
            f"Column(s) {missing_meta} not found in SCE column_data. "
            f"Available: {list(obs.columns)[:20]}"
        )

    frame = _expression_frame(sce, features, layer)
    for column in metadata_columns:
        frame[column] = obs[column].to_numpy()
    return frame


@plot_expression.register(SingleCellExperiment)
def _plot_expression_sce(
    data: SingleCellExperiment,
    features: List[str],
    group_by: str,
    layer: Optional[str] = None,
    color_by: Optional[str] = None,
    **kwargs,
):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_expression`."""

    fill_col = color_by if color_by is not None else group_by
    metadata_columns = list(dict.fromkeys([group_by, fill_col]))
    frame = _obs_expression_frame(data, features, layer, metadata_columns)
    return plot_expression(
        frame, features=features, group_by=group_by, layer=None, color_by=color_by, **kwargs
    )


@plot_dot.register(SingleCellExperiment)
def _plot_dot_sce(
    data: SingleCellExperiment,
    features: List[str],
    group_by: str,
    layer: Optional[str] = None,
    **kwargs,
):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_dot`."""

    frame = _obs_expression_frame(data, features, layer, [group_by])
    return plot_dot(frame, features=features, group_by=group_by, layer=None, **kwargs)


@plot_heatmap.register(SingleCellExperiment)
def _plot_heatmap_sce(
    data: SingleCellExperiment,
    features: List[str],
    group_by: Optional[str] = None,
    layer: Optional[str] = None,
    **kwargs,
):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_heatmap`."""

    metadata_columns = [group_by] if group_by is not None else []
    frame = _obs_expression_frame(data, features, layer, metadata_columns)
    return plot_heatmap(frame, features=features, group_by=group_by, layer=None, **kwargs)


@plot_highest_exprs.register(SingleCellExperiment)
def _plot_highest_exprs_sce(
    data: SingleCellExperiment,
    n: int = 50,
    layer: Optional[str] = None,
    features: Optional[List[str]] = None,
    color_cells_by: Optional[str] = None,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_highest_exprs`.

    Library size, per-feature medians, and ranking are computed on the
    sparse assay without densifying it; only the selected top-``n`` columns
    are ever converted to a dense array.
    """

    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    obs = _column_data(data)
    if color_cells_by is not None and color_cells_by not in obs.columns:
        raise KeyError(
            f"color_cells_by {color_cells_by!r} not found in SCE column_data. "
            f"Available: {list(obs.columns)[:20]}"
        )

    row_names = _row_names(data)
    matrix = _resolve_assay(data, layer)
    if features is not None:
        name_to_index = {name: index for index, name in enumerate(row_names)}
        missing = [f for f in features if f not in name_to_index]
        if missing:
            raise KeyError(
                f"features not found in sce.row_names: {missing}. "
                f"Available (first 20): {row_names[:20]}"
            )
        indices = [name_to_index[f] for f in features]
        matrix = matrix[indices, :]
        candidate_names = list(features)
    else:
        candidate_names = row_names

    # Transpose to obs x features orientation (cheap view for sparse/dense).
    matrix = matrix.T

    top_local_idx, top_fraction = _sparse_aware_top_n_medians(matrix, n)
    top_genes = [candidate_names[i] for i in top_local_idx]

    long_df = _long_fraction_frame(top_fraction, top_genes, obs.reset_index(drop=True), color_cells_by)
    return _build_highest_exprs_plot(long_df, top_genes, color_cells_by, palette, title)


@plot_violin_stats.register(SingleCellExperiment)
def _plot_violin_stats_sce(
    data: SingleCellExperiment,
    feature: str,
    group_by: str,
    layer: Optional[str] = None,
    **kwargs,
):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_violin_stats`."""

    feature_values, _ = _resolve_column(data, feature, layer=layer)
    obs = _column_data(data)
    if group_by not in obs.columns:
        raise KeyError(
            f"group_by {group_by!r} not found in SCE column_data. "
            f"Available: {list(obs.columns)[:20]}"
        )
    frame = pd.DataFrame({
        feature: feature_values.to_numpy(),
        group_by: obs[group_by].to_numpy(),
    })
    return plot_violin_stats(frame, feature=feature, group_by=group_by, layer=None, **kwargs)


@plot_box_stats.register(SingleCellExperiment)
def _plot_box_stats_sce(
    data: SingleCellExperiment,
    feature: str,
    group_by: str,
    layer: Optional[str] = None,
    **kwargs,
):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_box_stats`."""

    feature_values, _ = _resolve_column(data, feature, layer=layer)
    obs = _column_data(data)
    if group_by not in obs.columns:
        raise KeyError(
            f"group_by {group_by!r} not found in SCE column_data. "
            f"Available: {list(obs.columns)[:20]}"
        )
    frame = pd.DataFrame({
        feature: feature_values.to_numpy(),
        group_by: obs[group_by].to_numpy(),
    })
    return plot_box_stats(frame, feature=feature, group_by=group_by, layer=None, **kwargs)


@plot_scatter_marginal.register(SingleCellExperiment)
def _plot_scatter_marginal_sce(
    data: SingleCellExperiment,
    x: str,
    y: str,
    color: Optional[str] = None,
    layer: Optional[str] = None,
    **kwargs,
):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_scatter_marginal`."""

    x_values, _ = _resolve_column(data, x)
    y_values, _ = _resolve_column(data, y)
    frame = pd.DataFrame({x: x_values.to_numpy(), y: y_values.to_numpy()})
    if color is not None:
        color_values, _ = _resolve_column(data, color, layer=layer)
        frame[color] = color_values.to_numpy()
    return plot_scatter_marginal(frame, x=x, y=y, color=color, layer=None, **kwargs)


@plot_embedding_panel.register(SingleCellExperiment)
def _plot_embedding_panel_sce(
    data: SingleCellExperiment,
    features: List[str],
    dimred: str = "X_umap",
    layer: Optional[str] = None,
    **kwargs,
):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_embedding_panel`."""

    if not features:
        raise ValueError("`features` must be a non-empty list of feature names.")

    frame = _embedding_frame(data, dimred)
    missing = []
    for feat in features:
        try:
            values, _ = _resolve_column(data, feat, layer=layer)
        except KeyError:
            missing.append(feat)
            continue
        frame[feat] = values.to_numpy()

    if missing:
        raise ValueError(
            f"The following features could not be resolved in SCE column_data "
            f"or row_names: {missing}"
        )

    return plot_embedding_panel(frame, features=features, dimred=dimred, layer=None, **kwargs)


@plot_pairs.register(SingleCellExperiment)
def _plot_pairs_sce(
    data: SingleCellExperiment,
    dimred: str = "X_pca",
    n_components: int = 4,
    color_by: Optional[str] = None,
    size: Optional[float] = None,
    alpha: float = 0.6,
    palette=None,
    title: Optional[str] = None,
):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_pairs`."""

    emb_df = _embedding_frame(data, dimred)
    obs = _column_data(data)
    color_series = None
    if color_by is not None:
        if color_by not in obs.columns:
            raise KeyError(
                f"color_by {color_by!r} not found in SCE column_data. "
                f"Available: {list(obs.columns)[:20]}"
            )
        color_series = obs[color_by].reset_index(drop=True)

    return _pairs_plot_from_embedding(
        emb_df, n_components, color_series, color_by, size, alpha, palette, title
    )


@plot_clonotype_abundance.register(SingleCellExperiment)
def _plot_clonotype_abundance_sce(data: SingleCellExperiment, *args, **kwargs):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_clonotype_abundance`."""

    return plot_clonotype_abundance(column_data_frame(data), *args, **kwargs)


@plot_clonotype_overlap.register(SingleCellExperiment)
def _plot_clonotype_overlap_sce(data: SingleCellExperiment, *args, **kwargs):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_clonotype_overlap`."""

    return plot_clonotype_overlap(column_data_frame(data), *args, **kwargs)


@plot_clonotype_embedding.register(SingleCellExperiment)
def _plot_clonotype_embedding_sce(
    data: SingleCellExperiment,
    clonotype_col: str,
    dimred: str = "X_umap",
    components=(1, 2),
    expansion_thresholds=None,
    non_tcell_color: str = "#DDDDDD",
    palette=None,
    size: Optional[float] = None,
    stroke: Optional[float] = None,
    alpha: float = 0.8,
    title: Optional[str] = None,
):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_clonotype_embedding`."""

    obs = _column_data(data)
    if clonotype_col not in obs.columns:
        raise KeyError(
            f"clonotype_col {clonotype_col!r} not found in SCE column_data. "
            f"Available: {list(obs.columns)[:20]}"
        )
    emb_df = _embedding_frame(data, dimred)
    return _clonotype_embedding_plot(
        emb_df, components, obs[clonotype_col], dimred, expansion_thresholds,
        non_tcell_color, palette, size, stroke, alpha, title,
    )


@plot_pseudobulk_qc.register(SingleCellExperiment)
def _plot_pseudobulk_qc_sce(
    data: SingleCellExperiment,
    sample_by: str,
    group_by: str,
    condition_by: Optional[str] = None,
    min_cells: int = 10,
    palette=None,
    ncol: int = 2,
):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_pseudobulk_qc`.

    Library size and pseudobulk aggregation are computed via sparse-aware
    row sums / indicator-matrix products; the complete assay is never
    densified.
    """

    obs = _column_data(data)
    required = [sample_by, group_by] + ([condition_by] if condition_by is not None else [])
    missing = [c for c in required if c not in obs.columns]
    if missing:
        raise KeyError(
            f"Column(s) {missing} not found in SCE column_data. "
            f"Available: {list(obs.columns)[:20]}"
        )

    obs_df = obs.reset_index(drop=True)
    counts_df = _counts_panel_data(obs_df, sample_by, group_by)
    p1 = _counts_panel_plot(counts_df, sample_by, group_by, min_cells, palette)

    matrix = _resolve_assay(data, None).T  # features x obs -> obs x features
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
        _sample_condition_map(obs_df, sample_by, condition_by, unique_samples)
        if condition_by is not None else None
    )
    pca_df = _pca_panel_data(pb_mat, unique_samples, sample_by, condition_by, condition_map)
    p3 = _pca_panel_plot(pca_df, sample_by, condition_by)

    return (p1 | p2) / p3


@plot_adt_qc.register(SingleCellExperiment)
def _plot_adt_qc_sce(
    data: SingleCellExperiment,
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
    """SingleCellExperiment adapter for :func:`ggnomics.plot_adt_qc`."""

    del mod
    if not isotype_controls:
        raise ValueError("isotype_controls must be a non-empty list.")

    row_names = _row_names(data)
    candidate_features = list(features) if features is not None else row_names
    missing_features = [f for f in candidate_features if f not in row_names]
    if missing_features:
        raise KeyError(
            f"features not found in sce.row_names: {missing_features}. "
            f"Available (first 20): {row_names[:20]}"
        )
    missing_iso = [f for f in isotype_controls if f not in candidate_features]
    if missing_iso:
        raise KeyError(f"isotype_controls not found among candidate features: {missing_iso}.")
    obs = _column_data(data)
    if group_by is not None and group_by not in obs.columns:
        raise KeyError(
            f"group_by {group_by!r} not found in SCE column_data. "
            f"Available: {list(obs.columns)[:20]}"
        )

    frame = _expression_frame(data, candidate_features, layer)
    if group_by is not None:
        frame[group_by] = obs[group_by].to_numpy()

    return plot_adt_qc(
        frame, isotype_controls=isotype_controls, layer=None, group_by=group_by,
        log1p=log1p, palette=palette, ncol=ncol, title=title, features=candidate_features,
    )


@expression_violin.register(SingleCellExperiment)
def _expression_violin_sce_dispatch(
    sce: SingleCellExperiment,
    gene: str,
    group_col: str,
    *,
    assay: str = "logcounts",
    log1p: bool = False,
    title: Optional[str] = None,
):
    """SCE dispatch entry for the legacy :func:`ggnomics.expression_violin`."""

    return expression_violin_sce(sce, gene, group_col, assay=assay, log1p=log1p, title=title)


@plot_coldata.register(SingleCellExperiment)
def _plot_coldata_sce(data: SingleCellExperiment, *args, **kwargs):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_coldata`."""

    return plot_coldata(column_data_frame(data), *args, **kwargs)


@plot_rowdata.register(SingleCellExperiment)
def _plot_rowdata_sce(data: SingleCellExperiment, *args, **kwargs):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_rowdata`."""

    return plot_rowdata(row_data_frame(data), *args, **kwargs)


@plot_abundance.register(SingleCellExperiment)
def _plot_abundance_sce(data: SingleCellExperiment, *args, **kwargs):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_abundance`."""

    return plot_abundance(column_data_frame(data), *args, **kwargs)


@plot_scatter.register(SingleCellExperiment)
def _plot_scatter_sce(
    data: SingleCellExperiment,
    x: str,
    y: str,
    color: Optional[str] = None,
    **kwargs,
):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_scatter`."""

    layer = kwargs.pop("layer", None)
    facet_by = kwargs.get("facet_by")
    obs = _column_data(data)

    x_values, _ = _resolve_column(data, x)
    y_values, _ = _resolve_column(data, y)
    frame = pd.DataFrame({x: x_values.to_numpy(), y: y_values.to_numpy()})

    if color is not None:
        color_values, _ = _resolve_column(data, color, layer=layer)
        frame[color] = color_values.to_numpy()

    if facet_by is not None:
        if facet_by not in obs.columns:
            raise KeyError(
                f"facet_by {facet_by!r} not found in SCE column_data. "
                f"Available: {list(obs.columns)[:20]}"
            )
        frame[facet_by] = obs[facet_by].to_numpy()

    return plot_scatter(frame, x=x, y=y, color=color, layer=None, **kwargs)


@plot_embedding.register(SingleCellExperiment)
def _plot_embedding_sce(
    data: SingleCellExperiment,
    dimred: str = "X_umap",
    color: Optional[str] = None,
    **kwargs,
):
    """SingleCellExperiment adapter for :func:`ggnomics.plot_embedding`."""

    layer = kwargs.pop("layer", None)
    facet_by = kwargs.get("facet_by")
    obs = _column_data(data)
    frame = _embedding_frame(data, dimred)

    for column in obs.columns:
        if column not in frame.columns:
            frame[column] = obs[column].to_numpy()

    if color is not None and color not in obs.columns:
        color_values, _ = _resolve_column(data, color, layer=layer)
        frame[color] = color_values.to_numpy()

    if facet_by is not None and facet_by not in obs.columns:
        raise KeyError(
            f"facet_by {facet_by!r} not found in SCE column_data. "
            f"Available: {list(obs.columns)[:20]}"
        )

    return plot_embedding(frame, dimred=dimred, color=color, layer=None, **kwargs)
