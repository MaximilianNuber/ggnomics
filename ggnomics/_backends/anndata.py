"""AnnData registrations for scatter, embedding, and expression plots."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from anndata import AnnData

from ..abundance import plot_abundance
from ..coldata import plot_coldata, plot_rowdata
from ..expression import plot_dot, plot_expression, plot_heatmap
from ..highest_exprs import (
    _build_highest_exprs_plot,
    _long_fraction_frame,
    _sparse_aware_top_n_medians,
    plot_highest_exprs,
)
from ..multimodal import plot_adt_qc, plot_bimodal_scatter
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


def _expression_vector(
    adata: AnnData,
    feature: str,
    layer: Optional[str],
) -> np.ndarray:
    """Extract one feature without densifying the complete expression matrix."""

    try:
        feature_index = adata.var_names.get_loc(feature)
    except KeyError as exc:
        raise KeyError(f"Feature {feature!r} not found in adata.var_names.") from exc

    if layer is None:
        matrix = adata.X
    else:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} not found. Available: {list(adata.layers.keys())}")
        matrix = adata.layers[layer]

    vector = matrix[:, feature_index]
    if hasattr(vector, "toarray"):
        vector = vector.toarray()
    return np.asarray(vector).reshape(-1)


def _resolve_column(
    adata: AnnData,
    key: str,
    layer: Optional[str] = None,
) -> tuple[pd.Series, bool]:
    if key in adata.obs.columns:
        series = adata.obs[key].reset_index(drop=True)
        return series, bool(pd.api.types.is_numeric_dtype(series))

    if key in adata.var_names:
        return pd.Series(_expression_vector(adata, key, layer)), True

    raise KeyError(
        f"{key!r} not found in adata.obs or adata.var_names. "
        f"Obs columns (first 20): {list(adata.obs.columns)[:20]}. "
        f"Feature names (first 20): {list(adata.var_names)[:20]}"
    )


def _embedding_frame(adata: AnnData, key: str) -> pd.DataFrame:
    candidates = _embedding_key_candidates(key)
    keys_by_lower = {candidate.lower(): candidate for candidate in adata.obsm.keys()}

    found = next((candidate for candidate in candidates if candidate in adata.obsm), None)
    if found is None:
        found = next(
            (keys_by_lower[candidate.lower()] for candidate in candidates if candidate.lower() in keys_by_lower),
            None,
        )

    if found is None:
        raise KeyError(f"Embedding {key!r} not found in adata.obsm. Available: {list(adata.obsm.keys())}")

    values = np.asarray(adata.obsm[found])
    if values.ndim != 2:
        raise ValueError(f"adata.obsm[{found!r}] must be two-dimensional.")

    base = _strip_x_prefix(key).lower()
    return pd.DataFrame(
        values,
        columns=[f"{base}_{index + 1}" for index in range(values.shape[1])],
    )


def _resolve_matrix(adata: AnnData, layer: Optional[str]):
    if layer is None:
        return adata.X
    if layer not in adata.layers:
        raise KeyError(f"Layer {layer!r} not found. Available: {list(adata.layers.keys())}")
    return adata.layers[layer]


def _expression_frame(
    adata: AnnData,
    features: List[str],
    layer: Optional[str],
) -> pd.DataFrame:
    """Slice only the requested features, densifying just that slice."""

    missing = [f for f in features if f not in adata.var_names]
    if missing:
        raise KeyError(
            f"Feature(s) {missing} not found in adata.var_names. Available (first 20): {list(adata.var_names)[:20]}"
        )

    indices = adata.var_names.get_indexer(features)
    matrix = _resolve_matrix(adata, layer)
    sliced = matrix[:, indices]
    if hasattr(sliced, "toarray"):
        sliced = sliced.toarray()
    return pd.DataFrame(np.asarray(sliced), columns=features)


def _obs_expression_frame(
    adata: AnnData,
    features: List[str],
    layer: Optional[str],
    metadata_columns: List[str],
) -> pd.DataFrame:
    collisions = sorted(set(features) & set(metadata_columns))
    if collisions:
        raise ValueError(
            f"Name(s) {collisions} are both requested feature(s) and adata.obs metadata column(s), which is ambiguous."
        )
    missing_meta = [c for c in metadata_columns if c not in adata.obs.columns]
    if missing_meta:
        raise KeyError(f"Column(s) {missing_meta} not found in adata.obs. Available: {list(adata.obs.columns)[:20]}")

    frame = _expression_frame(adata, features, layer)
    for column in metadata_columns:
        frame[column] = adata.obs[column].to_numpy()
    return frame


@plot_expression.register(AnnData)
def _plot_expression_anndata(
    data: AnnData,
    features: List[str],
    group_by: str,
    layer: Optional[str] = None,
    color_by: Optional[str] = None,
    **kwargs,
):
    """AnnData adapter for :func:`ggnomics.plot_expression`."""

    fill_col = color_by if color_by is not None else group_by
    metadata_columns = list(dict.fromkeys([group_by, fill_col]))
    frame = _obs_expression_frame(data, features, layer, metadata_columns)
    return plot_expression(frame, features=features, group_by=group_by, layer=None, color_by=color_by, **kwargs)


@plot_dot.register(AnnData)
def _plot_dot_anndata(
    data: AnnData,
    features: List[str],
    group_by: str,
    layer: Optional[str] = None,
    **kwargs,
):
    """AnnData adapter for :func:`ggnomics.plot_dot`."""

    frame = _obs_expression_frame(data, features, layer, [group_by])
    return plot_dot(frame, features=features, group_by=group_by, layer=None, **kwargs)


@plot_heatmap.register(AnnData)
def _plot_heatmap_anndata(
    data: AnnData,
    features: List[str],
    group_by: Optional[str] = None,
    layer: Optional[str] = None,
    **kwargs,
):
    """AnnData adapter for :func:`ggnomics.plot_heatmap`."""

    metadata_columns = [group_by] if group_by is not None else []
    frame = _obs_expression_frame(data, features, layer, metadata_columns)
    return plot_heatmap(frame, features=features, group_by=group_by, layer=None, **kwargs)


@plot_highest_exprs.register(AnnData)
def _plot_highest_exprs_anndata(
    data: AnnData,
    n: int = 50,
    layer: Optional[str] = None,
    features: Optional[List[str]] = None,
    color_cells_by: Optional[str] = None,
    palette: Optional[Dict] = None,
    title: Optional[str] = None,
):
    """AnnData adapter for :func:`ggnomics.plot_highest_exprs`.

    Library size, per-feature medians, and ranking are computed on the
    sparse matrix without densifying it; only the selected top-``n``
    columns are ever converted to a dense array.
    """

    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if color_cells_by is not None and color_cells_by not in data.obs.columns:
        raise KeyError(
            f"color_cells_by {color_cells_by!r} not found in adata.obs. Available: {list(data.obs.columns)[:20]}"
        )

    matrix = _resolve_matrix(data, layer)
    if features is not None:
        missing = [f for f in features if f not in data.var_names]
        if missing:
            raise KeyError(
                f"features not found in adata.var_names: {missing}. Available (first 20): {list(data.var_names)[:20]}"
            )
        indices = data.var_names.get_indexer(features)
        matrix = matrix[:, indices]
        candidate_names = list(features)
    else:
        candidate_names = list(data.var_names)

    top_local_idx, top_fraction = _sparse_aware_top_n_medians(matrix, n)
    top_genes = [candidate_names[i] for i in top_local_idx]

    obs_df = data.obs.reset_index(drop=True)
    long_df = _long_fraction_frame(top_fraction, top_genes, obs_df, color_cells_by)
    return _build_highest_exprs_plot(long_df, top_genes, color_cells_by, palette, title)


@plot_violin_stats.register(AnnData)
def _plot_violin_stats_anndata(
    data: AnnData,
    feature: str,
    group_by: str,
    layer: Optional[str] = None,
    **kwargs,
):
    """AnnData adapter for :func:`ggnomics.plot_violin_stats`."""

    feature_values, _ = _resolve_column(data, feature, layer=layer)
    if group_by not in data.obs.columns:
        raise KeyError(f"group_by {group_by!r} not found in adata.obs. Available: {list(data.obs.columns)[:20]}")
    frame = pd.DataFrame(
        {
            feature: feature_values.to_numpy(),
            group_by: data.obs[group_by].to_numpy(),
        }
    )
    return plot_violin_stats(frame, feature=feature, group_by=group_by, layer=None, **kwargs)


@plot_box_stats.register(AnnData)
def _plot_box_stats_anndata(
    data: AnnData,
    feature: str,
    group_by: str,
    layer: Optional[str] = None,
    **kwargs,
):
    """AnnData adapter for :func:`ggnomics.plot_box_stats`."""

    feature_values, _ = _resolve_column(data, feature, layer=layer)
    if group_by not in data.obs.columns:
        raise KeyError(f"group_by {group_by!r} not found in adata.obs. Available: {list(data.obs.columns)[:20]}")
    frame = pd.DataFrame(
        {
            feature: feature_values.to_numpy(),
            group_by: data.obs[group_by].to_numpy(),
        }
    )
    return plot_box_stats(frame, feature=feature, group_by=group_by, layer=None, **kwargs)


@plot_scatter_marginal.register(AnnData)
def _plot_scatter_marginal_anndata(
    data: AnnData,
    x: str,
    y: str,
    color: Optional[str] = None,
    layer: Optional[str] = None,
    **kwargs,
):
    """AnnData adapter for :func:`ggnomics.plot_scatter_marginal`."""

    x_values, _ = _resolve_column(data, x)
    y_values, _ = _resolve_column(data, y)
    frame = pd.DataFrame({x: x_values.to_numpy(), y: y_values.to_numpy()})
    if color is not None:
        color_values, _ = _resolve_column(data, color, layer=layer)
        frame[color] = color_values.to_numpy()
    return plot_scatter_marginal(frame, x=x, y=y, color=color, layer=None, **kwargs)


@plot_embedding_panel.register(AnnData)
def _plot_embedding_panel_anndata(
    data: AnnData,
    features: List[str],
    dimred: str = "X_umap",
    layer: Optional[str] = None,
    **kwargs,
):
    """AnnData adapter for :func:`ggnomics.plot_embedding_panel`."""

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
        raise ValueError(f"The following features could not be resolved in adata.obs or adata.var_names: {missing}")

    return plot_embedding_panel(frame, features=features, dimred=dimred, layer=None, **kwargs)


@plot_pairs.register(AnnData)
def _plot_pairs_anndata(
    data: AnnData,
    dimred: str = "X_pca",
    n_components: int = 4,
    color_by: Optional[str] = None,
    size: Optional[float] = None,
    alpha: float = 0.6,
    palette=None,
    title: Optional[str] = None,
):
    """AnnData adapter for :func:`ggnomics.plot_pairs`."""

    emb_df = _embedding_frame(data, dimred)
    color_series = None
    if color_by is not None:
        if color_by not in data.obs.columns:
            raise KeyError(f"color_by {color_by!r} not found in adata.obs. Available: {list(data.obs.columns)[:20]}")
        color_series = data.obs[color_by].reset_index(drop=True)

    return _pairs_plot_from_embedding(emb_df, n_components, color_series, color_by, size, alpha, palette, title)


@plot_clonotype_abundance.register(AnnData)
def _plot_clonotype_abundance_anndata(data: AnnData, *args, **kwargs):
    """AnnData adapter for :func:`ggnomics.plot_clonotype_abundance`."""

    return plot_clonotype_abundance(data.obs.reset_index(drop=True), *args, **kwargs)


@plot_clonotype_overlap.register(AnnData)
def _plot_clonotype_overlap_anndata(data: AnnData, *args, **kwargs):
    """AnnData adapter for :func:`ggnomics.plot_clonotype_overlap`."""

    return plot_clonotype_overlap(data.obs.reset_index(drop=True), *args, **kwargs)


@plot_clonotype_embedding.register(AnnData)
def _plot_clonotype_embedding_anndata(
    data: AnnData,
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
    """AnnData adapter for :func:`ggnomics.plot_clonotype_embedding`."""

    if clonotype_col not in data.obs.columns:
        raise KeyError(
            f"clonotype_col {clonotype_col!r} not found in adata.obs. Available: {list(data.obs.columns)[:20]}"
        )
    emb_df = _embedding_frame(data, dimred)
    return _clonotype_embedding_plot(
        emb_df,
        components,
        data.obs[clonotype_col],
        dimred,
        expansion_thresholds,
        non_tcell_color,
        palette,
        size,
        stroke,
        alpha,
        title,
    )


@plot_pseudobulk_qc.register(AnnData)
def _plot_pseudobulk_qc_anndata(
    data: AnnData,
    sample_by: str,
    group_by: str,
    condition_by: Optional[str] = None,
    min_cells: int = 10,
    palette=None,
    ncol: int = 2,
):
    """AnnData adapter for :func:`ggnomics.plot_pseudobulk_qc`.

    Library size and pseudobulk aggregation are computed via sparse-aware
    row sums / indicator-matrix products; the complete cells x genes matrix
    is never densified.
    """

    required = [sample_by, group_by] + ([condition_by] if condition_by is not None else [])
    missing = [c for c in required if c not in data.obs.columns]
    if missing:
        raise KeyError(f"Column(s) {missing} not found in adata.obs. Available: {list(data.obs.columns)[:20]}")

    obs_df = data.obs.reset_index(drop=True)
    counts_df = _counts_panel_data(obs_df, sample_by, group_by)
    p1 = _counts_panel_plot(counts_df, sample_by, group_by, min_cells, palette)

    library_sizes = _sparse_row_sums(data.X)
    condition_values = obs_df[condition_by].to_numpy() if condition_by is not None else None
    libsize_df = _libsize_panel_data(
        obs_df[sample_by].to_numpy(), library_sizes, sample_by, condition_by, condition_values
    )
    p2 = _libsize_panel_plot(libsize_df, sample_by, condition_by, palette)

    sample_ids = obs_df[sample_by].to_numpy()
    unique_samples = pd.unique(sample_ids)
    pb_mat = _pseudobulk_aggregate(data.X, sample_ids, unique_samples)
    condition_map = (
        _sample_condition_map(obs_df, sample_by, condition_by, unique_samples) if condition_by is not None else None
    )
    pca_df = _pca_panel_data(pb_mat, unique_samples, sample_by, condition_by, condition_map)
    p3 = _pca_panel_plot(pca_df, sample_by, condition_by)

    return (p1 | p2) / p3


@plot_bimodal_scatter.register(AnnData)
def _plot_bimodal_scatter_anndata(
    data: AnnData,
    x_feature: str,
    y_feature: str,
    x_mod: str = "rna",
    y_mod: str = "prot",
    color: Optional[str] = None,
    layer_x: Optional[str] = None,
    layer_y: Optional[str] = None,
    **kwargs,
):
    """AnnData adapter for :func:`ggnomics.plot_bimodal_scatter`.

    ``layer_x``/``layer_y`` select the AnnData layer for each feature; when
    not given explicitly they fall back to ``x_mod``/``y_mod`` (backward
    compatible with the previous layer-selection convention). Explicit
    ``layer_x``/``layer_y`` always take precedence.
    """

    resolved_layer_x = layer_x if layer_x is not None else x_mod
    resolved_layer_y = layer_y if layer_y is not None else y_mod

    x_values = _expression_frame(data, [x_feature], resolved_layer_x)[x_feature]
    y_values = _expression_frame(data, [y_feature], resolved_layer_y)[y_feature]
    frame = pd.DataFrame({x_feature: x_values.to_numpy(), y_feature: y_values.to_numpy()})
    if color is not None:
        color_values, _ = _resolve_column(data, color)
        frame[color] = color_values.to_numpy()

    return plot_bimodal_scatter(frame, x_feature=x_feature, y_feature=y_feature, color=color, **kwargs)


@plot_adt_qc.register(AnnData)
def _plot_adt_qc_anndata(
    data: AnnData,
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
    """AnnData adapter for :func:`ggnomics.plot_adt_qc`."""

    del mod  # only meaningful for MuData modality selection
    if not isotype_controls:
        raise ValueError("isotype_controls must be a non-empty list.")

    candidate_features = list(features) if features is not None else list(data.var_names)
    missing_features = [f for f in candidate_features if f not in data.var_names]
    if missing_features:
        raise KeyError(
            f"features not found in adata.var_names: {missing_features}. "
            f"Available (first 20): {list(data.var_names)[:20]}"
        )
    missing_iso = [f for f in isotype_controls if f not in candidate_features]
    if missing_iso:
        raise KeyError(f"isotype_controls not found among candidate features: {missing_iso}.")
    if group_by is not None and group_by not in data.obs.columns:
        raise KeyError(f"group_by {group_by!r} not found in adata.obs. Available: {list(data.obs.columns)[:20]}")

    frame = _expression_frame(data, candidate_features, layer)
    if group_by is not None:
        frame[group_by] = data.obs[group_by].to_numpy()

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


@plot_coldata.register(AnnData)
def _plot_coldata_anndata(data: AnnData, *args, **kwargs):
    """AnnData adapter for :func:`ggnomics.plot_coldata`."""

    return plot_coldata(data.obs.reset_index(drop=True), *args, **kwargs)


@plot_rowdata.register(AnnData)
def _plot_rowdata_anndata(data: AnnData, *args, **kwargs):
    """AnnData adapter for :func:`ggnomics.plot_rowdata`."""

    return plot_rowdata(data.var.reset_index(drop=True), *args, **kwargs)


@plot_abundance.register(AnnData)
def _plot_abundance_anndata(data: AnnData, *args, **kwargs):
    """AnnData adapter for :func:`ggnomics.plot_abundance`."""

    return plot_abundance(data.obs.reset_index(drop=True), *args, **kwargs)


@plot_scatter.register(AnnData)
def _plot_scatter_anndata(
    data: AnnData,
    x: str,
    y: str,
    color: Optional[str] = None,
    **kwargs,
):
    """AnnData adapter for :func:`ggnomics.plot_scatter`."""

    layer = kwargs.pop("layer", None)
    facet_by = kwargs.get("facet_by")

    x_values, _ = _resolve_column(data, x)
    y_values, _ = _resolve_column(data, y)
    frame = pd.DataFrame({x: x_values.to_numpy(), y: y_values.to_numpy()})

    if color is not None:
        color_values, _ = _resolve_column(data, color, layer=layer)
        frame[color] = color_values.to_numpy()

    if facet_by is not None:
        if facet_by not in data.obs.columns:
            raise KeyError(f"facet_by {facet_by!r} not found in adata.obs. Available: {list(data.obs.columns)[:20]}")
        frame[facet_by] = data.obs[facet_by].to_numpy()

    return plot_scatter(frame, x=x, y=y, color=color, layer=None, **kwargs)


@plot_embedding.register(AnnData)
def _plot_embedding_anndata(
    data: AnnData,
    dimred: str = "X_umap",
    color: Optional[str] = None,
    **kwargs,
):
    """AnnData adapter for :func:`ggnomics.plot_embedding`."""

    layer = kwargs.pop("layer", None)
    facet_by = kwargs.get("facet_by")
    frame = _embedding_frame(data, dimred)

    for column in data.obs.columns:
        if column not in frame.columns:
            frame[column] = data.obs[column].to_numpy()

    if color is not None and color not in data.obs.columns:
        color_values, _ = _resolve_column(data, color, layer=layer)
        frame[color] = color_values.to_numpy()

    if facet_by is not None and facet_by not in data.obs.columns:
        raise KeyError(f"facet_by {facet_by!r} not found in adata.obs. Available: {list(data.obs.columns)[:20]}")

    return plot_embedding(frame, dimred=dimred, color=color, layer=None, **kwargs)
