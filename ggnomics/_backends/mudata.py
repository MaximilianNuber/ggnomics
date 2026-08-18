"""MuData registrations for cross-modality plots.

MuData is used only for genuinely cross-modality functions
(``plot_bimodal_scatter``, ``plot_adt_qc``) — not for the single-container
generics (scatter, embedding, expression, ...), which operate on one
modality's AnnData directly.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from mudata import MuData

from ..multimodal import plot_adt_qc, plot_bimodal_scatter
from .anndata import _expression_frame as _anndata_expression_frame


def _modality(mdata: MuData, mod: str):
    if mod not in mdata.mod:
        raise KeyError(f"Modality {mod!r} not found in MuData. Available: {list(mdata.mod.keys())}")
    return mdata.mod[mod]


def _feature_series(adata, feature: str) -> pd.Series:
    if feature not in adata.var_names:
        raise KeyError(
            f"Feature {feature!r} not found in modality var_names. Available (first 20): {list(adata.var_names)[:20]}"
        )
    return _anndata_expression_frame(adata, [feature], None)[feature].set_axis(adata.obs_names)


def _resolve_mudata_color(
    mdata: MuData,
    color: str,
    shared: List[str],
    mods: tuple[str, str],
) -> np.ndarray:
    if color in mdata.obs.columns:
        return mdata.obs.loc[shared, color].to_numpy()

    found = [mod for mod in mods if color in mdata.mod[mod].obs.columns]
    if not found:
        raise KeyError(f"color {color!r} not found in mdata.obs or in the obs of modalities {list(mods)}.")
    if len(found) > 1:
        raise ValueError(
            f"color {color!r} is ambiguous: present in multiple modalities {found}. "
            "Add it to mdata.obs (a shared column) to disambiguate."
        )
    adata = mdata.mod[found[0]]
    return adata.obs.loc[shared, color].to_numpy()


def _aligned_bimodal_frame(
    mdata: MuData,
    x_feature: str,
    y_feature: str,
    x_mod: str,
    y_mod: str,
    color: Optional[str],
) -> tuple[pd.DataFrame, str, str]:
    """Build the aligned (x, y[, color]) frame for a cross-modality scatter.

    Returns the frame together with the (possibly disambiguated) column keys
    actually used for x and y. ``x_feature``/``y_feature`` frequently share
    the same literal name across modalities in CITE-seq-style data (e.g. gene
    symbol ``"CD38"`` and antibody name ``"CD38"``) — building
    ``pd.DataFrame({x_feature: ..., y_feature: ...})`` directly would then
    collapse to a single column (the second dict entry silently overwrites
    the first), so both aesthetics would read the same values and the plot
    would show a spurious perfect diagonal. Column keys are qualified with
    the modality name whenever ``x_feature == y_feature`` to keep them
    distinct; the original feature name is preserved for axis labels by the
    caller.
    """
    x_adata = _modality(mdata, x_mod)
    y_adata = _modality(mdata, y_mod)

    x_series = _feature_series(x_adata, x_feature)
    y_series = _feature_series(y_adata, y_feature)

    y_names = set(y_adata.obs_names)
    shared = [name for name in x_adata.obs_names if name in y_names]

    dropped_x = x_adata.n_obs - len(shared)
    dropped_y = y_adata.n_obs - len(shared)
    if dropped_x or dropped_y:
        warnings.warn(
            f"plot_bimodal_scatter: dropped {dropped_x} observation(s) from "
            f"modality {x_mod!r} and {dropped_y} from {y_mod!r} that are not "
            "shared between the two modalities.",
            UserWarning,
            stacklevel=3,
        )
    if not shared:
        raise ValueError(f"Modalities {x_mod!r} and {y_mod!r} share no observations.")

    x_key, y_key = x_feature, y_feature
    if x_key == y_key:
        x_key, y_key = f"{x_mod}:{x_feature}", f"{y_mod}:{y_feature}"

    frame = pd.DataFrame(
        {
            x_key: x_series.loc[shared].to_numpy(),
            y_key: y_series.loc[shared].to_numpy(),
        }
    )
    if color is not None:
        frame[color] = _resolve_mudata_color(mdata, color, shared, (x_mod, y_mod))
    return frame, x_key, y_key


@plot_bimodal_scatter.register(MuData)
def _plot_bimodal_scatter_mudata(
    data: MuData,
    x_feature: str,
    y_feature: str,
    x_mod: str = "rna",
    y_mod: str = "prot",
    color: Optional[str] = None,
    layer_x: Optional[str] = None,
    layer_y: Optional[str] = None,
    **kwargs,
):
    """MuData adapter for :func:`ggnomics.plot_bimodal_scatter`.

    ``x_feature``/``y_feature`` are resolved from ``mdata.mod[x_mod]`` and
    ``mdata.mod[y_mod]`` respectively, aligned by observation name (the
    intersection, in ``x_mod``'s observation order). ``color`` is resolved
    from shared ``mdata.obs`` first, then from modality obs only when
    unambiguous.
    """

    del layer_x, layer_y  # MuData resolves features from each modality's .X
    frame, x_key, y_key = _aligned_bimodal_frame(data, x_feature, y_feature, x_mod, y_mod, color)
    # Preserve the original feature names as axis labels even when the
    # underlying columns were disambiguated (x_feature == y_feature case).
    kwargs.setdefault("x_label", x_feature)
    kwargs.setdefault("y_label", y_feature)
    return plot_bimodal_scatter(frame, x_feature=x_key, y_feature=y_key, color=color, **kwargs)


@plot_adt_qc.register(MuData)
def _plot_adt_qc_mudata(
    data: MuData,
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
):
    """MuData adapter for :func:`ggnomics.plot_adt_qc`.

    Resolves the ADT panel from ``mdata.mod[mod]`` (default ``"prot"``).
    """

    adata = _modality(data, mod)
    if not isotype_controls:
        raise ValueError("isotype_controls must be a non-empty list.")

    candidate_features = list(features) if features is not None else list(adata.var_names)
    missing_features = [f for f in candidate_features if f not in adata.var_names]
    if missing_features:
        raise KeyError(
            f"features not found in mdata.mod[{mod!r}].var_names: {missing_features}. "
            f"Available (first 20): {list(adata.var_names)[:20]}"
        )
    missing_iso = [f for f in isotype_controls if f not in candidate_features]
    if missing_iso:
        raise KeyError(f"isotype_controls not found among candidate features: {missing_iso}.")

    frame = _anndata_expression_frame(adata, candidate_features, layer)
    if group_by is not None:
        if group_by in data.obs.columns:
            frame[group_by] = data.obs.loc[adata.obs_names, group_by].to_numpy()
        elif group_by in adata.obs.columns:
            frame[group_by] = adata.obs[group_by].to_numpy()
        else:
            raise KeyError(
                f"group_by {group_by!r} not found in mdata.obs or in "
                f"mdata.mod[{mod!r}].obs. Available (mdata.obs, first 20): "
                f"{list(data.obs.columns)[:20]}"
            )

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


__all__: list[str] = []
