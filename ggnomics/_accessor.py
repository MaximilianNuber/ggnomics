"""DataAccessor — uniform interface over DataFrame, AnnData, and SingleCellExperiment."""

from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import pandas as pd


class DataAccessor:
    """Thin wrapper that normalises access to cell metadata and embeddings.

    Supported input types
    ---------------------
    - ``pd.DataFrame``       — columns are obs metadata *and* embedding dimensions
    - ``anndata.AnnData``    — obsm for embeddings, obs for metadata, X/layers for expression
    - ``SingleCellExperiment`` — reduced_dims for embeddings, colData for metadata

    Args:
        data: One of the supported objects described above.

    Raises:
        TypeError: If ``data`` is not one of the supported types.
    """

    def __init__(self, data) -> None:
        self._data = data
        self._type = self._detect_type(data)

    # ------------------------------------------------------------------
    # Type detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_type(data) -> str:
        type_name = type(data).__name__
        module = type(data).__module__ or ""

        if isinstance(data, pd.DataFrame):
            return "dataframe"

        # AnnData check (duck-typing so we don't force the import)
        if type_name == "AnnData" or "anndata" in module:
            return "anndata"

        # SingleCellExperiment / SummarizedExperiment (BiocPy)
        if type_name in ("SingleCellExperiment",) or "singlecellexperiment" in module:
            return "sce"

        if type_name in ("SummarizedExperiment",) or "summarizedexperiment" in module:
            return "se"

        raise TypeError(
            f"Unsupported data type '{type(data).__name__}'. "
            "Pass a pd.DataFrame, anndata.AnnData, or SingleCellExperiment."
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def object_type(self) -> str:
        """Return a short string identifying the wrapped object type.

        Returns:
            One of ``"dataframe"``, ``"anndata"``, ``"sce"``, ``"se"``.
        """
        return self._type

    def n_obs(self) -> int:
        """Number of observations (cells / samples).

        Returns:
            Integer count.
        """
        if self._type == "dataframe":
            return len(self._data)
        if self._type == "anndata":
            return self._data.n_obs
        # SCE / SE
        return self._data.shape[1]  # BiocPy: (features, cells)

    def obs_names(self) -> List[str]:
        """Cell / sample names.

        Returns:
            List of strings.
        """
        if self._type == "dataframe":
            return list(self._data.index.astype(str))
        if self._type == "anndata":
            return list(self._data.obs_names)
        # SCE / SE — BiocPy uses col_names / column_names
        for attr in ("col_names", "column_names"):
            val = getattr(self._data, attr, None)
            if val is not None:
                return list(val)
        return [str(i) for i in range(self.n_obs())]

    def var_names(self) -> List[str]:
        """Feature / gene names.

        Returns:
            List of strings.
        """
        if self._type == "dataframe":
            return list(self._data.columns)
        if self._type == "anndata":
            return list(self._data.var_names)
        # SCE / SE — BiocPy uses row_names
        for attr in ("row_names", "feature_names", "gene_names"):
            val = getattr(self._data, attr, None)
            if val is not None:
                return list(val)
        n_feats = self._data.shape[0]
        return [str(i) for i in range(n_feats)]

    def obs(self) -> pd.DataFrame:
        """Cell-level metadata as a DataFrame.

        Returns:
            DataFrame indexed by cell names.
        """
        if self._type == "dataframe":
            return self._data.copy()
        if self._type == "anndata":
            return self._data.obs.copy()
        # SCE / SE — BiocPy: col_data property (aliases column_data)
        for attr in ("col_data", "column_data"):
            cd = getattr(self._data, attr, None)
            if cd is not None:
                if hasattr(cd, "to_pandas"):
                    df = cd.to_pandas().copy()
                    # BiocFrame.to_pandas() adds a 'rownames' column — drop it
                    if "rownames" in df.columns:
                        df = df.drop(columns=["rownames"])
                    return df
                return pd.DataFrame(cd).copy()
        return pd.DataFrame(index=range(self.n_obs()))

    def get_embedding(self, key: str) -> pd.DataFrame:
        """Extract a low-dimensional embedding as a DataFrame.

        The key lookup is case-insensitive and tries common prefixes
        (``"X_"`` for AnnData, exact match, upper-case).

        Args:
            key: Embedding name, e.g. ``"X_pca"``, ``"pca"``, ``"UMAP"``.

        Returns:
            DataFrame with columns ``["{key}_1", "{key}_2", ...]`` and cells
            as the index.

        Raises:
            KeyError: If the embedding cannot be found.
            NotImplementedError: If called on a SummarizedExperiment (no
                embeddings supported).
        """
        if self._type == "se":
            raise NotImplementedError(
                "SummarizedExperiment objects do not store cell embeddings. "
                "Use a SingleCellExperiment or convert first."
            )

        if self._type == "dataframe":
            return self._embedding_from_df(key)

        if self._type == "anndata":
            return self._embedding_from_anndata(key)

        # SCE
        return self._embedding_from_sce(key)

    def resolve_column(
        self,
        key: str,
        layer: Optional[str] = None,
    ) -> "tuple[pd.Series, bool]":
        """Resolve a string key to a pd.Series and whether it is continuous.

        Tries obs columns first, then var_names (returns expression values).

        Args:
            key: Column name in obs/colData, or feature/gene name.
            layer: Expression layer/assay for gene expression lookup.

        Returns:
            Tuple ``(series, is_continuous)`` where ``series`` has integer
            index (0-based) and ``is_continuous`` is ``True`` for numeric
            or expression columns.

        Raises:
            KeyError: If the key cannot be found in obs columns or var_names.
        """
        obs_df = self.obs()
        if key in obs_df.columns:
            s = obs_df[key].reset_index(drop=True)
            is_continuous = bool(pd.api.types.is_numeric_dtype(s))
            return s, is_continuous

        if key in self.var_names():
            expr_df = self.get_expression([key], layer=layer)
            return expr_df[key].reset_index(drop=True), True

        raise KeyError(
            f"'{key}' not found in obs columns or feature names. "
            f"Obs columns (first 20): {list(obs_df.columns)[:20]}. "
            f"Feature names (first 20): {self.var_names()[:20]}"
        )

    def get_expression(
        self,
        features: List[str],
        layer: Optional[str] = None,
    ) -> pd.DataFrame:
        """Retrieve expression values for a list of features.

        Args:
            features: List of feature/gene names.
            layer: Layer name (AnnData ``layers`` key or SCE assay name).
                   ``None`` uses the default (AnnData ``X``, SCE ``"counts"``).

        Returns:
            DataFrame with shape ``(n_cells, len(features))``.

        Raises:
            KeyError: If any feature is not found.
        """
        if self._type == "dataframe":
            missing = [f for f in features if f not in self._data.columns]
            if missing:
                raise KeyError(f"Features not found in DataFrame columns: {missing}")
            return self._data[features].copy().reset_index(drop=True)

        if self._type == "anndata":
            return self._expr_from_anndata(features, layer)

        # SCE
        return self._expr_from_sce(features, layer)

    # ------------------------------------------------------------------
    # Private helpers — DataFrame
    # ------------------------------------------------------------------

    def _embedding_from_df(self, key: str) -> pd.DataFrame:
        cols = list(self._data.columns)
        # Strip leading X_ and lowercase for matching
        key_stripped = _strip_x_prefix(key).lower()

        def _matches(col: str) -> bool:
            lc = col.lower()
            # e.g. "umap_1", "umap_2"  or  "umap1", "umap2"  or  "pca_1"
            return (
                lc.startswith(key_stripped + "_")
                or lc.startswith("x_" + key_stripped + "_")
                # digit immediately after key: UMAP1, PCA2, …
                or (lc.startswith(key_stripped) and len(lc) > len(key_stripped) and lc[len(key_stripped)].isdigit())
            )

        matched = sorted([c for c in cols if _matches(c)])

        if not matched:
            # Single exact-match column
            matched = [c for c in cols if c.lower() == key_stripped or c.lower() == key.lower()]

        if not matched:
            raise KeyError(
                f"Embedding '{key}' not found in DataFrame columns. "
                f"Available columns: {cols[:20]}"
            )

        emb = self._data[matched].copy().reset_index(drop=True)
        base = key_stripped
        emb.columns = [f"{base}_{i+1}" for i in range(len(matched))]
        return emb

    # ------------------------------------------------------------------
    # Private helpers — AnnData
    # ------------------------------------------------------------------

    def _embedding_from_anndata(self, key: str) -> pd.DataFrame:
        obsm = self._data.obsm
        candidates = _embedding_key_candidates(key)

        # Exact match first
        found = None
        for c in candidates:
            if c in obsm:
                found = c
                break

        # Case-insensitive fallback
        if found is None:
            obsm_lower = {k.lower(): k for k in obsm.keys()}
            for c in candidates:
                if c.lower() in obsm_lower:
                    found = obsm_lower[c.lower()]
                    break

        if found is None:
            raise KeyError(
                f"Embedding '{key}' not found in adata.obsm. "
                f"Available keys: {list(obsm.keys())}"
            )
        arr = np.asarray(obsm[found])
        # Use stripped key name for columns
        base = _strip_x_prefix(key).lower()
        cols = [f"{base}_{i+1}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)

    def _expr_from_anndata(
        self, features: List[str], layer: Optional[str]
    ) -> pd.DataFrame:
        var_names = list(self._data.var_names)
        indices = _resolve_feature_indices(features, var_names)

        if layer is None:
            mat = self._data.X
        else:
            if layer not in self._data.layers:
                raise KeyError(
                    f"Layer '{layer}' not found. Available: {list(self._data.layers.keys())}"
                )
            mat = self._data.layers[layer]

        arr = _dense_slice(mat, indices)
        return pd.DataFrame(arr, columns=features)

    # ------------------------------------------------------------------
    # Private helpers — SCE
    # ------------------------------------------------------------------

    def _embedding_from_sce(self, key: str) -> pd.DataFrame:
        candidates = _embedding_key_candidates(key)

        # BiocPy API: get_reduced_dimension(name) / reduced_dim(name)
        for getter in ("get_reduced_dimension", "reduced_dim"):
            if hasattr(self._data, getter):
                fn = getattr(self._data, getter)
                for c in candidates:
                    try:
                        arr = fn(c)
                        if arr is not None:
                            arr = np.asarray(arr)
                            if arr.ndim == 2:
                                base = _strip_x_prefix(key).lower()
                                cols = [f"{base}_{i+1}" for i in range(arr.shape[1])]
                                return pd.DataFrame(arr, columns=cols)
                    except Exception:
                        pass

        # Dict-like fallback: reduced_dims / reduced_dimensions
        for rd_attr in ("reduced_dims", "reduced_dimensions", "_reduced_dims"):
            rd = getattr(self._data, rd_attr, None)
            if rd is None:
                continue
            for c in candidates:
                try:
                    val = rd[c]
                    if val is not None:
                        arr = np.asarray(val)
                        if arr.ndim == 2:
                            base = _strip_x_prefix(key).lower()
                            cols = [f"{base}_{i+1}" for i in range(arr.shape[1])]
                            return pd.DataFrame(arr, columns=cols)
                except (KeyError, TypeError):
                    pass

        # Collect available names for the error message
        available: list = []
        for name_fn in ("get_reduced_dimension_names", "reduced_dim_names"):
            if hasattr(self._data, name_fn):
                try:
                    available = list(getattr(self._data, name_fn)())
                    break
                except Exception:
                    pass

        raise KeyError(
            f"Embedding '{key}' not found in SCE reduced dims. "
            f"Available: {available}"
        )

    def _expr_from_sce(
        self, features: List[str], layer: Optional[str]
    ) -> pd.DataFrame:
        assay_name = layer if layer is not None else "counts"
        # Try assay() method first (BiocPy SCE)
        if hasattr(self._data, "assay"):
            try:
                mat = self._data.assay(assay_name)
            except Exception as exc:
                raise KeyError(
                    f"Assay '{assay_name}' not found in SCE. "
                    f"Original error: {exc}"
                ) from exc
        elif hasattr(self._data, "assays"):
            try:
                mat = self._data.assays[assay_name]
            except Exception as exc:
                raise KeyError(f"Assay '{assay_name}' not found.") from exc
        else:
            raise AttributeError("SCE object has no .assay() method or .assays attribute.")

        var_names = self.var_names()
        indices = _resolve_feature_indices(features, var_names)

        # SCE stores (features, cells) — transpose to (cells, features)
        arr = _dense_slice(mat.T, indices)
        return pd.DataFrame(arr, columns=features)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _strip_x_prefix(key: str) -> str:
    """Remove leading 'X_' prefix (AnnData convention)."""
    if key.lower().startswith("x_"):
        return key[2:]
    return key


def _embedding_key_candidates(key: str) -> List[str]:
    """Generate a priority list of key variants to try."""
    stripped = _strip_x_prefix(key)
    candidates = []
    for variant in (key, "X_" + stripped, stripped, stripped.upper(), key.upper(), key.lower()):
        if variant not in candidates:
            candidates.append(variant)
    return candidates


def _resolve_feature_indices(features: List[str], var_names: List[str]) -> List[int]:
    """Map feature names to integer column indices."""
    name_to_idx = {n: i for i, n in enumerate(var_names)}
    indices = []
    missing = []
    for f in features:
        if f in name_to_idx:
            indices.append(name_to_idx[f])
        else:
            missing.append(f)
    if missing:
        raise KeyError(f"Features not found: {missing}. Available (first 20): {var_names[:20]}")
    return indices


def _dense_slice(mat, col_indices: List[int]) -> np.ndarray:
    """Slice columns from a dense or sparse matrix and return a dense array."""
    if hasattr(mat, "toarray"):
        arr = mat.toarray()
    else:
        arr = np.asarray(mat)
    return arr[:, col_indices]


def _common_prefix(strings: List[str]) -> str:
    """Return the longest common prefix of a list of strings."""
    if not strings:
        return ""
    prefix = strings[0]
    for s in strings[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
        if not prefix:
            break
    return prefix
