from typing import Optional, Union, Dict
import pandas as pd
import numpy as np
from ..pca.methods import run_pca_sklearn, run_pca_svd
from ..pca.result import PcaResult
from ..plotting.utils import plot_reduced_dim

def run_pca(
    sce,
    assay_name: str = "logcounts",
    n_components: int = 50,
    method: str = "sklearn",
    dest_key: str = "PCA",
    **kwargs
) -> PcaResult:
    """
    Run PCA for single-cell data (SingleCellExperiment).
    Stores result in reduced_dims.
    """
    mat = sce.assay(assay_name)
    # Transpose to (samples x features)
    data = mat.T
    
    if method == "sklearn":
        res = run_pca_sklearn(data, n_components=n_components, **kwargs)
    elif method == "svd":
        res = run_pca_svd(data, n_components=n_components)
    else:
        raise ValueError(f"Unknown method {method}")

    res.sample_names = sce.column_names
    res.feature_names = sce.row_names
    
    # Store in reduced_dims
    if not hasattr(sce, "reduced_dims"):
        # If naive object, skip
        pass
    else:
        # Assuming dict-like access for reduced_dims
        sce.reduced_dims[dest_key] = res.scores
        
    return res

def build_plot_df_from_sce(
    sce,
    dim_name: str,
    prefix: Optional[str] = None,
) -> pd.DataFrame:
    if hasattr(sce, "reduced_dim"):
        emb = sce.reduced_dim(dim_name)
    elif hasattr(sce, "reduced_dims") and isinstance(sce.reduced_dims, dict):
        emb = sce.reduced_dims[dim_name]
    else:
        raise AttributeError("SingleCellExperiment object has no reduced_dim or reduced_dims")

    emb = np.asarray(emb)
    n_comps = emb.shape[1]

    if prefix is None:
        prefix = dim_name.lower()

    dim_cols = [f"{prefix}_{i+1}" for i in range(n_comps)]
    plot_df = pd.DataFrame(emb, columns=dim_cols)

    if hasattr(sce.col_data, "to_pandas"):
        meta = sce.col_data.to_pandas().copy()
    else:
        meta = pd.DataFrame(sce.col_data).copy()
    if getattr(sce, "column_names", None) is not None:
        plot_df.index = sce.column_names
        meta = meta.reindex(plot_df.index)

    plot_df = pd.concat([plot_df, meta], axis=1)
    return plot_df

def plot_pca(
    sce,
    dim_name: str = "PCA",
    x_comp: int = 1,
    y_comp: int = 2,
    color: Optional[str] = None,
    size: Optional[float] = None,
    run_if_missing: bool = True,
    return_result: bool = False,
    **kwargs
):
    """
    Plot PCA for SingleCellExperiment.
    """
    # Check if dim exists
    has_dim = False
    if hasattr(sce, "reduced_dim"):
        try:
            sce.reduced_dim(dim_name)
            has_dim = True
        except:
            pass
    elif hasattr(sce, "reduced_dims") and dim_name in sce.reduced_dims:
        has_dim = True
        
    pca_res = None
    if not has_dim and run_if_missing:
        pca_res = run_pca(sce, dest_key=dim_name)
        
    plot_df = build_plot_df_from_sce(sce, dim_name=dim_name, prefix="pca")
    
    x = f"pca_{x_comp}"
    y = f"pca_{y_comp}"
    
    p = plot_reduced_dim(plot_df, x=x, y=y, color=color, size=size, **kwargs)
    
    if return_result:
        return p, pca_res
    return p
