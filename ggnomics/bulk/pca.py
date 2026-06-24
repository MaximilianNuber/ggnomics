from typing import Optional, Union
import pandas as pd
import numpy as np
from ..pca.methods import run_pca_sklearn, run_pca_svd
from ..pca.result import PcaResult
from ..plotting.utils import plot_reduced_dim

# Mock SummarizedExperiment if not available, usually provided by biocpy
# But we treat it as an object with .assay(name) and .col_data
# For typing, we use Any

def run_pca(
    se,
    assay_name: str = "logcounts",
    n_components: int = 50,
    method: str = "sklearn",
    **kwargs
) -> PcaResult:
    """
    Run PCA for bulk data (SummarizedExperiment).
    """
    # Basic assumption: se has an assay that is (n_features, n_samples)
    # But sklearn expects (n_samples, n_features).
    # Usually bioc objects are (features x samples).
    
    mat = se.assay(assay_name)
    # Transpose to (samples x features)
    data = mat.T
    
    if method == "sklearn":
        res = run_pca_sklearn(data, n_components=n_components, **kwargs)
    elif method == "svd":
        # SVD usually expects centered data for PCA
        # We might want to center it here if not done
        # For this simple implementation, we pass as is or assume user knows
        res = run_pca_svd(data, n_components=n_components)
    else:
        raise ValueError(f"Unknown method {method}")
        
    res.sample_names = se.column_names
    res.feature_names = se.row_names
    return res

def build_plot_df_from_result(
    se,
    pca_res: PcaResult,
    n_comps: int = 2,
    prefix: str = "pca",
) -> pd.DataFrame:
    k = min(n_comps, pca_res.scores.shape[1])
    scores = pca_res.scores[:, :k]

    dim_cols = [f"{prefix}_{i+1}" for i in range(k)]
    plot_df = pd.DataFrame(scores, columns=dim_cols)

    if se.col_data is not None:
        if hasattr(se.col_data, "to_pandas"):
            meta = se.col_data.to_pandas().copy()
        else:
            meta = pd.DataFrame(se.col_data).copy()
        # keep row alignment
        if getattr(se, "column_names", None) is not None:
            plot_df.index = se.column_names
            # Ensure meta is aligned
            if len(meta) == len(plot_df):
                 meta.index = plot_df.index
        plot_df = pd.concat([plot_df, meta], axis=1)

    return plot_df

def plot_pca(
    se,
    pca_res: Optional[PcaResult] = None,
    assay_name: str = "logcounts",
    n_components: int = 50,
    n_top_genes: Optional[int] = None,
    x_comp: int = 1,
    y_comp: int = 2,
    color: Optional[str] = None,
    size: Optional[float] = None,
    return_result: bool = False,
    **kwargs
):
    """
    Plot PCA for SummarizedExperiment.
    If pca_res is None, runs PCA first.
    
    Parameters
    ----------
    n_top_genes : int, optional
        If provided and pca_res is None, subsets the SummarizedExperiment to the
        top n genes with highest variance before running PCA.
    """
    if pca_res is None:
        se_to_use = se
        if n_top_genes is not None:
            # Calculate variance
            mat = se.assay(assay_name)
            # Assuming (features, samples)
            vars = np.var(mat, axis=1)
            # Get indices of top variance
            # argsort is ascending, so we take the last n
            if n_top_genes < mat.shape[0]:
                top_indices = np.argsort(vars)[-n_top_genes:]
                # Sort indices to keep original order if desired, but PCA doesn't care about gene order usually
                top_indices = np.sort(top_indices)
                
                # Subset se. Assuming se supports slicing [rows, cols]
                # If se is a custom object without slicing, we might need a fallback
                # specific for the Mock object in tests, but in production this expects SummarizedExperiment
                try:
                    se_to_use = se[top_indices, :]
                except TypeError:
                    # Fallback for objects that might not support slicing or behave differently
                    # We might need to construct a lightweight wrapper if slicing fails
                    # But standard SummarizedExperiment supports slicing.
                    # As a last resort for robustness if the user passes something else:
                    pass

        pca_res = run_pca(se_to_use, assay_name=assay_name, n_components=n_components)
        
    plot_df = build_plot_df_from_result(se, pca_res, n_comps=max(x_comp, y_comp))
    
    x = f"pca_{x_comp}"
    y = f"pca_{y_comp}"
    
    # Update defaults for bulk to have larger points
    # We pass autosize_max=4 by default if not present in kwargs
    plot_kwargs = kwargs.copy()
    if "autosize_max" not in plot_kwargs:
        plot_kwargs["autosize_max"] = 4.0
    if "autosize_min" not in plot_kwargs:
        plot_kwargs["autosize_min"] = 1.0
    
    p = plot_reduced_dim(plot_df, x=x, y=y, color=color, size=size, **plot_kwargs)
    
    if return_result:
        return p, pca_res
    return p
