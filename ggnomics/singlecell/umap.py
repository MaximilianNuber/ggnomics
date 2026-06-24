from typing import Optional
from ggnomics.plotting.utils import plot_reduced_dim
from .pca import build_plot_df_from_sce

# Placeholder for UMAP logic since the prompt asked for "do the same thing for UMAP"
# Assuming we might want a run_umap wrapper later, but for now we focus on plotting
# or assuming it's available. The prompt said: "Only in the singlecell module we do the same thing for UMAP."
# Implies run_umap and plot_umap.

def run_umap(
    sce,
    assay_name: str = "logcounts",
    n_neighbors: int = 15,
    n_components: int = 2,
    dest_key: str = "UMAP",
    **kwargs
):
    """
    Run UMAP for single-cell data.
    Requires umap-learn package.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required for run_umap")

    mat = sce.assay(assay_name)
    # Transpose to (samples x features)
    data = mat.T
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, **kwargs)
    embedding = reducer.fit_transform(data)
    
    # Store in reduced_dims
    if hasattr(sce, "reduced_dims") and isinstance(sce.reduced_dims, dict):
        sce.reduced_dims[dest_key] = embedding
        
    return embedding

def plot_umap(
    sce,
    dim_name: str = "UMAP",
    x_comp: int = 1,
    y_comp: int = 2,
    color: Optional[str] = None,
    size: Optional[float] = None,
    run_if_missing: bool = True,
    **kwargs
):
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
        
    if not has_dim and run_if_missing:
        run_umap(sce, dest_key=dim_name)

    plot_df = build_plot_df_from_sce(sce, dim_name=dim_name, prefix="umap")
    
    x = f"umap_{x_comp}"
    y = f"umap_{y_comp}"
    
    return plot_reduced_dim(plot_df, x=x, y=y, color=color, size=size, **kwargs)
