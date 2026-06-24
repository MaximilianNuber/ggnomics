import numpy as np
from sklearn.decomposition import PCA
from scipy import linalg
from .result import PcaResult

def run_pca_sklearn(data: np.ndarray, n_components: int = 50, **kwargs) -> PcaResult:
    """
    Run PCA using scikit-learn.
    """
    pca = PCA(n_components=n_components, **kwargs)
    scores = pca.fit_transform(data)
    loadings = pca.components_.T
    explained_variance = pca.explained_variance_ratio_
    
    return PcaResult(
        scores=scores,
        loadings=loadings,
        explained_variance_ratio=explained_variance,
        model=pca
    )

def run_pca_svd(data: np.ndarray, n_components: int = 50) -> PcaResult:
    """
    Run PCA using SVD (hand-rolled).
    Assumes data is already centered if centered PCA is desired.
    """
    # Simply using SVD on the data matrix X = U S V^T
    # Scores = U * S
    # Loadings = V^T (or V, depending on definition. sklearn components_ is V^T)
    # data is (n_samples, n_features)
    
    U, s, Vt = linalg.svd(data, full_matrices=False)
    
    # helper to truncate
    k = min(n_components, len(s))
    U = U[:, :k]
    s = s[:k]
    Vt = Vt[:k, :]
    
    scores = U * s
    loadings = Vt.T 
    
    # Explain variance
    # total variance = sum(s^2) / (n - 1) usually (if centered)
    # here just returning proportion of s^2
    eigvals = s**2
    total_var = np.sum(eigvals)
    explained_variance_ratio = eigvals / total_var
    
    return PcaResult(
        scores=scores,
        loadings=loadings,
        explained_variance_ratio=explained_variance_ratio,
        model=None
    )
