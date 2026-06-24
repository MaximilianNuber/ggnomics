"""Reproducible mock datasets for ggnomics examples and tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# DataFrame mock
# ---------------------------------------------------------------------------


def make_mock_df(
    n_cells: int = 500,
    n_genes: int = 200,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a mock single-cell DataFrame with embeddings and expression.

    The DataFrame contains:

    - ``UMAP1``, ``UMAP2``: 2-D UMAP coordinates.
    - ``PCA1`` … ``PCA5``: first 5 principal components.
    - ``cluster``: 5 Leiden-like cluster labels (``"C0"``–``"C4"``).
    - ``batch``: 2 batch labels.
    - ``sample``: 4 sample labels.
    - ``n_counts``, ``n_genes_detected``, ``mito_frac``: QC metrics.
    - ``Gene0001`` … ``Gene{n_genes:04d}``: raw count expression columns.

    Args:
        n_cells: Number of cells (rows).
        n_genes: Number of genes (columns).
        seed: Random seed for reproducibility.

    Returns:
        A ``pd.DataFrame`` with shape ``(n_cells, 5 + 5 + n_genes + …)``.
    """
    rng = np.random.default_rng(seed)

    n_clusters = 5
    cluster_ids = rng.integers(0, n_clusters, size=n_cells)
    clusters = np.array([f"C{i}" for i in cluster_ids])

    # UMAP: clusters arranged roughly on a circle
    angles = (cluster_ids / n_clusters) * 2 * np.pi
    umap1 = np.cos(angles) * 3 + rng.normal(0, 0.7, n_cells)
    umap2 = np.sin(angles) * 3 + rng.normal(0, 0.7, n_cells)

    # PCA: random + cluster signal
    pca_base = rng.normal(0, 1, (n_cells, 5))
    cluster_signal = np.eye(5)[cluster_ids % 5] * 2
    pca = pca_base + cluster_signal

    # Gene names
    gene_names = [f"Gene{i+1:04d}" for i in range(n_genes)]

    # Expression: NB-like counts per cluster with marker genes
    expr = rng.negative_binomial(2, 0.5, size=(n_cells, n_genes)).astype(float)
    # Add cluster markers: first 10 genes per cluster are 3× enriched
    for c in range(n_clusters):
        mask = cluster_ids == c
        g_start = c * 10
        g_end = g_start + 10
        expr[np.ix_(mask, list(range(g_start, g_end)))] *= 3

    # QC
    n_counts = expr.sum(axis=1)
    n_genes_det = (expr > 0).sum(axis=1)
    mito_frac = rng.beta(1, 10, n_cells)

    batch = np.where(rng.random(n_cells) < 0.5, "batch1", "batch2")
    sample = np.choose(rng.integers(0, 4, n_cells), ["S1", "S2", "S3", "S4"])

    data: dict = {
        "UMAP1": umap1,
        "UMAP2": umap2,
        "PCA1": pca[:, 0],
        "PCA2": pca[:, 1],
        "PCA3": pca[:, 2],
        "PCA4": pca[:, 3],
        "PCA5": pca[:, 4],
        "cluster": clusters,
        "batch": batch,
        "sample": sample,
        "n_counts": n_counts,
        "n_genes_detected": n_genes_det,
        "mito_frac": mito_frac,
    }
    for g, col in zip(gene_names, expr.T):
        data[g] = col

    return pd.DataFrame(data, index=[f"cell{i+1:05d}" for i in range(n_cells)])


# ---------------------------------------------------------------------------
# AnnData mock
# ---------------------------------------------------------------------------


def make_mock_anndata(
    n_cells: int = 500,
    n_genes: int = 200,
    seed: int = 42,
):
    """Build a mock ``anndata.AnnData`` object.

    ``obsm`` contains ``"X_pca"`` (50 dims) and ``"X_umap"`` (2 dims).
    ``obs`` contains cluster, batch, sample, and QC columns.
    ``X`` contains raw integer counts.
    ``layers["logcounts"]`` contains ``log1p``-normalised counts.
    ``var`` contains ``mean_expr`` and ``highly_variable`` columns.

    Args:
        n_cells: Number of cells.
        n_genes: Number of genes.
        seed: Random seed.

    Returns:
        An ``anndata.AnnData`` object.
    """
    import anndata

    df = make_mock_df(n_cells=n_cells, n_genes=n_genes, seed=seed)
    gene_names = [f"Gene{i+1:04d}" for i in range(n_genes)]

    X = df[gene_names].values.copy()
    obs = df.drop(columns=gene_names + ["UMAP1", "UMAP2", "PCA1", "PCA2", "PCA3", "PCA4", "PCA5"])

    rng = np.random.default_rng(seed)
    pca_coords = rng.normal(0, 1, (n_cells, 50))
    umap_coords = np.column_stack([df["UMAP1"].values, df["UMAP2"].values])

    adata = anndata.AnnData(X=X.astype(np.float32))
    adata.obs_names = df.index.tolist()
    adata.var_names = gene_names
    adata.obs = obs.copy()
    adata.obsm["X_pca"] = pca_coords.astype(np.float32)
    adata.obsm["X_umap"] = umap_coords.astype(np.float32)

    # Normalised layer
    lib_sizes = X.sum(axis=1, keepdims=True).astype(float)
    lib_sizes[lib_sizes == 0] = 1.0
    logcounts = np.log1p(X / lib_sizes * 1e4).astype(np.float32)
    adata.layers["logcounts"] = logcounts

    # var metadata
    adata.var["mean_expr"] = X.mean(axis=0)
    adata.var["highly_variable"] = adata.var["mean_expr"] > adata.var["mean_expr"].median()

    # Extra fields for repertoire / pseudobulk tests
    rng2 = np.random.default_rng(seed + 1)
    clonotype_ids = np.array([f"clone{i:03d}" for i in rng2.integers(0, 80, n_cells)], dtype=object)
    has_tcr = rng2.random(n_cells) < 0.4
    clonotype_ids[~has_tcr] = np.nan
    adata.obs["clonotype_id"] = clonotype_ids
    adata.obs["pseudotime"] = rng2.uniform(0, 1, n_cells).astype(np.float32)

    return adata


# ---------------------------------------------------------------------------
# SingleCellExperiment mock
# ---------------------------------------------------------------------------


def make_mock_sce(
    n_cells: int = 500,
    n_genes: int = 200,
    seed: int = 42,
):
    """Build a mock ``SingleCellExperiment``.

    ``reducedDims`` contains ``"PCA"`` (50 dims) and ``"UMAP"`` (2 dims).
    ``colData`` contains cluster, batch, sample, and QC columns.
    Assay ``"counts"`` contains raw integer counts; ``"logcounts"`` contains
    log1p-normalised counts.

    Args:
        n_cells: Number of cells.
        n_genes: Number of genes.
        seed: Random seed.

    Returns:
        A ``singlecellexperiment.SingleCellExperiment`` object.

    Raises:
        ImportError: If ``singlecellexperiment`` is not installed.
    """
    try:
        from singlecellexperiment import SingleCellExperiment
        import biocframe
    except ImportError as exc:
        raise ImportError(
            "singlecellexperiment and biocframe are required. "
            "Install with: pip install singlecellexperiment biocframe"
        ) from exc

    df = make_mock_df(n_cells=n_cells, n_genes=n_genes, seed=seed)
    gene_names = [f"Gene{i+1:04d}" for i in range(n_genes)]

    counts = df[gene_names].values.T.astype(np.float32)  # (genes, cells)
    lib_sizes = counts.sum(axis=0, keepdims=True).astype(float)
    lib_sizes[lib_sizes == 0] = 1.0
    logcounts = np.log1p(counts / lib_sizes * 1e4).astype(np.float32)

    obs_meta = df.drop(columns=gene_names + ["UMAP1", "UMAP2", "PCA1", "PCA2", "PCA3", "PCA4", "PCA5"])
    col_data = biocframe.BiocFrame(obs_meta.to_dict(orient="list"))

    rng = np.random.default_rng(seed)
    pca_coords = rng.normal(0, 1, (n_cells, 50)).astype(np.float32)
    umap_coords = np.column_stack([df["UMAP1"].values, df["UMAP2"].values]).astype(np.float32)

    sce = SingleCellExperiment(
        assays={"counts": counts, "logcounts": logcounts},
        row_names=gene_names,
        column_names=list(df.index),
        column_data=col_data,
        reduced_dimensions={"PCA": pca_coords, "UMAP": umap_coords},
    )
    return sce


# ---------------------------------------------------------------------------
# Convenience size variants
# ---------------------------------------------------------------------------


def make_small_df() -> pd.DataFrame:
    """50-cell DataFrame for fast unit tests."""
    return make_mock_df(n_cells=50, n_genes=50, seed=0)


def make_large_df() -> pd.DataFrame:
    """5000-cell DataFrame for adaptive-size smoke tests."""
    return make_mock_df(n_cells=5_000, n_genes=200, seed=1)


# ---------------------------------------------------------------------------
# New mock data for DE, coefficient, multimodal, and repertoire plots
# ---------------------------------------------------------------------------


def make_mock_de_results(n_genes: int = 2000, seed: int = 42) -> pd.DataFrame:
    """DESeq2-style DataFrame with log2FoldChange, padj, baseMean, and gene names."""
    rng = np.random.default_rng(seed)
    gene_names = [f"Gene{i+1:04d}" for i in range(n_genes)]
    logfc = rng.normal(0, 1.5, n_genes)
    base_mean = rng.lognormal(4, 1.5, n_genes)
    pval_raw = rng.beta(0.5, 5, n_genes)
    # Make ~10% significant
    pval_raw[:200] = rng.uniform(0, 0.001, 200)
    padj = np.minimum(pval_raw * n_genes, 1.0)
    df = pd.DataFrame({
        "gene": gene_names,
        "log2FoldChange": logfc,
        "baseMean": base_mean,
        "pvalue": pval_raw,
        "padj": padj,
    })
    return df


def make_mock_coefs(n_features: int = 100, seed: int = 42) -> pd.DataFrame:
    """Elastic net coefficient table with 'coefficient' and 'se' columns.

    About 30 % of features have non-zero coefficients.
    """
    rng = np.random.default_rng(seed)
    features = [f"Gene{i+1:04d}" for i in range(n_features)]
    coefs = np.zeros(n_features)
    nonzero_mask = rng.random(n_features) < 0.30
    coefs[nonzero_mask] = rng.normal(0, 0.5, nonzero_mask.sum())
    se = np.abs(rng.normal(0, 0.1, n_features)) + 0.02
    df = pd.DataFrame({"coefficient": coefs, "se": se}, index=features)
    df.index.name = "feature"
    return df


def make_mock_mudata(
    n_cells: int = 500,
    n_rna: int = 200,
    n_prot: int = 30,
    seed: int = 42,
):
    """MuData with RNA and protein modalities sharing the same obs."""
    try:
        import mudata
        import anndata
    except ImportError as e:
        raise ImportError(
            "mudata and anndata are required. Install with: pip install mudata anndata"
        ) from e

    rng = np.random.default_rng(seed)

    # RNA modality
    rna_names = [f"Gene{i+1:04d}" for i in range(n_rna)]
    X_rna = rng.negative_binomial(2, 0.5, (n_cells, n_rna)).astype(np.float32)
    adata_rna = anndata.AnnData(X=X_rna)
    adata_rna.var_names = rna_names

    # Protein modality
    prot_names = [f"Prot{i+1:02d}" for i in range(n_prot)]
    X_prot = np.abs(rng.normal(3, 1, (n_cells, n_prot))).astype(np.float32)
    adata_prot = anndata.AnnData(X=X_prot)
    adata_prot.var_names = prot_names

    # Shared obs
    cell_names = [f"cell{i+1:05d}" for i in range(n_cells)]
    cluster_ids = rng.integers(0, 5, n_cells)
    clusters = np.array([f"C{i}" for i in cluster_ids])

    for ad in (adata_rna, adata_prot):
        ad.obs_names = cell_names
        ad.obs["cluster"] = clusters
        ad.obs["batch"] = np.where(rng.random(n_cells) < 0.5, "batch1", "batch2")

    # Isotype controls for ADT testing
    isotype_names = ["IgG1", "IgG2a"]
    X_iso = np.abs(rng.normal(1, 0.3, (n_cells, 2))).astype(np.float32)
    adata_iso = anndata.AnnData(X=X_iso)
    adata_iso.obs_names = cell_names
    adata_iso.var_names = isotype_names
    for col in ("cluster", "batch"):
        adata_iso.obs[col] = adata_rna.obs[col].values

    mdata = mudata.MuData({"rna": adata_rna, "prot": adata_prot, "iso": adata_iso})
    return mdata


def make_mock_repertoire(
    n_cells: int = 500,
    n_clonotypes: int = 80,
    n_samples: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """obs DataFrame with clonotype_id, sample, and expansion columns.

    About 40 % of cells have a clonotype (rest are NaN).
    """
    rng = np.random.default_rng(seed)
    samples = [f"S{i+1}" for i in range(n_samples)]
    sample_col = np.choose(rng.integers(0, n_samples, n_cells), samples)

    clonotype_ids = np.array(
        [f"clone{i:03d}" for i in rng.integers(0, n_clonotypes, n_cells)],
        dtype=object,
    )
    has_tcr = rng.random(n_cells) < 0.4
    clonotype_ids[~has_tcr] = np.nan

    cluster_ids = rng.integers(0, 5, n_cells)
    clusters = np.array([f"C{i}" for i in cluster_ids])

    df = pd.DataFrame({
        "clonotype_id": clonotype_ids,
        "sample": sample_col,
        "cluster": clusters,
    })
    return df


if __name__ == "__main__":
    print("Building mock datasets …")
    df = make_mock_df()
    print(f"  DataFrame:  {df.shape}")
    adata = make_mock_anndata()
    print(f"  AnnData:    {adata.shape}")
    sce = make_mock_sce()
    print(f"  SCE:        {sce.shape}")
    print("Done.")
