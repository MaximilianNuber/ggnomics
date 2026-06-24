import numpy as np
import pandas as pd

from gg_singlecell import (
    plot_reduced_dim,
    expression_violin,
    volcano_plot,
    heatmap_from_matrix,
    marker_dotplot_from_matrix,
    qc_scatter,
    qc_histogram,
    cluster_composition_barplot,
    ridge_density,
)

# Synthetic embedding
np.random.seed(0)
n = 1000
clusters = np.random.choice(list("ABC"), size=n, p=[0.4, 0.3, 0.3])
cond = np.random.choice(["ctrl", "stim"], size=n)
df = pd.DataFrame({
    "umap_1": np.random.randn(n),
    "umap_2": np.random.randn(n),
    "cluster": clusters,
    "condition": cond,
    "n_counts": np.random.gamma(2, 1000, size=n),
    "n_genes": np.random.gamma(2, 500, size=n),
    "mito_frac": np.random.beta(2, 10, size=n),
})

# 1) Reduced-dim plot
p1 = plot_reduced_dim(df, x="umap_1", y="umap_2", color="cluster", title="Embedding")
print(p1)

# 2) QC scatter + histogram
p2 = qc_scatter(df, x="n_counts", y="n_genes", color="mito_frac", title="QC scatter")
print(p2)

p3 = qc_histogram(df, value="mito_frac", title="Mito fraction")
print(p3)

# 3) Violin plot of a metric by cluster
p4 = expression_violin(df, value="n_counts", group="cluster", log1p=True, title="n_counts by cluster")
print(p4)

# 4) Volcano
m = 200
de = pd.DataFrame({
    "log2FC": np.random.randn(m),
    "FDR": np.clip(np.random.rand(m), 1e-10, 1.0),
})
p5 = volcano_plot(de, log2fc="log2FC", padj="FDR", title="Volcano")
print(p5)

# 5) Heatmap from matrix
mat = np.random.randn(10, 5)
p6 = heatmap_from_matrix(mat, row_names=[f"g{i}" for i in range(10)], col_names=[f"c{j}" for j in range(5)], zscore_rows=True)
print(p6)

# 6) Marker dot-plot from matrix
expr = np.abs(np.random.randn(n, 6))
p7 = marker_dotplot_from_matrix(expr, groups=clusters, genes=["g0","g1","g2"])  # gene names will default
print(p7)

# 7) Composition
p8 = cluster_composition_barplot(df, cluster_col="cluster", group_col="condition", normalize=True, title="Cluster composition")
print(p8)

# 8) Ridge density
p9 = ridge_density(df, value="n_counts", group="cluster", title="Ridge density")
print(p9)
