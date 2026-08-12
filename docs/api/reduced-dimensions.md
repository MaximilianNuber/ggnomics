# Reduced dimensions

`plot_scatter` and `plot_embedding` are the workhorses; `plot_reduced_dim`/
`dim_plot` are true aliases of `plot_embedding`, and `plot_umap`/`plot_pca`/
`plot_tsne` are `dimred=` presets. All accept `pd.DataFrame`, `AnnData`, and
`SingleCellExperiment` (not `SummarizedExperiment`, which has no reduced
dimensions — see the [dispatch note](../index.md#unsupported-combinations)).

See [Getting started](https://yourname.github.io/ggnomics/vignettes/getting-started.html) and
[Single-cell analysis with scranpy](https://yourname.github.io/ggnomics/vignettes/single-cell-scranpy.html)
for worked examples.

::: ggnomics.scatter.plot_scatter
::: ggnomics.scatter.plot_embedding
::: ggnomics.scatter.plot_umap
::: ggnomics.scatter.plot_pca
::: ggnomics.scatter.plot_tsne

## Analysis-and-plot PCA/UMAP submodules

Three separate implementations exist — pick the one matching your
container and dependency budget:

* `ggnomics.bulk_pca` (`ggnomics.bulk.pca`) — SummarizedExperiment, sklearn/SVD PCA.
* `ggnomics.sc_pca` (`ggnomics.singlecell.pca`) — SingleCellExperiment, sklearn/SVD PCA.
* `ggnomics.sc_umap` (`ggnomics.singlecell.umap`) — SingleCellExperiment, requires `umap-learn`.

::: ggnomics.bulk.pca.run_pca
::: ggnomics.bulk.pca.plot_pca
::: ggnomics.singlecell.pca.run_pca
::: ggnomics.singlecell.pca.plot_pca
::: ggnomics.singlecell.umap.run_umap
::: ggnomics.singlecell.umap.plot_umap
