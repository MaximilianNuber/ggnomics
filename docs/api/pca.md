# PCA

Three independent implementations exist:

* `ggnomics.pca` (`run_pca_sklearn`, `run_pca_svd`, `PcaResult`) — the
  generic, container-agnostic engine. Not re-exported at the top level;
  consumed internally by the two modules below.
* `ggnomics.bulk_pca` — SummarizedExperiment-oriented (bulk data).
* `ggnomics.sc_pca` / `ggnomics.sc_umap` — SingleCellExperiment-oriented.

The top-level `ggnomics.plot_pca` (in `ggnomics.scatter`) is a different
thing entirely: it only *displays* an already-computed embedding and
performs no PCA itself.

::: ggnomics.pca.methods.run_pca_sklearn
::: ggnomics.pca.methods.run_pca_svd
::: ggnomics.pca.result.PcaResult
::: ggnomics.bulk.pca.run_pca
::: ggnomics.bulk.pca.plot_pca
::: ggnomics.singlecell.pca.run_pca
::: ggnomics.singlecell.pca.plot_pca
