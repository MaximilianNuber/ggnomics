# ggnomics

Composable, ggplot2-style genomics visualization for Python. Every public
`plot_*` function returns a real [plotnine](https://plotnine.org) object —
inspect it, extend it with ordinary plotnine layers, theme it, facet it, or
compose it with other panels.

This site has two parts:

* **[API reference](api/reduced-dimensions.md)** — every public function,
  organized by family, generated from its docstring.
* **[Vignettes](vignettes.md)** — the full, real-dataset Quarto workflows
  (getting started, single-cell analysis with scranpy, bulk RNA-seq with
  PyDESeq2, multimodal CITE-seq, specialized plots, and a comprehensive plot
  gallery).

See the [project README](https://github.com/yourname/ggnomics#readme) for
installation and a quick overview.

## Design in one sentence

Every public `plot_*` function is a single `functools.singledispatch`
generic. The base registration is `pd.DataFrame` (its docstring is the
canonical documentation); optional adapters for `AnnData`,
`SingleCellExperiment`, `SummarizedExperiment`, and `MuData` build an
equivalent DataFrame internally and delegate to that same implementation,
registering themselves automatically the moment the corresponding package is
importable.

## Unsupported combinations

A plain `SummarizedExperiment` has no stored reduced dimensions —
embedding-dependent functions (`plot_umap`, `plot_pca`, `plot_reduced_dim`,
`plot_pairs`, `plot_embedding_panel`, `plot_clonotype_embedding`) correctly
raise the ordinary singledispatch `TypeError` for it. Bulk PCA is instead an
explicit analysis step, `ggnomics.bulk_pca`, that computes and plots PCA
scores from an assay rather than reading a stored embedding. `MuData` is
registered only for genuinely cross-modality functions
(`plot_bimodal_scatter`, `plot_adt_qc`) — every other function operates on
one modality's `AnnData` directly.
