# Tutorials

These workflow guides run against real, publicly available datasets distributed
through BiocPy and scverse data-access packages. Each page is an executed
Jupyter notebook: the figures shown are the ones the code produced.

The notebooks are generated from the Quarto sources in `vignettes/` with
`quarto render <vignette>.qmd --to ipynb` and are committed with their outputs,
so building the documentation needs neither network access nor the analysis
stack. See [Contributing](../contributing.md) for how to regenerate them.

```{toctree}
:maxdepth: 1

getting-started
single-cell-scranpy
bulk-rnaseq
upset
multimodal
specialized-plots
plot-gallery
```

## Which guide to read

* **Getting started** introduces DataFrame, `SingleCellExperiment`, and AnnData
  input, and shows how to customize the returned plots.
* **Single-cell analysis with scranpy** follows a mouse brain dataset from
  quality control through clustering to UMAP.
* **Bulk RNA-seq** uses PyDESeq2 on a rice salt-stress experiment.
* **UpSet plots** compares differentially expressed genes across three
  timepoints with `ggnomics.upset`, a native plotnine set-intersection plot.
* **Multimodal** combines RNA and ADT measurements from CITE-seq data.
* **Specialized plots** covers abundance, pseudobulk QC, statistics,
  significance brackets, palettes, heatmaps, and immune repertoires.
* **Plot gallery** is a compact reference organized by plot family.
