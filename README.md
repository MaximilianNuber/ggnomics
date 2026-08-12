# ggnomics

**Composable, ggplot2-style genomics visualization for Python.**

`ggnomics` brings composable, ggplot2-style genomics visualization to Python
by returning [plotnine](https://plotnine.org) objects that users can inspect,
extend, theme, facet, and combine.

## Why not a fixed plotting API?

Most genomics plotting libraries hand you a finished PNG-shaped object: fix
the title, fix the palette, fix the layout, or drop down to raw matplotlib.
`ggnomics` instead returns the same `plotnine.ggplot` object you'd build by
hand — every `plot_*` function is a thin, well-tested builder of a real
grammar-of-graphics object, not a rendering black box. Add a theme, swap a
scale, facet by a column that wasn't part of the original call, or compose
several panels with `|`/`/` — all with plotnine you already know, no
ggnomics-specific extension API required.

## What input containers are supported?

| Container | Package | Notes |
|---|---|---|
| `pd.DataFrame` | pandas | The canonical, documented interface — core install only |
| `AnnData` | anndata / scverse | |
| `SingleCellExperiment` | BiocPy `singlecellexperiment` | |
| `SummarizedExperiment` | BiocPy `summarizedexperiment` | No reduced dimensions — see below |
| `MuData` | mudata / scverse | Cross-modality functions only |

Every public `plot_*` function is a `functools.singledispatch` generic whose
base registration is `pd.DataFrame` — that implementation's docstring is
the documentation you see in `help()`/Jupyter. Container adapters build an
equivalent DataFrame internally and delegate to it, and they register
themselves automatically the moment the corresponding package is
importable — there is no separate "enable this backend" step. A plain
`SummarizedExperiment` has no stored reduced dimensions, so
embedding-dependent functions (`plot_umap`, `plot_pca`, `plot_pairs`, ...)
correctly raise `TypeError` for it; bulk PCA is instead an explicit analysis
step (`ggnomics.bulk_pca`).

## What kinds of plots?

Reduced dimensions (UMAP/PCA/t-SNE), expression violins/dot plots/heatmaps,
cell/sample metadata plots, abundance and composition, QC panels,
highest-expressed-feature plots, differential-expression volcano/MA/
coefficient plots, pseudobulk QC and DE, CITE-seq-style multimodal scatter
and ADT QC, immune-repertoire clonotype plots, and ggsignif-style
significance brackets. The full inventory, with a live example per
function, is the [plot gallery](https://yourname.github.io/ggnomics/vignettes/plot-gallery.html).

## Installation

```bash
pip install ggnomics
```

Optional extras register their container/statistics backends automatically
once installed — see [Optional dependencies](#optional-dependencies) below.

## A small, realistic example

Using a deterministic subset of the Zeisel et al. (2015) mouse brain
single-cell dataset, fetched via BiocPy's
[`scrnaseq`](https://github.com/BiocPy/scrnaseq) (full walkthrough:
[Getting started](https://yourname.github.io/ggnomics/vignettes/getting-started.html)):

```python
import scrnaseq
import ggnomics as gg

sce = scrnaseq.fetch_dataset("zeisel-brain-2015", "2023-12-14", realize_assays=True)
gg.plot_expression(sce, features=["Gfap"], group_by="level1class", layer="counts", log1p=True)
```

## Returned plots are composable

```python
import ggnomics as gg
from plotnine import theme_bw, labs

# sce_with_umap: a SingleCellExperiment with a UMAP already computed
# (see "Single-cell analysis with scranpy" for how to get here from raw counts)
umap = gg.plot_umap(sce_with_umap, color="cluster")
violin = gg.plot_expression(sce_with_umap, features=["Gfap"], group_by="cluster", layer="logcounts")

# Extend with any plotnine layer
umap + theme_bw() + labs(title="My figure")

# Compose panels side by side / stacked
(umap | violin) / gg.plot_volcano(de_results)
```

## Feature gallery

Every figure below is generated from a real public dataset in this
repository's Quarto documentation — nothing here is synthetic. Each links to
the full workflow that produced it.

| | |
|---|---|
| ![Zeisel mouse brain UMAP, colored by cell class](docs/img/readme/zeisel_umap_level1class.png) `plot_umap` — full scranpy QC-to-UMAP workflow on the Zeisel et al. (2015) mouse brain dataset. [Single-cell analysis with scranpy →](https://yourname.github.io/ggnomics/vignettes/single-cell-scranpy.html) | ![Marker gene violins across Zeisel cell classes](docs/img/readme/zeisel_marker_violins.png) `plot_expression` — canonical marker genes across the same cell classes. [Single-cell analysis with scranpy →](https://yourname.github.io/ggnomics/vignettes/single-cell-scranpy.html) |
| ![E-MTAB-1625 volcano plot](docs/img/readme/emtab1625_volcano.png) `plot_volcano` — PyDESeq2 differential expression, 24h rice salt stress vs. control (E-MTAB-1625). [Bulk RNA-seq →](https://yourname.github.io/ggnomics/vignettes/bulk-rnaseq.html) | ![Baron pancreas cell-type composition per donor](docs/img/readme/baron_composition.png) `plot_abundance` — pancreatic cell-type composition per donor, Baron et al. (2016) human pancreas. [Specialized plots →](https://yourname.github.io/ggnomics/vignettes/specialized-plots.html) |
| ![RNA- and protein-colored UMAP panels, CITE-seq PBMC data](docs/img/readme/cite_seq_rna_protein_umap.png) `plot_bimodal_scatter`/`plot_umap` — RNA/ADT interoperability on real CITE-seq PBMC data. [Multimodal →](https://yourname.github.io/ggnomics/vignettes/multimodal.html) | ![TCR clonotype abundance, CTCL vs. control](docs/img/readme/tcr_clonotype_abundance.png) `plot_clonotype_abundance` — real ECCITE-seq TCR repertoire data, CTCL vs. control. [Specialized plots →](https://yourname.github.io/ggnomics/vignettes/specialized-plots.html) |
| ![Marsilea heatmap of top differentially expressed genes](docs/img/readme/marsilea_heatmap.png) Real ggnomics/PyDESeq2 results rendered as a [Marsilea](https://marsilea.readthedocs.io/) composable heatmap (interoperability, not a ggnomics wrapper). [Specialized plots →](https://yourname.github.io/ggnomics/vignettes/specialized-plots.html) | ![Significance brackets on a Zeisel violin plot](docs/img/readme/significance_brackets.png) `plot_violin_stats` / `geom_signif` — ggsignif-style significance brackets, real Zeisel expression data. [Specialized plots →](https://yourname.github.io/ggnomics/vignettes/specialized-plots.html) |

## Interoperability

`ggnomics` doesn't perform analysis — it visualizes the output of the tools
that do. The full documentation site exercises every row below against a
real public dataset:

| Tool | Role |
|---|---|
| pandas | The canonical input/output data structure |
| BiocPy `SingleCellExperiment` / `SummarizedExperiment` | Bioconductor-style containers |
| AnnData / scverse | Single-cell container and ecosystem |
| MuData | Multimodal (CITE-seq-style) container |
| [scranpy](https://github.com/BiocPy/scranpy) | Single-cell QC, normalization, clustering, UMAP |
| [PyDESeq2](https://pydeseq2.readthedocs.io/) | Bulk differential expression |
| [Marsilea](https://marsilea.readthedocs.io/) | Composable clustered heatmaps |
| [plotnine](https://plotnine.org) | Every returned object, and everything you add to it |

## Documentation

* [Documentation home](https://yourname.github.io/ggnomics/vignettes/index.html)
* [Getting started](https://yourname.github.io/ggnomics/vignettes/getting-started.html)
* [Single-cell analysis with scranpy](https://yourname.github.io/ggnomics/vignettes/single-cell-scranpy.html)
* [Bulk RNA-seq](https://yourname.github.io/ggnomics/vignettes/bulk-rnaseq.html)
* [Multimodal](https://yourname.github.io/ggnomics/vignettes/multimodal.html)
* [Specialized plots](https://yourname.github.io/ggnomics/vignettes/specialized-plots.html)
* [Plot gallery](https://yourname.github.io/ggnomics/vignettes/plot-gallery.html)
* [API reference](https://yourname.github.io/ggnomics/api/reduced-dimensions/)

## Optional dependencies

The core install (`numpy`, `pandas`, `plotnine`) is enough for every
DataFrame-based plotting function. Everything else is opt-in:

```bash
pip install "ggnomics[anndata]"    # AnnData / scverse
pip install "ggnomics[sce]"        # SingleCellExperiment (BiocPy)
pip install "ggnomics[se]"         # SummarizedExperiment (BiocPy)
pip install "ggnomics[multimodal]" # MuData
pip install "ggnomics[stats]"      # significance brackets (scipy, statsmodels)
pip install "ggnomics[pca]"        # PCA utilities (scikit-learn, scipy)
pip install "ggnomics[pseudobulk]" # pseudobulk QC PCA panel
pip install "ggnomics[upset]"      # UpSet-style DE overlap plots
pip install "ggnomics[umap]"       # ggnomics.sc_umap (umap-learn)
pip install "ggnomics[all]"        # everything above
```

`scranpy`, `PyDESeq2`, `Marsilea`, `scrnaseq`, `pyexpressionatlas`, and
`experimenthub` are demonstrated throughout the documentation but are **not**
ggnomics dependencies — they're independent tools ggnomics interoperates
with.

## Status and license

`ggnomics` is under active development; the public API documented here and
in the [full documentation site](https://yourname.github.io/ggnomics/) is
the current, authoritative surface. A small legacy API (`expression_violin`,
`volcano_plot`, `heatmap_from_matrix`, ...) is kept for backwards
compatibility — see the [plot gallery](https://yourname.github.io/ggnomics/vignettes/plot-gallery.html)
for the current-vs-legacy mapping.

Licensed under the [MIT License](LICENSE). Contributions welcome — see
[`docs/api/`](docs/api/) for the API reference and `tests/` for the test
suite conventions.
