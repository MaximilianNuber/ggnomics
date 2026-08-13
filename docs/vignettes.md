# Vignettes

The full, real-dataset workflows are a separate [Quarto](https://quarto.org)
project under `vignettes/` (not part of this MkDocs build — Quarto and
MkDocs are two independent static sites, combined at deploy time; see
below). Render them locally with:

```bash
mamba run -n ggnomics-docs env \
  QUARTO_PYTHON="$(mamba run -n ggnomics-docs which python)" \
  quarto render vignettes
```

which produces `vignettes/_site/*.html`. To publish both sites together
under one domain (this MkDocs site plus the Quarto vignettes under
`/vignettes/`):

```bash
mkdocs build
cp -r vignettes/_site site/vignettes
```

## Pages

* [Home](https://maximiliannuber.github.io/ggnomics/vignettes/index.html) — landing
  page, dataset provenance, dispatch overview.
* [Getting started](https://maximiliannuber.github.io/ggnomics/vignettes/getting-started.html) —
  the fastest path to a working plot, on a deterministic Zeisel subset.
* [Single-cell analysis with scranpy](https://maximiliannuber.github.io/ggnomics/vignettes/single-cell-scranpy.html) —
  a complete QC-to-UMAP workflow on the full Zeisel dataset.
* [Bulk RNA-seq](https://maximiliannuber.github.io/ggnomics/vignettes/bulk-rnaseq.html) —
  PyDESeq2 differential expression on E-MTAB-1625 (rice salt stress).
* [UpSet plots](https://maximiliannuber.github.io/ggnomics/vignettes/upset.html) —
  native Plotnine set-intersection plots comparing DE genes across the three
  E-MTAB-1625 timepoints.
* [Multimodal](https://maximiliannuber.github.io/ggnomics/vignettes/multimodal.html) —
  RNA/ADT CITE-seq interoperability.
* [Specialized plots](https://maximiliannuber.github.io/ggnomics/vignettes/specialized-plots.html) —
  abundance, pseudobulk, statistics, significance brackets, palettes,
  immune repertoire.
* [Plot gallery](https://maximiliannuber.github.io/ggnomics/vignettes/plot-gallery.html) —
  every public function, one place, reusing cached results from the pages
  above.

Dataset provenance (accessions, versions, retrieval dates, and any fallback
resource actually used) is tracked in
[`vignettes/datasets.yml`](https://github.com/MaximilianNuber/ggnomics/blob/main/vignettes/datasets.yml)
and rendered on the vignettes home page.
