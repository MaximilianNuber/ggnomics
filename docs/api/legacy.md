# Legacy API

These functions predate the current `plot_*` generics and remain
fully functional and backwards-compatible — they are independent
implementations, not thin wrappers, so behavior can differ slightly (fewer
options, no container dispatch beyond what's noted).

| Legacy | Current equivalent |
|---|---|
| `expression_violin` / `expression_violin_sce` / `expression_violin_se` | `plot_expression` |
| `volcano_plot` | `plot_volcano` |
| `heatmap_long` / `heatmap_from_matrix` | `plot_heatmap` |
| `marker_dotplot` / `marker_dotplot_from_matrix` | `plot_dot` |
| `qc_scatter` / `qc_histogram` | `plot_coldata` |
| `cluster_composition_barplot` | `plot_abundance` |
| `ridge_density` | *(no modern replacement)* |

## A note on the removed accessor API

An earlier `ggnomics._accessor` module existed at one point in this
project's history and has since been removed entirely — there is no pandas/
AnnData `.ggnomics` accessor namespace in the current API. All plotting goes
through the top-level `plot_*` functions documented here.

::: ggnomics.violin.expression_violin
::: ggnomics.violin.expression_violin_sce
::: ggnomics.violin.expression_violin_se
::: ggnomics.volcano.volcano_plot
::: ggnomics.dotplot.marker_dotplot
::: ggnomics.dotplot.marker_dotplot_from_matrix
::: ggnomics.ridge.ridge_density
