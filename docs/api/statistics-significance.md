# Statistics & significance

`ggnomics.signif` is the ggsignif-inspired significance-bracket submodule.
`geom_signif` in automatic mode (comparisons given, no manual `y_position`)
returns a deferred placeholder rather than a plain plotnine layer — call
`.resolve(df, x_col, y_col)` to get concrete layers, or use
`plot_violin_stats`/`plot_box_stats`, which do this internally.

::: ggnomics.signif.geom_signif
::: ggnomics.signif.run_comparisons
::: ggnomics.signif.map_pvalue_to_stars
::: ggnomics.stats_plots.plot_violin_stats
::: ggnomics.stats_plots.plot_box_stats
::: ggnomics.stats_plots.plot_scatter_marginal
::: ggnomics.stats_plots.plot_embedding_panel
::: ggnomics.pairs.plot_pairs
