# UpSet plots

`ggnomics.upset` is a native [plotnine](https://plotnine.org) implementation
of set-intersection ("UpSet") plots in the style of
[ComplexUpset](https://github.com/krassowski/complex-upset). It has no
runtime dependency on the Matplotlib-based `upsetplot` package. See the
[Native UpSet plots](https://maximiliannuber.github.io/ggnomics/vignettes/upset.html)
vignette for a full real-data workflow; this page documents individual
function signatures.

Both import forms resolve to the module:

```python
import ggnomics.upset as upset
from ggnomics import upset
```

`ggnomics.upset.upset` is the plotting function.

## High-level plotting

::: ggnomics.upset.upset

## Compute, select, apply

`upset()` and `upset_data()` are convenience wrappers around three
independently usable steps.

::: ggnomics.upset.upset_data
::: ggnomics.upset.compute_intersections
::: ggnomics.upset.select_intersections
::: ggnomics.upset.apply_intersection_selection
::: ggnomics.upset.report_dropped_sets

## Annotations and layout components

::: ggnomics.upset.intersection_size
::: ggnomics.upset.intersection_ratio
::: ggnomics.upset.upset_annotate
::: ggnomics.upset.intersection_matrix
::: ggnomics.upset.upset_set_size
::: ggnomics.upset.upset_stripes

## Modes and percentage helpers

::: ggnomics.upset.get_size_mode
::: ggnomics.upset.upset_mode
::: ggnomics.upset.upset_text_percentage
::: ggnomics.upset.aes_percentage
::: ggnomics.upset.reverse_log_trans

## Queries and themes

::: ggnomics.upset.upset_query
::: ggnomics.upset.upset_default_themes
::: ggnomics.upset.upset_modify_themes

## Statistical comparisons

Requires the optional `ggnomics[upset]` extra (`scipy`, `statsmodels`),
imported lazily only when these functions are called. Only inferentially
valid when the rows compared represent appropriate independent statistical
units — see the vignette's
["Statistical-helper caution"](https://maximiliannuber.github.io/ggnomics/vignettes/upset.html#15-statistical-helper-caution).

::: ggnomics.upset.compare_between_intersections
::: ggnomics.upset.upset_test

## Venn helpers

Approximate geometric/raster representations for two to four sets; region
areas are visually indicative, not exactly proportional.

::: ggnomics.upset.arrange_venn
::: ggnomics.upset.compute_venn_layout
::: ggnomics.upset.geom_venn_circle
::: ggnomics.upset.geom_venn_region
::: ggnomics.upset.geom_venn_label_region
::: ggnomics.upset.geom_venn_label_set
::: ggnomics.upset.scale_color_venn_mix
::: ggnomics.upset.scale_fill_venn_mix

## Result and specification dataclasses

::: ggnomics.upset.UpSetData
::: ggnomics.upset.IntersectionStatistics
::: ggnomics.upset.IntersectionSelection
::: ggnomics.upset.UpSetAnnotation
::: ggnomics.upset.UpSetQuery
::: ggnomics.upset.IntersectionMatrixSpec
::: ggnomics.upset.SetSizeSpec
::: ggnomics.upset.UpSetStripes
::: ggnomics.upset.VennLayout
