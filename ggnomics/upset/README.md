# `ggnomics.upset`

This directory implements a Plotnine-native, composable translation of the
public [ComplexUpset](https://github.com/krassowski/complex-upset) R package
API. It does not depend on the Matplotlib-based `upsetplot` package. See the
[Native UpSet plots vignette](https://maximiliannuber.github.io/ggnomics/vignettes/upset.html)
for a full real-data workflow, and the
[API reference](https://maximiliannuber.github.io/ggnomics/api/upset.html)
for individual function signatures.

```python
import ggnomics.upset as upset

plot = upset.upset(
    data,
    ["set_a", "set_b", "set_c"],
    queries=[upset.upset_query(intersect=["set_a", "set_b"], color="#CC3311")],
)
plot.save("upset.png", width=10, height=6, dpi=150)
```

The high-level `upset()` function is backed by an explicit three-step API:

```python
statistics = upset.compute_intersections(
    data,
    intersect=["set_a", "set_b", "set_c"],
)
selection = upset.select_intersections(statistics, min_size=5)
prepared = upset.apply_intersection_selection(
    statistics,
    selection=selection,
)
plot = upset.compose_upset(prepared)
```

The submodule exports the same named entry points as ComplexUpset, including
intersection annotations, queries, themes, set-size and matrix specifications,
four membership modes, statistical comparisons, and the two-/three-/four-set Venn
helpers. Python-specific result dataclasses make intermediate state inspectable
and prevent accidental mutation of caller-owned data.

The high-level `upset()` function and everything needed to build and render a
plot (`compute_intersections`, `select_intersections`,
`apply_intersection_selection`, annotations, matrix/stripe/set-size specs,
queries, themes, and the Venn helpers) require only ggnomics's normal package
dependencies. `compare_between_intersections()` and `upset_test()` (between-
intersection statistical comparisons) additionally require `scipy` and
`statsmodels`, imported lazily only when those two functions are called;
install them with `pip install "ggnomics[upset]"`. Plain
`import ggnomics.upset` never requires `scipy` or `statsmodels`.

Plotnine `0.16.0a11` is pinned because nested composition layout dimensions and
guide collection are used by the UpSet layout. The top-level plotting function
returns a native `plotnine.composition.Compose`, so it can be combined with
other Plotnine plots using `/`, `|`, and `+ plot_layout(...)`.

See [`THIRD-PARTY-NOTICES-upset.md`](THIRD-PARTY-NOTICES-upset.md) for the
ComplexUpset MIT license attribution.
