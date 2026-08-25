"""Regression tests: without an explicit palette, ggnomics must not override plotnine's defaults.

The failure these guard against is subtle. plotnine picks a plot's default
color scale from the mapped column's dtype: an *ordered* pandas Categorical
selects ``scale_*_ordinal`` (a viridis ramp) while an unordered one selects
``scale_*_discrete`` (plotnine's hue palette). ggnomics builds categoricals
only to pin the order categories appear in, so marking them ordered silently
swapped every discrete legend to viridis. See
:func:`ggnomics._utils.display_dtype`.
"""

import matplotlib
import matplotlib.colors
import numpy as np
import pandas as pd
import pytest
from mizani.palettes import hue_pal
from plotnine import aes, geom_point, ggplot

import ggnomics as gg
from ggnomics._utils import display_categorical, display_dtype

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rendered_hexes(plot):
    """Distinct hex colors of the drawn data layers, in first-seen order."""
    figure = (plot.plot if hasattr(plot, "plot") else plot).draw()
    colors = []
    try:
        for axis in figure.axes:
            for collection in axis.collections:
                face = np.asarray(collection.get_facecolor(), dtype=float)
                if face.size:
                    colors += [matplotlib.colors.to_hex(row[:3]) for row in np.atleast_2d(face) if len(row) >= 3]
            for patch in axis.patches:
                face = patch.get_facecolor()
                if face is not None and len(face) >= 3:
                    colors.append(matplotlib.colors.to_hex(face[:3]))
    finally:
        matplotlib.pyplot.close(figure)

    seen, unique = set(), []
    for color in colors:
        lowered = color.lower()
        if lowered not in seen:
            seen.add(lowered)
            unique.append(lowered)
    return unique


def _hue_defaults(n):
    """The colors plotnine's default discrete scale assigns to ``n`` categories."""
    return {matplotlib.colors.to_hex(color).lower() for color in hue_pal()(n)}


@pytest.fixture(scope="module")
def df(mock_df):
    return mock_df


GENES = ["Gene0001", "Gene0002", "Gene0003"]


# ---------------------------------------------------------------------------
# The dtype helper itself
# ---------------------------------------------------------------------------


def test_display_dtype_is_unordered_but_keeps_order():
    order = ["E", "C", "A", "D", "B"]
    dtype = display_dtype(order)
    assert dtype.ordered is False
    assert list(dtype.categories) == order


def test_display_categorical_keeps_declared_order():
    order = ["E", "C", "A", "D", "B"]
    values = display_categorical(list("ABCDE"), order)
    assert values.ordered is False
    assert list(values.categories) == order


def test_ordered_categorical_would_change_the_scale():
    """Documents the plotnine behavior these tests exist to protect against."""
    frame = pd.DataFrame({"g": list("ABCDE") * 4, "v": range(20)})

    unordered = frame.assign(g=display_categorical(frame["g"], list("ABCDE")))
    ordered = frame.assign(g=pd.Categorical(frame["g"], categories=list("ABCDE"), ordered=True))

    def scale_names(data):
        plot = ggplot(data) + aes("v", "v", color="g") + geom_point()
        matplotlib.pyplot.close(plot.draw())
        return {type(scale).__name__ for scale in plot.scales}

    assert any("discrete" in name for name in scale_names(unordered))
    assert any("ordinal" in name for name in scale_names(ordered))


# ---------------------------------------------------------------------------
# Public plotting functions: discrete color/fill with no palette
# ---------------------------------------------------------------------------


def _discrete_cases(df):
    """(name, plot, n_categories) for every function mapping a category to color/fill."""
    clusters = df["cluster"].nunique()
    samples = df["sample"].nunique()
    return [
        ("plot_scatter", gg.plot_scatter(df, "UMAP1", "UMAP2", color="cluster"), clusters),
        ("plot_umap", gg.plot_umap(df, color="cluster"), clusters),
        ("plot_pca", gg.plot_pca(df, color="cluster"), clusters),
        ("plot_coldata/point", gg.plot_coldata(df, x="UMAP1", y="UMAP2", color_by="cluster"), clusters),
        (
            "plot_coldata/violin",
            gg.plot_coldata(df, x="cluster", y="n_counts", shape="violin", color_by="cluster"),
            clusters,
        ),
        ("plot_coldata/box", gg.plot_coldata(df, x="cluster", y="n_counts", shape="box", color_by="cluster"), clusters),
        ("plot_coldata/bar", gg.plot_coldata(df, x="cluster", y="n_counts", shape="bar", color_by="cluster"), clusters),
        ("plot_expression", gg.plot_expression(df, features=GENES, group_by="cluster"), clusters),
        ("plot_violin_stats", gg.plot_violin_stats(df, feature="Gene0001", group_by="cluster"), clusters),
        ("plot_box_stats", gg.plot_box_stats(df, feature="Gene0001", group_by="cluster"), clusters),
        ("plot_highest_exprs", gg.plot_highest_exprs(df, n=5, features=GENES, color_cells_by="cluster"), clusters),
        ("plot_pairs", gg.plot_pairs(df, dimred="PCA", n_components=3, color_by="cluster"), clusters),
        ("plot_abundance", gg.plot_abundance(df, group_by="sample", color_by="cluster"), clusters),
        ("plot_scatter_marginal", gg.plot_scatter_marginal(df, x="UMAP1", y="UMAP2", color="cluster"), clusters),
        ("qc_scatter", gg.qc_scatter(df, x="n_counts", y="n_genes_detected", color="cluster"), clusters),
        ("ridge_density", gg.ridge_density(df, value="n_counts", group="cluster"), clusters),
        (
            "cluster_composition_barplot",
            gg.cluster_composition_barplot(df, cluster_col="cluster", group_col="sample"),
            clusters,
        ),
        # Two composed panels, colored by sample and by cluster respectively.
        (
            "plot_pseudobulk_qc",
            gg.plot_pseudobulk_qc(df, sample_by="sample", group_by="cluster", features=GENES),
            max(samples, clusters),
        ),
    ]


def test_discrete_color_uses_plotnine_default_palette(df):
    """No palette given → every mapped color is one plotnine's own hue scale would pick."""
    # Neutral ink plotnine draws regardless of the color scale (outlines,
    # boxplot fills, significance brackets, reference lines).
    structural = {"#000000", "#ffffff", "#333333", "#595959"}

    offenders = {}
    for name, plot, n_categories in _discrete_cases(df):
        mapped = [color for color in _rendered_hexes(plot) if color not in structural]
        # A composed figure may draw panels with different category counts,
        # so accept any hue palette from 1..n.
        allowed = set().union(*(_hue_defaults(k) for k in range(1, n_categories + 1)))
        unexpected = [color for color in mapped if color not in allowed]
        if unexpected or not mapped:
            offenders[name] = unexpected or "no colors rendered"

    assert not offenders, f"functions not using plotnine's default discrete palette: {offenders}"


def test_continuous_color_matches_bare_plotnine(df):
    """A numeric color column must render exactly as the equivalent bare plotnine plot."""
    frame = df[["UMAP1", "UMAP2", "n_counts"]]
    reference = ggplot(frame) + aes(x="UMAP1", y="UMAP2", color="n_counts") + geom_point()

    assert _rendered_hexes(gg.plot_scatter(frame, "UMAP1", "UMAP2", color="n_counts")) == _rendered_hexes(reference)


def test_no_color_scale_is_added_without_a_palette(df):
    """The defaults hold because ggnomics adds no scale at all, not because it re-picks one."""
    for _, plot, _ in _discrete_cases(df):
        target = plot.plot if hasattr(plot, "plot") else plot
        if not hasattr(target, "scales"):
            continue
        matplotlib.pyplot.close(target.draw())
        imposed = [
            type(scale).__name__
            for scale in target.scales
            if ("color" in type(scale).__name__ or "fill" in type(scale).__name__)
            and not type(scale).__name__.endswith(("_discrete", "_ordinal", "_continuous", "_datetime"))
        ]
        assert not imposed, f"unexpected explicit color scale(s): {imposed}"


# ---------------------------------------------------------------------------
# Partial palettes
# ---------------------------------------------------------------------------


def test_partial_palette_falls_back_for_unlisted_categories(df):
    """A palette covering one category must not flatten the rest to grey.

    Every function routes through :func:`ggnomics._utils.color_scale`, so an
    unlisted category keeps a real color from the default palette instead of
    the single ``na_value`` plotnine would otherwise assign it.
    """
    first = sorted(df["cluster"].unique())[0]
    palette = {first: "#000000"}
    n_categories = df["cluster"].nunique()

    cases = {
        "plot_scatter": gg.plot_scatter(df, "UMAP1", "UMAP2", color="cluster", palette=palette),
        "plot_umap": gg.plot_umap(df, color="cluster", palette=palette),
        "plot_coldata": gg.plot_coldata(df, x="UMAP1", y="UMAP2", color_by="cluster", palette=palette),
        "plot_violin_stats": gg.plot_violin_stats(df, feature="Gene0001", group_by="cluster", palette=palette),
    }

    for name, plot in cases.items():
        colors = set(_rendered_hexes(plot))
        assert "#000000" in colors, f"{name}: the explicitly requested color is missing"
        # Distinct colors for the remaining categories, not one repeated grey.
        distinct = colors - {"#333333", "#ffffff"}
        assert len(distinct) >= n_categories, f"{name}: unlisted categories collapsed to {sorted(distinct)}"


def test_partial_palette_is_consistent_across_functions(df):
    """The scatter path and the coldata path resolve a partial palette identically."""
    palette = {sorted(df["cluster"].unique())[0]: "#000000"}

    scatter = set(_rendered_hexes(gg.plot_scatter(df, "UMAP1", "UMAP2", color="cluster", palette=palette)))
    coldata = set(_rendered_hexes(gg.plot_coldata(df, x="UMAP1", y="UMAP2", color_by="cluster", palette=palette)))

    assert scatter == coldata
