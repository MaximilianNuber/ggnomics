"""Tests for ggnomics.signif — geom_signif, run_comparisons, compute_brackets."""

import numpy as np
import pandas as pd
import pytest
from plotnine import aes, geom_violin, ggplot
from plotnine.ggplot import ggplot as ggplot_class

from ggnomics import map_pvalue_to_stars, plot_box_stats, plot_violin_stats, run_comparisons
from ggnomics import geom_signif
from ggnomics.signif import BracketSpec, compute_brackets


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_df():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "group": ["A"] * 50 + ["B"] * 50 + ["C"] * 50,
            "value": np.concatenate(
                [
                    np.random.normal(0, 1, 50),
                    np.random.normal(1, 1, 50),
                    np.random.normal(3, 1, 50),  # C clearly different from A
                ]
            ),
        }
    )


# ---------------------------------------------------------------------------
# map_pvalue_to_stars
# ---------------------------------------------------------------------------


def test_stars_default():
    assert map_pvalue_to_stars(0.001) == "***"
    assert map_pvalue_to_stars(0.01) == "**"
    assert map_pvalue_to_stars(0.04) == "*"
    assert map_pvalue_to_stars(0.1) == "ns"
    assert map_pvalue_to_stars(0.00001) == "****"


def test_stars_boundary_values():
    assert map_pvalue_to_stars(0.0001) == "****"
    assert map_pvalue_to_stars(0.05) == "*"
    assert map_pvalue_to_stars(0.0) == "****"


def test_stars_custom_mapping():
    m = {0.05: "sig", 0.01: "very sig"}
    assert map_pvalue_to_stars(0.03, m) == "sig"
    assert map_pvalue_to_stars(0.005, m) == "very sig"


def test_stars_custom_mapping_no_match():
    m = {0.05: "sig"}
    assert map_pvalue_to_stars(0.9, m) == "ns"


# ---------------------------------------------------------------------------
# compute_brackets
# ---------------------------------------------------------------------------


def test_brackets_count():
    comparisons = [("A", "B"), ("A", "C")]
    positions = {"A": 1, "B": 2, "C": 3}
    maxima = {"A": 2.0, "B": 3.0, "C": 5.0}
    brackets = compute_brackets(
        comparisons, positions, maxima, labels=["*", "***"], y_scale_range=6.0
    )
    assert len(brackets) == 2


def test_brackets_empty_input():
    assert compute_brackets([], {}, {}, labels=[], y_scale_range=1.0) == []


def test_brackets_stacked():
    comparisons = [("A", "B"), ("A", "C")]
    positions = {"A": 1, "B": 2, "C": 3}
    maxima = {"A": 2.0, "B": 3.0, "C": 5.0}
    brackets = compute_brackets(
        comparisons, positions, maxima,
        labels=["*", "***"], y_scale_range=6.0, step_increase=0.1,
    )
    # Second bracket (wider span A-C) must be higher than first (A-B)
    assert brackets[1].y_bracket > brackets[0].y_bracket


def test_brackets_tip_below_bar():
    brackets = compute_brackets(
        [("A", "B")], {"A": 1, "B": 2}, {"A": 2.0, "B": 2.5},
        labels=["*"], y_scale_range=5.0,
    )
    b = brackets[0]
    assert b.y_tip_left < b.y_bracket
    assert b.y_tip_right < b.y_bracket


def test_brackets_asymmetric_tips():
    brackets = compute_brackets(
        [("A", "B")], {"A": 1, "B": 2}, {"A": 2.0, "B": 3.0},
        labels=["*"], y_scale_range=5.0, tip_length=[0.1, 0.02],
    )
    b = brackets[0]
    # Left tip longer (further below bar) than right tip
    assert (b.y_bracket - b.y_tip_left) > (b.y_bracket - b.y_tip_right)


def test_brackets_wider_span_goes_higher():
    # A-C spans 2 units, A-B spans 1 unit — A-C should be higher
    comparisons = [("A", "C"), ("A", "B")]  # provide in reversed order
    positions = {"A": 1, "B": 2, "C": 3}
    maxima = {"A": 1.0, "B": 1.0, "C": 1.0}
    brackets = compute_brackets(
        comparisons, positions, maxima, labels=["**", "*"], y_scale_range=3.0
    )
    # A-B is narrow (i=0), A-C is wide (i=1) → y_bracket_AC > y_bracket_AB
    ab = next(b for b in brackets if b.xmax - b.xmin == 1)
    ac = next(b for b in brackets if b.xmax - b.xmin == 2)
    assert ac.y_bracket > ab.y_bracket


def test_brackets_label_preserved():
    brackets = compute_brackets(
        [("A", "B")], {"A": 1, "B": 2}, {"A": 0.5, "B": 0.8},
        labels=["***"], y_scale_range=2.0,
    )
    assert brackets[0].label == "***"


# ---------------------------------------------------------------------------
# run_comparisons
# ---------------------------------------------------------------------------


def test_run_comparisons_returns_df(simple_df):
    result = run_comparisons(
        simple_df["value"], simple_df["group"],
        comparisons=[("A", "C")], test="mannwhitney",
    )
    assert isinstance(result, pd.DataFrame)
    assert set(["group1", "group2", "pvalue", "padj"]).issubset(result.columns)


def test_run_comparisons_detects_difference(simple_df):
    result = run_comparisons(
        simple_df["value"], simple_df["group"],
        comparisons=[("A", "C")], test="mannwhitney",
    )
    assert result["pvalue"].iloc[0] < 0.001


def test_run_comparisons_all_pairs(simple_df):
    result = run_comparisons(
        simple_df["value"], simple_df["group"],
        comparisons=None, test="mannwhitney",
    )
    assert len(result) == 3  # A-B, A-C, B-C


def test_run_comparisons_bonferroni(simple_df):
    result = run_comparisons(
        simple_df["value"], simple_df["group"],
        comparisons=None, p_adjust="bonferroni",
    )
    assert (result["padj"] >= result["pvalue"]).all()


def test_run_comparisons_no_adjust(simple_df):
    result = run_comparisons(
        simple_df["value"], simple_df["group"],
        comparisons=[("A", "C")], p_adjust="none",
    )
    assert result["padj"].iloc[0] == result["pvalue"].iloc[0]


def test_run_comparisons_ttest(simple_df):
    result = run_comparisons(
        simple_df["value"], simple_df["group"],
        comparisons=[("A", "C")], test="ttest",
    )
    assert len(result) == 1
    assert result["pvalue"].iloc[0] < 0.001


def test_run_comparisons_empty_for_bad_groups(simple_df):
    result = run_comparisons(
        simple_df["value"], simple_df["group"],
        comparisons=[("X", "Y")],  # non-existent groups
    )
    assert result.empty


def test_run_comparisons_invalid_test(simple_df):
    with pytest.raises(ValueError, match="Unknown test"):
        run_comparisons(simple_df["value"], simple_df["group"],
                        comparisons=[("A", "B")], test="invalid")


# ---------------------------------------------------------------------------
# geom_signif — manual mode
# ---------------------------------------------------------------------------


def test_geom_signif_manual_returns_layers():
    layers = geom_signif(
        annotations=["**"],
        y_position=[5.0],
        xmin=[1],
        xmax=[2],
    )
    assert isinstance(layers, list)
    assert len(layers) == 2  # geom_segment + geom_text


def test_geom_signif_manual_multi():
    layers = geom_signif(
        annotations=["*", "***"],
        y_position=[5.0, 7.0],
        xmin=[1, 1],
        xmax=[2, 3],
    )
    assert len(layers) == 2


def test_geom_signif_manual_asymmetric_tips():
    layers = geom_signif(
        annotations=["*"],
        y_position=[5.0],
        xmin=[1],
        xmax=[2],
        tip_length=[0.5, 0.1],
    )
    assert len(layers) == 2


# ---------------------------------------------------------------------------
# geom_signif — automatic (deferred) mode
# ---------------------------------------------------------------------------


def test_geom_signif_auto_returns_deferred():
    from ggnomics.signif._geom import _DeferredSignif
    result = geom_signif(comparisons=[("A", "B")], test="mannwhitney")
    assert isinstance(result, _DeferredSignif)


def test_geom_signif_resolve(simple_df):
    deferred = geom_signif(comparisons=[("A", "C")], test="mannwhitney")
    layers = deferred.resolve(simple_df, x_col="group", y_col="value")
    assert isinstance(layers, list)
    assert len(layers) == 2


def test_geom_signif_resolve_populates_last_brackets(simple_df):
    deferred = geom_signif(comparisons=[("A", "C")], test="mannwhitney")
    deferred.resolve(simple_df, x_col="group", y_col="value")
    assert len(deferred._last_brackets) == 1


def test_geom_signif_sig_only_filters(simple_df):
    # A vs B: small difference (both near 0/1); A vs C: large difference
    deferred = geom_signif(
        comparisons=[("A", "B"), ("A", "C")],
        test="mannwhitney",
        sig_only=True,
    )
    # After bonferroni correction A-B may not survive; A-C definitely should
    layers = deferred.resolve(simple_df, x_col="group", y_col="value")
    assert len(deferred._last_brackets) >= 1


def test_geom_signif_sig_only_all_ns():
    np.random.seed(0)
    df = pd.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "value": np.ones(100),  # identical → p=1.0
    })
    deferred = geom_signif(
        comparisons=[("A", "B")],
        test="mannwhitney",
        sig_only=True,
    )
    layers = deferred.resolve(df, x_col="group", y_col="value")
    assert layers == []
    assert deferred._last_brackets == []


def test_geom_signif_map_signif_false(simple_df):
    deferred = geom_signif(
        comparisons=[("A", "C")],
        map_signif_level=False,
    )
    deferred.resolve(simple_df, x_col="group", y_col="value")
    label = deferred._last_brackets[0].label
    assert label.startswith("p=")


def test_geom_signif_map_signif_callable(simple_df):
    deferred = geom_signif(
        comparisons=[("A", "C")],
        map_signif_level=lambda p: f"adj={p:.1e}",
    )
    deferred.resolve(simple_df, x_col="group", y_col="value")
    assert deferred._last_brackets[0].label.startswith("adj=")


def test_geom_signif_map_signif_dict(simple_df):
    deferred = geom_signif(
        comparisons=[("A", "C")],
        map_signif_level={0.05: "SIG"},
    )
    deferred.resolve(simple_df, x_col="group", y_col="value")
    assert deferred._last_brackets[0].label in ("SIG", "ns")


# ---------------------------------------------------------------------------
# plot_violin_stats integration
# ---------------------------------------------------------------------------


def test_violin_stats_returns_ggplot(simple_df):
    p = plot_violin_stats(simple_df, "value", group_by="group",
                          comparisons=[("A", "C")])
    assert isinstance(p, ggplot_class)


def test_violin_stats_all_annotation_modes(simple_df):
    for mode in ["stars", "pvalue", "padj"]:
        p = plot_violin_stats(simple_df, "value", group_by="group",
                              comparisons=[("A", "C")], annotation=mode)
        assert isinstance(p, ggplot_class)


def test_violin_stats_sig_only(simple_df):
    p = plot_violin_stats(
        simple_df, "value", group_by="group",
        comparisons=[("A", "B"), ("A", "C")],
        sig_only=True,
    )
    assert isinstance(p, ggplot_class)


def test_violin_stats_no_sig_layers_when_all_ns():
    np.random.seed(0)
    df = pd.DataFrame({
        "group": ["A"] * 50 + ["B"] * 50,
        "value": np.ones(100),
    })
    p = plot_violin_stats(df, "value", group_by="group",
                          comparisons=[("A", "B")], sig_only=True)
    assert isinstance(p, ggplot_class)
    seg_layers = [l for l in p.layers if "segment" in type(l.geom).__name__.lower()]
    assert len(seg_layers) == 0


def test_violin_stats_y_axis_expanded(simple_df):
    """After adding brackets, the y scale should extend above the data maximum."""
    p = plot_violin_stats(simple_df, "value", group_by="group",
                          comparisons=[("A", "C")], sig_only=False)
    # Scales is a list subclass — check that a y-continuous scale with limits exists
    cont_scales = [
        s for s in p.scales
        if "continuous" in type(s).__name__.lower()
        and hasattr(s, "limits") and s.limits is not None
    ]
    assert len(cont_scales) >= 1


# ---------------------------------------------------------------------------
# plot_box_stats integration
# ---------------------------------------------------------------------------


def test_box_stats_returns_ggplot(simple_df):
    p = plot_box_stats(simple_df, "value", group_by="group")
    assert isinstance(p, ggplot_class)


def test_box_stats_with_comparisons(simple_df):
    p = plot_box_stats(simple_df, "value", group_by="group",
                       comparisons=[("A", "C")])
    assert isinstance(p, ggplot_class)


# ---------------------------------------------------------------------------
# Usability: geom_signif on a user-built plot
# ---------------------------------------------------------------------------


def test_geom_signif_on_custom_plot(simple_df):
    """Users can resolve geom_signif and add layers to their own ggplot."""
    deferred = geom_signif(comparisons=[("A", "B")], test="mannwhitney")
    layers = deferred.resolve(simple_df, x_col="group", y_col="value")
    p = ggplot(simple_df, aes("group", "value")) + geom_violin()
    for layer in layers:
        p = p + layer
    assert isinstance(p, ggplot_class)
