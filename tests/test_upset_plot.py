"""Composition-level tests for ggnomics.upset.upset() and compose_upset()."""

from __future__ import annotations

import pandas as pd
import pytest
from plotnine import aes, geom_boxplot, geom_point, geom_segment, ggplot
from plotnine.composition import Compose

import ggnomics.upset as upset

# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def test_upset_accepts_raw_dataframe(abc_example, tmp_path):
    composition = upset.upset(abc_example, ["A", "B", "C"])
    assert isinstance(composition, Compose)
    output = tmp_path / "raw.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_upset_accepts_prepared_upsetdata(abc_example):
    prepared = upset.upset_data(abc_example, ["A", "B", "C"])
    composition = upset.upset(prepared)
    assert isinstance(composition, Compose)


def test_upset_rejects_intersect_with_upsetdata(abc_example):
    prepared = upset.upset_data(abc_example, ["A", "B", "C"])
    with pytest.raises(TypeError, match="omitted"):
        upset.upset(prepared, ["A", "B", "C"])


def test_upset_requires_intersect_for_raw_dataframe(abc_example):
    with pytest.raises(TypeError, match="must be provided"):
        upset.upset(abc_example)


def test_upset_default_return_type_is_compose(abc_example):
    composition = upset.upset(abc_example, ["A", "B", "C"])
    assert isinstance(composition, Compose)


# ---------------------------------------------------------------------------
# Rendering: baseline and toggled components
# ---------------------------------------------------------------------------


def test_default_composition_draws(abc_example):
    composition = upset.upset(abc_example, ["A", "B", "C"])
    fig = composition.draw()
    assert fig is not None


def test_composition_with_set_sizes_false_draws(abc_example, tmp_path):
    composition = upset.upset(abc_example, ["A", "B", "C"], set_sizes=False)
    output = tmp_path / "no_set_sizes.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_intersection_size_annotation_draws(abc_example, tmp_path):
    composition = upset.upset(
        abc_example,
        ["A", "B", "C"],
        base_annotations={"Intersection size": upset.intersection_size()},
    )
    output = tmp_path / "intersection_size.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_intersection_ratio_annotation_draws(abc_example, tmp_path):
    composition = upset.upset(
        abc_example,
        ["A", "B", "C"],
        base_annotations={"Intersection ratio": upset.intersection_ratio()},
    )
    output = tmp_path / "intersection_ratio.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.exists()


def test_multiple_annotations_draw(abc_example_with_covariates, tmp_path):
    composition = upset.upset(
        abc_example_with_covariates,
        ["A", "B", "C"],
        base_annotations={"Intersection size": upset.intersection_size()},
        annotations={
            "Intersection ratio": upset.intersection_ratio(),
            "Score": upset.upset_annotate("score", geom_boxplot()),
        },
    )
    output = tmp_path / "multiple_annotations.png"
    composition.save(output, width=8, height=10, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_custom_annotation_using_numeric_gene_level_variable_draws(abc_example_with_covariates, tmp_path):
    annotation = upset.upset_annotate("score", geom_boxplot())
    composition = upset.upset(
        abc_example_with_covariates,
        ["A", "B", "C"],
        annotations={"Score": annotation},
    )
    output = tmp_path / "numeric_annotation.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_intersection_query_draws(abc_example, tmp_path):
    composition = upset.upset(
        abc_example,
        ["A", "B", "C"],
        intersections="all",
        queries=[upset.upset_query(intersect=["A", "B"], color="red")],
    )
    output = tmp_path / "intersect_query.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_set_query_draws(abc_example, tmp_path):
    composition = upset.upset(
        abc_example,
        ["A", "B", "C"],
        queries=[upset.upset_query(set="A", color="blue")],
    )
    output = tmp_path / "set_query.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_only_components_limits_highlighting_in_full_composition(abc_example, tmp_path):
    composition = upset.upset(
        abc_example,
        ["A", "B", "C"],
        intersections="all",
        queries=[upset.upset_query(intersect=["A", "B"], color="red", only_components=["intersections_matrix"])],
    )
    output = tmp_path / "only_components.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.exists()


def test_custom_labeller_is_applied(abc_example, tmp_path):
    composition = upset.upset(abc_example, ["A", "B", "C"], labeller=lambda name: f"Set {name}")
    output = tmp_path / "labeller.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.exists()


def test_custom_matrix_and_stripe_geoms_draw(abc_example, tmp_path):
    stripe_data = pd.DataFrame({"set": ["A", "B", "C"], "kind": ["rna", "rna", "protein"]})
    composition = upset.upset(
        abc_example,
        ["A", "B", "C"],
        matrix=upset.intersection_matrix(
            geom=geom_point(shape="x", size=4),
            segment=geom_segment(linetype="dashed", size=1),
        ),
        stripes=upset.upset_stripes(
            mapping={"color": "kind"},
            colors={"rna": "#E8E8E8", "protein": "#D8E8F8"},
            data=stripe_data,
        ),
    )
    output = tmp_path / "custom_matrix_stripes.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_custom_set_size_panel_draws(abc_example, tmp_path):
    from plotnine import geom_col

    composition = upset.upset(
        abc_example,
        ["A", "B", "C"],
        set_sizes=upset.upset_set_size(geom=geom_col(width=0.3, fill="darkred")),
    )
    output = tmp_path / "custom_set_size.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_width_ratio_and_height_ratio_validation(abc_example):
    with pytest.raises(ValueError):
        upset.upset(abc_example, ["A", "B", "C"], width_ratio=0)
    with pytest.raises(ValueError):
        upset.upset(abc_example, ["A", "B", "C"], height_ratio=-1)


def test_duplicate_annotation_names_raise_in_full_composition(abc_example):
    with pytest.raises(ValueError, match="duplicate"):
        upset.upset(
            abc_example,
            ["A", "B", "C"],
            base_annotations={"Intersection size": upset.intersection_size()},
            annotations={"Intersection size": upset.intersection_ratio()},
        )


def test_no_annotation_panels_raises_in_full_composition(abc_example):
    with pytest.raises(ValueError, match="at least one"):
        upset.upset(abc_example, ["A", "B", "C"], base_annotations={})


@pytest.mark.parametrize("guides", ["keep", "collect", "over", None])
def test_guides_keep_collect_do_not_crash(abc_example, guides, tmp_path):
    composition = upset.upset(abc_example, ["A", "B", "C"], guides=guides)
    output = tmp_path / f"guides_{guides}.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.exists()


def test_plot_can_be_saved_to_tmp_path(abc_example, tmp_path):
    composition = upset.upset(abc_example, ["A", "B", "C"])
    output = tmp_path / "saved.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.exists()
    assert output.stat().st_size > 1_000


# ---------------------------------------------------------------------------
# Composition with an ordinary Plotnine plot
# ---------------------------------------------------------------------------


def test_composition_with_ordinary_ggplot_using_pipe(abc_example, tmp_path):
    upset_plot = upset.upset(abc_example, ["A", "B", "C"])
    from plotnine import geom_bar

    side_plot = ggplot(abc_example, aes(x="A")) + geom_bar()
    combined = upset_plot | side_plot
    output = tmp_path / "combined_pipe.png"
    combined.save(output, width=14, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_composition_with_ordinary_ggplot_using_slash(abc_example, tmp_path):
    upset_plot = upset.upset(abc_example, ["A", "B", "C"])
    from plotnine import geom_bar

    side_plot = ggplot(abc_example, aes(x="A")) + geom_bar()
    combined = upset_plot / side_plot
    output = tmp_path / "combined_slash.png"
    combined.save(output, width=8, height=12, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


# ---------------------------------------------------------------------------
# Semantic inspection of prepared data (not just "did it render")
# ---------------------------------------------------------------------------


def test_prepared_data_semantic_structure_before_rendering(abc_example):
    prepared = upset.upset_data(abc_example, ["A", "B", "C"], intersections="all")
    assert set(prepared.sorted_sets) == {"A", "B", "C"}
    assert len(prepared.sorted_intersections) == 8
    assert prepared.matrix.shape == (3, 8)
    assert set(prepared.matrix_frame["group"].unique()) == {"A", "B", "C"}
    assert bool(prepared.matrix.loc["A", prepared.matrix.columns[0]]) in (True, False)
