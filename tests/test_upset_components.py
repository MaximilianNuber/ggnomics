"""Component-level tests: queries, annotations, matrix, set sizes, stripes, themes.

These test the composable specification objects in isolation and, where
relevant, their effect once composed into a real plot via ``upset.upset()``.
"""

from __future__ import annotations

import pandas as pd
import pytest
from plotnine import aes, geom_col, geom_point, geom_segment, geom_text, theme

import ggnomics.upset as upset

# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------


def test_query_requires_exactly_one_of_set_intersect_group():
    with pytest.raises(ValueError, match="exactly one"):
        upset.upset_query(color="red")
    with pytest.raises(ValueError, match="exactly one"):
        upset.upset_query(set="A", intersect=["A"], color="red")
    with pytest.raises(ValueError, match="exactly one"):
        upset.upset_query(set="A", group="g", color="red")


def test_query_requires_at_least_one_aesthetic():
    with pytest.raises(ValueError, match="highlight aesthetic"):
        upset.upset_query(set="A")


def test_query_rejects_duplicate_intersection_members():
    with pytest.raises(ValueError, match="duplicate"):
        upset.upset_query(intersect=["A", "A"], color="red")


def test_query_only_components_is_preserved():
    query = upset.upset_query(set="A", color="red", only_components=["Intersection size"])
    assert query.only_components == ("Intersection size",)


def test_query_objects_are_immutable():
    query = upset.upset_query(set="A", color="red")
    with pytest.raises(Exception):
        query.set = "B"
    with pytest.raises(Exception):
        query.aesthetics["color"] = "blue"


def test_query_intersect_is_stored_as_tuple():
    query = upset.upset_query(intersect=["B", "A"], color="red")
    assert query.intersect == ("B", "A")


def test_query_unknown_set_rejected_during_composition(abc_example):
    prepared = upset.upset_data(abc_example, ["A", "B", "C"])
    with pytest.raises(KeyError):
        upset.compose_upset(prepared, queries=[upset.upset_query(set="Z", color="red")])


def test_query_unknown_intersection_rejected_during_composition(abc_example):
    prepared = upset.upset_data(abc_example, ["A", "B", "C"])
    with pytest.raises(KeyError):
        upset.compose_upset(
            prepared, queries=[upset.upset_query(intersect=["Z"], color="red")]
        )


def test_group_query_raises_not_implemented_rather_than_silently_ignored(abc_example):
    """ggnomics.upset does not implement group-based highlighting: no
    component reads UpSetQuery.group. This is a documented, intentional
    deviation from ComplexUpset — raise explicitly rather than accept and
    silently do nothing."""
    prepared = upset.upset_data(abc_example, ["A", "B", "C"])
    with pytest.raises(NotImplementedError):
        upset.compose_upset(
            prepared, queries=[upset.upset_query(group="somegroup", color="red")]
        )


def test_set_query_highlights_matrix_and_set_size(abc_example, tmp_path):
    prepared = upset.upset_data(abc_example, ["A", "B", "C"], intersections="all")
    composition = upset.compose_upset(
        prepared, queries=[upset.upset_query(set="A", color="red")]
    )
    output = tmp_path / "set_query.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_intersection_query_highlights_selected_intersection(abc_example, tmp_path):
    prepared = upset.upset_data(abc_example, ["A", "B", "C"], intersections="all")
    composition = upset.compose_upset(
        prepared, queries=[upset.upset_query(intersect=["A", "B"], color="red")]
    )
    output = tmp_path / "intersect_query.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_query_only_components_limits_highlighting(abc_example, tmp_path):
    prepared = upset.upset_data(abc_example, ["A", "B", "C"], intersections="all")
    query = upset.upset_query(
        intersect=["A", "B"], color="red", only_components=["intersections_matrix"]
    )
    # Not applied to the "Intersection size" annotation panel.
    assert not upset.plot._query_applies(query, "Intersection size")
    assert upset.plot._query_applies(query, "intersections_matrix")
    composition = upset.compose_upset(prepared, queries=[query])
    output = tmp_path / "only_components.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.exists()


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------


def test_intersection_size_default_construction():
    annotation = upset.intersection_size()
    assert annotation.kind == "intersection_size"
    assert annotation.mode == "exclusive_intersection"
    assert annotation.options["counts"] is True


def test_intersection_size_counts_toggle():
    on = upset.intersection_size(counts=True)
    off = upset.intersection_size(counts=False)
    assert on.options["counts"] is True
    assert off.options["counts"] is False


def test_intersection_size_invalid_threshold_raises():
    with pytest.raises(ValueError):
        upset.intersection_size(bar_number_threshold=1.5)
    with pytest.raises(ValueError):
        upset.intersection_size(bar_number_threshold=-0.1)


def test_intersection_size_invalid_width_raises():
    with pytest.raises(ValueError):
        upset.intersection_size(width=0)
    with pytest.raises(ValueError):
        upset.intersection_size(width=-1)


def test_intersection_size_text_color_validation():
    with pytest.raises(ValueError, match="text_colors"):
        upset.intersection_size(text_colors={"on_background": "black"})


def test_intersection_ratio_denominator_mode_normalized():
    annotation = upset.intersection_ratio(denominator_mode="union")
    assert annotation.options["denominator_mode"] == "inclusive_union"


def test_upset_annotate_requires_nonempty_y():
    with pytest.raises(ValueError):
        upset.upset_annotate("", geom_col())


def test_upset_annotate_requires_at_least_one_geom():
    with pytest.raises(ValueError):
        upset.upset_annotate("score", [])


def test_upset_annotate_custom_text_mapping(abc_example_with_covariates, tmp_path):
    annotation = upset.upset_annotate("score", geom_col())
    assert annotation.default_y == "score"
    prepared = upset.upset_data(abc_example_with_covariates, ["A", "B", "C"])
    composition = upset.upset(prepared, annotations={"Score": annotation})
    output = tmp_path / "custom_annotation.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_annotation_mode_can_be_changed_without_mutating_original():
    annotation = upset.intersection_size(mode="distinct")
    changed = annotation + upset.upset_mode("union")
    assert annotation.mode == "exclusive_intersection"
    assert changed.mode == "inclusive_union"
    assert changed is not annotation


def test_annotation_plus_geom_returns_new_object_without_mutating_original():
    annotation = upset.upset_annotate("score", geom_col())
    extended = annotation + theme(legend_position="none")
    assert extended is not annotation
    assert extended.plot is not annotation.plot


def test_intersection_size_and_ratio_render_in_a_real_plot(abc_example, tmp_path):
    composition = upset.upset(
        abc_example,
        ["A", "B", "C"],
        base_annotations={
            "Intersection size": upset.intersection_size(),
            "Intersection ratio": upset.intersection_ratio(),
        },
    )
    output = tmp_path / "size_ratio.png"
    composition.save(output, width=8, height=8, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_duplicate_annotation_names_raise(abc_example):
    with pytest.raises(ValueError, match="duplicate"):
        upset.upset(
            abc_example,
            ["A", "B", "C"],
            base_annotations={"Intersection size": upset.intersection_size()},
            annotations={"Intersection size": upset.intersection_size()},
        )


def test_no_annotation_panels_raises(abc_example):
    with pytest.raises(ValueError, match="at least one"):
        upset.upset(abc_example, ["A", "B", "C"], base_annotations={})


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------


def test_intersection_matrix_default_construction():
    spec = upset.intersection_matrix()
    assert spec.outline_color == {"active": "black", "inactive": "#B3B3B3"}


def test_intersection_matrix_missing_outline_keys_raises():
    with pytest.raises(ValueError):
        upset.intersection_matrix(outline_color={"active": "black"})


def test_intersection_matrix_custom_point_geom_is_used(abc_example, tmp_path):
    spec = upset.intersection_matrix(geom=geom_point(shape="x", size=4))
    composition = upset.upset(abc_example, ["A", "B", "C"], matrix=spec)
    output = tmp_path / "custom_point.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_intersection_matrix_custom_segment_geom_is_used(abc_example, tmp_path):
    spec = upset.intersection_matrix(segment=geom_segment(linetype="dashed", size=1))
    composition = upset.upset(abc_example, ["A", "B", "C"], matrix=spec)
    output = tmp_path / "custom_segment.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_intersection_matrix_custom_geom_params_survive_composition(abc_example):
    spec = upset.intersection_matrix(geom=geom_point(shape="x", size=7))
    assert spec.geom.aes_params.get("shape") == "x"
    assert spec.geom.aes_params.get("size") == 7
    # Compose and confirm no error swallows the custom geom.
    upset.upset(abc_example, ["A", "B", "C"], matrix=spec)


# ---------------------------------------------------------------------------
# Set sizes
# ---------------------------------------------------------------------------


def test_upset_set_size_default_position_is_left():
    spec = upset.upset_set_size()
    assert spec.position == "left"


def test_upset_set_size_right_position():
    spec = upset.upset_set_size(position="right")
    assert spec.position == "right"


def test_upset_set_size_invalid_position_raises():
    with pytest.raises(ValueError):
        upset.upset_set_size(position="middle")


def test_upset_set_size_custom_mapping_and_geom(abc_example, tmp_path):
    spec = upset.upset_set_size(
        mapping=aes(fill="group"), geom=geom_col(width=0.4)
    )
    composition = upset.upset(abc_example, ["A", "B", "C"], set_sizes=spec)
    output = tmp_path / "custom_set_size.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_upset_set_size_filter_intersections_true_vs_false(abc_example):
    prepared = upset.upset_data(
        abc_example,
        ["A", "B", "C"],
        intersections=[["A"], ["B"]],
        keep_empty_groups=True,
    )
    filtered_spec = upset.upset_set_size(filter_intersections=True)
    unfiltered_spec = upset.upset_set_size(filter_intersections=False)
    filtered_plot = upset.plot._build_set_sizes(
        prepared, spec=filtered_spec, queries=(), themes=upset.upset_themes
    )
    unfiltered_plot = upset.plot._build_set_sizes(
        prepared, spec=unfiltered_spec, queries=(), themes=upset.upset_themes
    )
    # Filtered sizes only count rows whose intersection was kept; unfiltered
    # uses the full per-set membership totals (>= filtered, since "C" rows
    # dropped from the selection still count toward its raw total).
    filtered_c = filtered_plot.data.set_index("group").loc["C", "size"]
    unfiltered_c = unfiltered_plot.data.set_index("group").loc["C", "size"]
    assert filtered_c == 0
    assert unfiltered_c > 0


# ---------------------------------------------------------------------------
# Stripes
# ---------------------------------------------------------------------------


def test_upset_stripes_default_alternating_colors():
    spec = upset.upset_stripes()
    assert spec.colors == ("white", "#F2F2F2")


def test_upset_stripes_metadata_mapped_colors(abc_example, tmp_path):
    metadata = pd.DataFrame({"set": ["A", "B", "C"], "kind": ["rna", "rna", "protein"]})
    spec = upset.upset_stripes(
        mapping={"color": "kind"},
        colors={"rna": "#E8E8E8", "protein": "#D8E8F8"},
        data=metadata,
    )
    composition = upset.upset(abc_example, ["A", "B", "C"], stripes=spec)
    output = tmp_path / "metadata_stripes.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_upset_stripes_missing_set_column_raises():
    metadata = pd.DataFrame({"group": ["A", "B"], "kind": ["rna", "protein"]})
    with pytest.raises(KeyError):
        upset.upset_stripes(data=metadata)


def test_upset_stripes_empty_color_sequence_raises():
    with pytest.raises(ValueError):
        upset.upset_stripes(colors=())


def test_upset_stripes_custom_geom(abc_example, tmp_path):
    spec = upset.upset_stripes(geom=geom_segment(size=10, alpha=0.5))
    composition = upset.upset(abc_example, ["A", "B", "C"], stripes=spec)
    output = tmp_path / "custom_stripe_geom.png"
    composition.save(output, width=8, height=6, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_upset_stripes_caller_owned_metadata_not_mutated(abc_example):
    metadata = pd.DataFrame({"set": ["A", "B", "C"], "kind": ["rna", "rna", "protein"]})
    original = metadata.copy(deep=True)
    upset.upset_stripes(
        mapping={"color": "kind"},
        colors={"rna": "#E8E8E8", "protein": "#D8E8F8"},
        data=metadata,
    )
    upset.upset(
        abc_example,
        ["A", "B", "C"],
        stripes=upset.upset_stripes(
            mapping={"color": "kind"},
            colors={"rna": "#E8E8E8", "protein": "#D8E8F8"},
            data=metadata,
        ),
    )
    pd.testing.assert_frame_equal(metadata, original)


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------


def test_default_theme_collection_has_expected_components():
    assert set(upset.upset_themes) == {
        "intersections_matrix",
        "Intersection size",
        "overall_sizes",
        "default",
    }


def test_upset_default_themes_applies_globally():
    modified = upset.upset_default_themes(legend_position="none")
    for name, components in modified.items():
        assert components[-1] == theme(legend_position="none")
        assert components[:-1] == upset.upset_themes[name]


def test_upset_modify_themes_is_component_specific():
    modified = upset.upset_modify_themes(
        {"overall_sizes": theme(axis_text_x=None)}
    )
    assert modified["overall_sizes"][-1] == theme(axis_text_x=None)
    assert modified["intersections_matrix"] == upset.upset_themes["intersections_matrix"]


def test_upset_modify_themes_accepts_a_sequence_of_additions():
    modified = upset.upset_modify_themes(
        {"default": [theme(legend_position="none"), theme(axis_text_x=None)]}
    )
    assert len(modified["default"]) == len(upset.upset_themes["default"]) + 2


def test_upset_modify_themes_unknown_component_raises():
    with pytest.raises(KeyError):
        upset.upset_modify_themes({"not_a_real_component": theme()})


def test_upset_modify_themes_does_not_mutate_defaults():
    before = dict(upset.upset_themes)
    upset.upset_modify_themes({"default": theme(legend_position="none")})
    assert dict(upset.upset_themes) == before
