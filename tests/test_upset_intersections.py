"""compute_intersections -> select_intersections -> apply_intersection_selection.

Uses ``create_upset_abc_example()`` as the golden semantic fixture (see
ggnomics.upset.examples): 325 rows with known exact exclusive/inclusive
counts for every combination of A, B, C.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ggnomics.upset as upset


def _id_for(prepared, members):
    return next(
        identifier
        for identifier, value in prepared.intersection_members.items()
        if value == members
    )


# ---------------------------------------------------------------------------
# Exact semantic counts (golden fixture)
# ---------------------------------------------------------------------------


def test_exclusive_intersection_counts_match_known_values(abc_example):
    statistics = upset.compute_intersections(
        abc_example, intersect=["A", "B", "C"], intersections="all"
    )
    expected = {
        (): 2,
        ("A",): 50,
        ("B",): 50,
        ("C",): 200,
        ("A", "B"): 10,
        ("A", "C"): 6,
        ("B", "C"): 6,
        ("A", "B", "C"): 1,
    }
    by_members = {members: identifier for identifier, members in statistics.intersection_members.items()}
    observed = {
        members: int(statistics.sizes.loc[by_members[members], "exclusive_intersection"])
        for members in expected
    }
    assert observed == expected


def test_four_modes_for_target_a_b(abc_example):
    statistics = upset.compute_intersections(
        abc_example, intersect=["A", "B", "C"], intersections="all"
    )
    by_members = {members: identifier for identifier, members in statistics.intersection_members.items()}
    identifier = by_members[("A", "B")]
    assert statistics.sizes.loc[identifier].to_dict() == {
        "exclusive_intersection": 10,
        "inclusive_intersection": 11,
        "exclusive_union": 110,
        "inclusive_union": 123,
    }


def test_empty_intersection_count_is_2(abc_example):
    statistics = upset.compute_intersections(
        abc_example, intersect=["A", "B", "C"], intersections="all"
    )
    by_members = {members: identifier for identifier, members in statistics.intersection_members.items()}
    assert int(statistics.sizes.loc[by_members[()], "exclusive_intersection"]) == 2


def test_intersections_all_includes_every_combination(abc_example):
    statistics = upset.compute_intersections(
        abc_example, intersect=["A", "B", "C"], intersections="all"
    )
    observed = set(statistics.intersection_members.values())
    expected = {
        (), ("A",), ("B",), ("C",),
        ("A", "B"), ("A", "C"), ("B", "C"), ("A", "B", "C"),
    }
    assert observed == expected


def test_intersections_observed_excludes_absent_combinations():
    # Only A alone and B alone are ever observed; A&B, C, etc. never occur.
    data = pd.DataFrame(
        {
            "A": [True, True, False, False],
            "B": [False, False, True, True],
            "C": [False, False, False, False],
        }
    )
    statistics = upset.compute_intersections(
        data, intersect=["A", "B", "C"], intersections="observed"
    )
    observed = set(statistics.intersection_members.values())
    assert observed == {("A",), ("B",)}
    assert ("A", "B") not in observed
    assert ("C",) not in observed
    assert () not in observed


def test_explicit_intersection_sequences_preserve_canonical_set_order(abc_example):
    statistics = upset.compute_intersections(
        abc_example,
        intersect=["A", "B", "C"],
        intersections=[["B", "A"], ["C"]],
    )
    # Canonical order follows `intersect`, not the order given in the request.
    assert set(statistics.intersection_members.values()) == {("A", "B"), ("C",)}


def test_duplicate_requested_intersections_are_deduplicated_deterministically(abc_example):
    statistics = upset.compute_intersections(
        abc_example,
        intersect=["A", "B", "C"],
        intersections=[["A"], ["A"], ["B"]],
    )
    assert list(statistics.intersection_members.values()) == [("A",), ("B",)]


def test_invalid_set_name_in_intersections_raises_keyerror(abc_example):
    with pytest.raises(KeyError):
        upset.compute_intersections(
            abc_example, intersect=["A", "B", "C"], intersections=[["Z"]]
        )


def test_fewer_than_two_intersect_columns_raises_valueerror(abc_example):
    with pytest.raises(ValueError, match="at least two"):
        upset.compute_intersections(abc_example, intersect=["A"])


def test_duplicate_intersect_names_raise_valueerror(abc_example):
    with pytest.raises(ValueError, match="unique"):
        upset.compute_intersections(abc_example, intersect=["A", "A", "B"])


def test_missing_membership_columns_raise_keyerror(abc_example):
    with pytest.raises(KeyError):
        upset.compute_intersections(abc_example, intersect=["A", "B", "Nonexistent"])


def test_missing_membership_values_raise_valueerror():
    data = pd.DataFrame({"A": [True, None, False], "B": [True, False, True]})
    with pytest.raises(ValueError, match="missing memberships"):
        upset.compute_intersections(data, intersect=["A", "B"])


def test_boolean_membership_columns_accepted_without_warning(abc_example, recwarn):
    upset.compute_intersections(abc_example, intersect=["A", "B", "C"])
    assert len(recwarn) == 0


def test_zero_one_columns_converted_to_boolean_and_reported():
    data = pd.DataFrame({"A": [1, 0, 1, 0], "B": [0, 1, 1, 0]})
    with pytest.warns(UserWarning, match="Converted"):
        statistics = upset.compute_intersections(data, intersect=["A", "B"])
    assert statistics.converted_columns == ("A", "B")
    assert statistics.memberships["A"].dtype == bool


def test_zero_one_columns_can_convert_silently():
    data = pd.DataFrame({"A": [1, 0, 1, 0], "B": [0, 1, 1, 0]})
    statistics = upset.compute_intersections(
        data, intersect=["A", "B"], warn_when_converting=False
    )
    assert statistics.converted_columns == ("A", "B")


def test_invalid_integer_values_raise_typeerror():
    data = pd.DataFrame({"A": [1, 2, 0], "B": [0, 1, 1]})
    with pytest.raises(TypeError):
        upset.compute_intersections(data, intersect=["A", "B"])


def test_invalid_string_values_raise_typeerror():
    data = pd.DataFrame({"A": ["yes", "no", "yes"], "B": [True, False, True]})
    with pytest.raises(TypeError):
        upset.compute_intersections(data, intersect=["A", "B"])


def test_duplicate_dataframe_columns_raise_valueerror():
    data = pd.DataFrame([[True, False, True]], columns=["A", "A", "B"])
    with pytest.raises(ValueError, match="unique"):
        upset.compute_intersections(data, intersect=["A", "B"])


def test_compute_select_apply_does_not_mutate_caller_input(abc_example):
    original = abc_example.copy(deep=True)
    statistics = upset.compute_intersections(
        abc_example, intersect=["A", "B", "C"], intersections="all"
    )
    selection = upset.select_intersections(statistics, min_size=10, min_degree=1, sort_sets=False)
    prepared = upset.apply_intersection_selection(statistics, selection=selection)
    pd.testing.assert_frame_equal(abc_example, original)
    assert all(prepared.sizes["exclusive_intersection"] >= 10)
    assert all(prepared.statistics.degrees.loc[list(prepared.sorted_intersections)] >= 1)


def test_custom_index_remains_aligned():
    data = pd.DataFrame(
        {"A": [True, False, True], "B": [False, True, True]},
        index=["gene_x", "gene_y", "gene_z"],
    )
    statistics = upset.compute_intersections(data, intersect=["A", "B"])
    prepared = upset.apply_intersection_selection(
        statistics, selection=upset.select_intersections(statistics)
    )
    assert list(prepared.with_sizes.index) == list(
        data.index[data[["A", "B"]].any(axis=1)]
    )


def test_combination_growth_limit_raises_before_materializing():
    data = pd.DataFrame(
        {name: np.random.default_rng(0).integers(0, 2, 100).astype(bool) for name in "ABCDEFGHIJKLMNOPQRSTUVWXY"}
    )
    with pytest.raises(ValueError, match="max_combinations_datapoints_n"):
        upset.compute_intersections(
            data,
            intersect=list(data.columns),
            intersections="all",
            max_combinations_datapoints_n=1000,
        )


def test_empty_input_raises_informative_error():
    data = pd.DataFrame({"A": pd.Series([], dtype=bool), "B": pd.Series([], dtype=bool)})
    with pytest.raises(ValueError, match="No intersections remain"):
        upset.upset_data(data, ["A", "B"])


def test_size_columns_suffix_controls_output_column_names(abc_example):
    prepared = upset.upset_data(
        abc_example, ["A", "B", "C"], size_columns_suffix="_n"
    )
    assert "exclusive_intersection_n" in prepared.with_sizes.columns
    assert "exclusive_intersection_size" not in prepared.with_sizes.columns
    for canonical in (
        "exclusive_intersection",
        "inclusive_intersection",
        "exclusive_union",
        "inclusive_union",
    ):
        assert f"{canonical}_n" in prepared.with_sizes.columns


def test_size_columns_suffix_default_matches_get_size_mode(abc_example):
    prepared = upset.upset_data(abc_example, ["A", "B", "C"])
    assert "exclusive_intersection_size" in prepared.with_sizes.columns
    assert upset.get_size_mode("distinct") == "exclusive_intersection_size"


def test_generated_categorical_axes_contain_only_selected_intersections(abc_example):
    prepared = upset.upset_data(
        abc_example, ["A", "B", "C"], intersections="all", min_size=10
    )
    axis_values = set(prepared.matrix_frame["intersection"].cat.categories)
    assert axis_values == set(prepared.sorted_intersections)
    assert set(prepared.sizes.index.astype(str)) == set(prepared.sorted_intersections)


def test_apply_intersection_selection_rejects_mismatched_selection(abc_example):
    statistics_abc = upset.compute_intersections(abc_example, intersect=["A", "B", "C"])
    other_data = pd.DataFrame({"A": [True, False], "B": [False, True]})
    statistics_ab = upset.compute_intersections(other_data, intersect=["A", "B"])
    mismatched_selection = upset.select_intersections(statistics_ab)
    with pytest.raises(KeyError):
        upset.apply_intersection_selection(statistics_abc, selection=mismatched_selection)


# ---------------------------------------------------------------------------
# select_intersections
# ---------------------------------------------------------------------------


def test_select_min_size(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"], intersections="all")
    selection = upset.select_intersections(statistics, min_size=10)
    assert all(statistics.sizes.loc[list(selection.intersections), "exclusive_intersection"] >= 10)


def test_select_max_size(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"], intersections="all")
    selection = upset.select_intersections(statistics, max_size=10)
    assert all(statistics.sizes.loc[list(selection.intersections), "exclusive_intersection"] <= 10)


def test_select_min_degree(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"], intersections="all")
    selection = upset.select_intersections(statistics, min_degree=2)
    assert all(statistics.degrees.loc[list(selection.intersections)] >= 2)


def test_select_max_degree(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"], intersections="all")
    selection = upset.select_intersections(statistics, max_degree=1)
    assert all(statistics.degrees.loc[list(selection.intersections)] <= 1)


def test_select_n_intersections_keeps_largest(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"], intersections="all")
    selection = upset.select_intersections(statistics, n_intersections=3, sort_intersections=False)
    assert len(selection.intersections) == 3
    kept_sizes = set(statistics.sizes.loc[list(selection.intersections), "exclusive_intersection"])
    all_sizes = sorted(statistics.sizes["exclusive_intersection"], reverse=True)
    assert kept_sizes == set(all_sizes[:3])


def test_select_keep_empty_groups_retains_unused_sets(abc_example):
    # Restrict to intersections that never touch "C" to make it droppable.
    statistics_ab_only = upset.compute_intersections(
        abc_example, intersect=["A", "B", "C"], intersections=[["A"], ["B"]]
    )
    dropped = upset.select_intersections(statistics_ab_only, keep_empty_groups=False)
    kept = upset.select_intersections(statistics_ab_only, keep_empty_groups=True)
    assert "C" not in dropped.sets
    assert "C" in kept.sets
    assert "C" in dropped.dropped_sets


def test_select_sort_intersections_by_cardinality_descending(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"], intersections="all")
    selection = upset.select_intersections(
        statistics, sort_intersections="descending", sort_intersections_by=["cardinality"]
    )
    sizes = statistics.sizes.loc[list(selection.intersections), "exclusive_intersection"].tolist()
    assert sizes == sorted(sizes, reverse=True)


def test_select_sort_intersections_by_cardinality_ascending(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"], intersections="all")
    selection = upset.select_intersections(
        statistics, sort_intersections="ascending", sort_intersections_by=["cardinality"]
    )
    sizes = statistics.sizes.loc[list(selection.intersections), "exclusive_intersection"].tolist()
    assert sizes == sorted(sizes)


def test_select_sort_intersections_by_degree(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"], intersections="all")
    selection = upset.select_intersections(
        statistics, sort_intersections="descending", sort_intersections_by=["degree"]
    )
    degrees = statistics.degrees.loc[list(selection.intersections)].tolist()
    assert degrees == sorted(degrees, reverse=True)


def test_select_sort_intersections_by_ratio(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"], intersections="all")
    selection = upset.select_intersections(
        statistics,
        sort_intersections="descending",
        sort_intersections_by=["ratio"],
        sort_ratio_numerator="exclusive_intersection",
        sort_ratio_denominator="inclusive_union",
    )
    numerator = statistics.sizes.loc[list(selection.intersections), "exclusive_intersection"]
    denominator = statistics.sizes.loc[list(selection.intersections), "inclusive_union"].replace(0, np.nan)
    ratios = (numerator / denominator).tolist()
    assert ratios == sorted(ratios, reverse=True, key=lambda v: (v is not None and not np.isnan(v), v if not np.isnan(v) else -np.inf))


def test_select_sort_intersections_false_preserves_input_order(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"], intersections="all")
    selection = upset.select_intersections(statistics, sort_intersections=False)
    assert list(selection.intersections) == statistics.sizes.index.tolist()


def test_select_sort_sets_ascending_descending_false(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"], intersections="all")
    ascending = upset.select_intersections(statistics, sort_sets="ascending")
    descending = upset.select_intersections(statistics, sort_sets="descending")
    unsorted_ = upset.select_intersections(statistics, sort_sets=False)
    asc_sizes = statistics.set_sizes.loc[list(ascending.sets)].tolist()
    desc_sizes = statistics.set_sizes.loc[list(descending.sets)].tolist()
    assert asc_sizes == sorted(asc_sizes)
    assert desc_sizes == sorted(desc_sizes, reverse=True)
    assert unsorted_.sets == statistics.sets


def test_select_group_by_degree(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"], intersections="all")
    selection = upset.select_intersections(statistics, group_by="degree", sort_intersections=False)
    assert selection.group_by == "degree"


def test_select_group_by_sets_orders_by_set_rank(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"], intersections="all")
    selection = upset.select_intersections(
        statistics, group_by="sets", sort_intersections=False, sort_sets="descending"
    )
    # First set in `sets` order should lead the grouped intersections.
    lead_members = [statistics.intersection_members[i] for i in selection.intersections]
    first_set = selection.sets[0]
    leading_positions = [
        i for i, members in enumerate(lead_members) if first_set in members
    ]
    if leading_positions:
        assert leading_positions == list(range(min(leading_positions), min(leading_positions) + len(leading_positions)))


def test_select_dropped_set_reporting(abc_example):
    statistics = upset.compute_intersections(
        abc_example, intersect=["A", "B", "C"], intersections=[["A"], ["B"]]
    )
    selection = upset.select_intersections(statistics, keep_empty_groups=False)
    assert selection.dropped_sets == ("C",)


def test_report_dropped_sets_emits_message(abc_example):
    messages: list[str] = []
    statistics = upset.compute_intersections(
        abc_example, intersect=["A", "B", "C"], intersections=[["A"], ["B"]]
    )
    selection = upset.select_intersections(statistics, keep_empty_groups=False)
    result = upset.report_dropped_sets(selection, emit=messages.append)
    assert any("C" in message for message in messages)
    assert result is selection


def test_select_no_intersections_remaining_raises(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"], intersections="all")
    with pytest.raises(ValueError, match="No intersections remain"):
        upset.select_intersections(statistics, min_size=1_000_000)


def test_select_invalid_bounds_raise(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"])
    with pytest.raises(ValueError):
        upset.select_intersections(statistics, min_size=-1)
    with pytest.raises(ValueError):
        upset.select_intersections(statistics, min_size=10, max_size=5)
    with pytest.raises(ValueError):
        upset.select_intersections(statistics, min_degree=-1)
    with pytest.raises(ValueError):
        upset.select_intersections(statistics, min_degree=3, max_degree=1)
    with pytest.raises(ValueError):
        upset.select_intersections(statistics, n_intersections=0)


def test_select_invalid_sorting_values_raise(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"])
    with pytest.raises(ValueError):
        upset.select_intersections(statistics, sort_sets="sideways")
    with pytest.raises(ValueError):
        upset.select_intersections(statistics, sort_intersections="sideways")
    with pytest.raises(ValueError):
        upset.select_intersections(statistics, sort_intersections_by=["bogus"])
    with pytest.raises(ValueError):
        upset.select_intersections(statistics, group_by="bogus")


def test_select_invalid_ratio_modes_raise(abc_example):
    statistics = upset.compute_intersections(abc_example, intersect=["A", "B", "C"])
    with pytest.raises(ValueError):
        upset.select_intersections(statistics, sort_ratio_numerator="not_a_mode")
    with pytest.raises(ValueError):
        upset.select_intersections(statistics, sort_ratio_denominator="not_a_mode")
