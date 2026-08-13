"""Venn-diagram helper tests.

The Venn implementation is an approximate geometric/raster representation
(circle outlines plus a rasterized region grid): region areas are visually
indicative, not exactly proportional to set sizes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from plotnine import coord_equal, ggplot, theme_void

import ggnomics.upset as upset


def _membership_df(n=200, seed=0, n_sets=3):
    rng = np.random.default_rng(seed)
    columns = {chr(ord("A") + i): rng.random(n) < 0.5 for i in range(n_sets)}
    return pd.DataFrame(columns)


# ---------------------------------------------------------------------------
# Set-count support
# ---------------------------------------------------------------------------


def test_two_sets_supported():
    layout = upset.compute_venn_layout(_membership_df(n_sets=2), sets=["A", "B"])
    assert layout.sets == ("A", "B")
    assert layout.circles["set"].nunique() == 2


def test_three_sets_supported():
    layout = upset.compute_venn_layout(_membership_df(n_sets=3), sets=["A", "B", "C"])
    assert layout.sets == ("A", "B", "C")
    assert layout.circles["set"].nunique() == 3


def test_four_sets_supported():
    layout = upset.compute_venn_layout(_membership_df(n_sets=4), sets=["A", "B", "C", "D"])
    assert layout.sets == ("A", "B", "C", "D")
    assert layout.circles["set"].nunique() == 4


def test_fewer_than_two_sets_raises():
    with pytest.raises(ValueError, match="two, three, or four"):
        upset.compute_venn_layout(_membership_df(n_sets=1), sets=["A"])


def test_more_than_four_sets_raises():
    with pytest.raises(ValueError, match="two, three, or four"):
        upset.compute_venn_layout(_membership_df(n_sets=5), sets=list("ABCDE"))


def test_inferred_boolean_set_columns():
    data = _membership_df(n_sets=2)
    data["not_boolean"] = range(len(data))
    layout = upset.compute_venn_layout(data)
    assert set(layout.sets) == {"A", "B"}


def test_explicitly_supplied_set_columns_override_inference():
    data = _membership_df(n_sets=3)
    layout = upset.compute_venn_layout(data, sets=["A", "B"])
    assert layout.sets == ("A", "B")


def test_missing_set_columns_raise_keyerror():
    data = _membership_df(n_sets=2)
    with pytest.raises(KeyError):
        upset.compute_venn_layout(data, sets=["A", "Z"])


def test_invalid_membership_values_raise_typeerror():
    data = pd.DataFrame({"A": ["yes", "no"], "B": [True, False]})
    with pytest.raises(TypeError):
        upset.compute_venn_layout(data, sets=["A", "B"])


def test_missing_membership_values_raise_valueerror():
    data = pd.DataFrame({"A": [True, None], "B": [True, False]})
    with pytest.raises(ValueError):
        upset.compute_venn_layout(data, sets=["A", "B"])


def test_invalid_radius_raises():
    with pytest.raises(ValueError):
        upset.compute_venn_layout(_membership_df(n_sets=2), sets=["A", "B"], radius=0)
    with pytest.raises(ValueError):
        upset.compute_venn_layout(_membership_df(n_sets=2), sets=["A", "B"], radius=-1)


def test_invalid_grid_size_raises():
    with pytest.raises(ValueError):
        upset.compute_venn_layout(
            _membership_df(n_sets=2), sets=["A", "B"], starting_grid_size=5
        )


def test_deterministic_circle_count():
    layout = upset.compute_venn_layout(
        _membership_df(n_sets=3), sets=["A", "B", "C"], starting_grid_size=60
    )
    # 240 boundary angles per circle, one circle per set.
    assert len(layout.circles) == 240 * 3


# ---------------------------------------------------------------------------
# Region counts vs direct tabulation
# ---------------------------------------------------------------------------


def test_region_counts_agree_with_direct_boolean_tabulation():
    data = _membership_df(n=300, n_sets=2, seed=3)
    layout = upset.compute_venn_layout(data, sets=["A", "B"], starting_grid_size=60)

    direct_counts = {
        "A&B": int((data["A"] & data["B"]).sum()),
        "A": int((data["A"] & ~data["B"]).sum()),
        "B": int((~data["A"] & data["B"]).sum()),
    }
    label_counts = layout.region_labels.set_index("region")["count"].to_dict()
    for region, expected in direct_counts.items():
        assert label_counts.get(region, 0) == expected


def test_set_labels_exist_exactly_once_per_set():
    layout = upset.compute_venn_layout(
        _membership_df(n_sets=3), sets=["A", "B", "C"], starting_grid_size=60
    )
    counts = layout.set_labels["set"].value_counts()
    assert counts.to_dict() == {"A": 1, "B": 1, "C": 1}


# ---------------------------------------------------------------------------
# arrange_venn wraps compute_venn_layout with ComplexUpset-compatible args
# ---------------------------------------------------------------------------


def test_arrange_venn_wraps_compute_venn_layout_with_complexupset_args():
    data = _membership_df(n_sets=2)
    layout = upset.arrange_venn(data, sets=["A", "B"], verbose=True, extract_sets=True)
    assert layout.sets == ("A", "B")
    assert layout.circles["set"].nunique() == 2


# ---------------------------------------------------------------------------
# Layer/geom rendering
# ---------------------------------------------------------------------------


@pytest.fixture
def venn_layout():
    data = _membership_df(n=200, n_sets=2, seed=1)
    return upset.compute_venn_layout(data, sets=["A", "B"], starting_grid_size=60)


def test_region_layer_renders(venn_layout, tmp_path):
    result = ggplot() + upset.geom_venn_region(venn_layout) + coord_equal() + theme_void()
    output = tmp_path / "regions.png"
    result.save(output, width=5, height=5, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_circle_layer_renders(venn_layout, tmp_path):
    result = ggplot() + upset.geom_venn_circle(venn_layout) + coord_equal() + theme_void()
    output = tmp_path / "circles.png"
    result.save(output, width=5, height=5, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_region_label_rendering(venn_layout, tmp_path):
    result = (
        ggplot()
        + upset.geom_venn_region(venn_layout)
        + upset.geom_venn_label_region(venn_layout)
        + coord_equal()
        + theme_void()
    )
    output = tmp_path / "region_labels.png"
    result.save(output, width=5, height=5, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_set_label_rendering(venn_layout, tmp_path):
    result = (
        ggplot()
        + upset.geom_venn_circle(venn_layout)
        + upset.geom_venn_label_set(venn_layout)
        + coord_equal()
        + theme_void()
    )
    output = tmp_path / "set_labels.png"
    result.save(output, width=5, height=5, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


def test_full_venn_composition_renders(venn_layout, tmp_path):
    result = (
        ggplot()
        + upset.geom_venn_region(venn_layout)
        + upset.geom_venn_circle(venn_layout)
        + upset.geom_venn_label_region(venn_layout)
        + upset.geom_venn_label_set(venn_layout)
        + upset.scale_fill_venn_mix(venn_layout, colors=["#4477AA", "#CC6677"])
        + coord_equal()
        + theme_void()
    )
    output = tmp_path / "full.png"
    result.save(output, width=5, height=5, dpi=60, verbose=False)
    assert output.stat().st_size > 1_000


# ---------------------------------------------------------------------------
# Color scales
# ---------------------------------------------------------------------------


def _mapped(scale, keys):
    scale.train(list(keys))
    return dict(zip(keys, scale.map(list(keys))))


def test_sequential_color_input(venn_layout):
    scale = upset.scale_fill_venn_mix(venn_layout, colors=["#111111", "#222222"])
    mapped = _mapped(scale, ["A", "B"])
    assert mapped["A"] != mapped["B"]


def test_mapping_color_input(venn_layout):
    scale = upset.scale_color_venn_mix(venn_layout, colors={"A": "#111111", "B": "#222222"})
    mapped = _mapped(scale, ["A", "B"])
    assert mapped["A"] != mapped["B"]


def test_insufficient_colors_raises(venn_layout):
    with pytest.raises(ValueError, match="at least one color per set"):
        upset.scale_fill_venn_mix(venn_layout, colors=["#111111"])


def test_highlighted_regions_use_active_inactive_colors(venn_layout):
    scale = upset.scale_fill_venn_mix(
        venn_layout,
        highlight=[["A", "B"]],
        active_color="orange",
        inactive_color="grey",
    )
    regions = list(dict.fromkeys(venn_layout.regions["region"]))
    mapped = _mapped(scale, regions)
    assert mapped["A&B"] == "orange"
    other_regions = [name for name in regions if name != "A&B"]
    assert all(mapped[name] == "grey" for name in other_regions)


def test_color_and_fill_scale_constructors_return_distinct_scale_types(venn_layout):
    color_scale = upset.scale_color_venn_mix(venn_layout)
    fill_scale = upset.scale_fill_venn_mix(venn_layout)
    assert type(color_scale).__name__ == "scale_color_manual"
    assert type(fill_scale).__name__ == "scale_fill_manual"


def test_caller_owned_data_not_mutated():
    data = _membership_df(n_sets=2)
    original = data.copy(deep=True)
    upset.compute_venn_layout(data, sets=["A", "B"], starting_grid_size=60)
    pd.testing.assert_frame_equal(data, original)
