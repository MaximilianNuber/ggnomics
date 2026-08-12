"""Tests for the palette registry (ggnomics.palettes)."""

import pandas as pd
import pytest

from ggnomics.palettes import (
    BIOC_COLORS,
    IGV_ALTERNATING,
    IGV_DEFAULT,
    TABLEAU_10,
    TABLEAU_20,
    get_palette,
    resolve_palette,
)


# ---------------------------------------------------------------------------
# get_palette
# ---------------------------------------------------------------------------


def test_get_palette_default_matches_tableau10():
    p = get_palette(5)
    assert p == {i: list(TABLEAU_10.values())[i] for i in range(5)}


def test_get_palette_aliases_are_equivalent():
    assert get_palette(4, name="tableau") == get_palette(4, name="tableau10")
    assert get_palette(4, name="tab10") == get_palette(4, name="tableau10")
    assert get_palette(4, name="tab20") == get_palette(4, name="tableau20")
    assert get_palette(4, name="igv") == get_palette(4, name="igv_default")
    assert get_palette(4, name="bioconductor") == get_palette(4, name="bioc")


def test_get_palette_deterministic_ordering():
    p1 = get_palette(6, name="bioc")
    p2 = get_palette(6, name="bioc")
    assert p1 == p2
    assert list(p1.values()) == list(BIOC_COLORS.values())[:6]


def test_get_palette_cycles_and_warns_on_overflow():
    n = len(TABLEAU_10) + 3
    with pytest.warns(UserWarning, match="will repeat"):
        p = get_palette(n, name="tableau10")
    assert len(p) == n
    colors = list(TABLEAU_10.values())
    assert p[len(TABLEAU_10)] == colors[0]
    assert p[len(TABLEAU_10) + 1] == colors[1]


def test_get_palette_no_warning_when_within_range(recwarn):
    get_palette(3, name="tableau10")
    assert not any("will repeat" in str(w.message) for w in recwarn.list)


def test_get_palette_invalid_name_raises():
    with pytest.raises(ValueError, match="Unknown palette"):
        get_palette(3, name="not_a_real_palette")


def test_get_palette_negative_n_raises():
    with pytest.raises(ValueError, match="n must be >= 0"):
        get_palette(-1)


def test_get_palette_zero_n_returns_empty():
    assert get_palette(0) == {}


def test_get_palette_returns_int_keyed_dict():
    p = get_palette(3, name="tableau20")
    assert list(p.keys()) == [0, 1, 2]


def test_igv_alternating_two_colors():
    p = get_palette(2, name="igv_alternating")
    assert p == {0: IGV_ALTERNATING["even"], 1: IGV_ALTERNATING["odd"]}


# ---------------------------------------------------------------------------
# resolve_palette
# ---------------------------------------------------------------------------


def test_resolve_palette_full_user_mapping_no_warning(recwarn):
    palette = {"A": "#111111", "B": "#222222"}
    resolved = resolve_palette(["B", "A", "B"], palette=palette)
    assert resolved == {"B": "#222222", "A": "#111111"}
    assert not any("filled from" in str(w.message) for w in recwarn.list)


def test_resolve_palette_first_appearance_order_not_sorted():
    resolved = resolve_palette(["zebra", "apple", "zebra", "mango"])
    assert list(resolved.keys()) == ["zebra", "apple", "mango"]


def test_resolve_palette_categorical_dtype_respects_declared_order():
    cat = pd.Categorical(["b", "a", "b"], categories=["z", "a", "b"])
    resolved = resolve_palette(cat)
    assert list(resolved.keys()) == ["z", "a", "b"]


def test_resolve_palette_series_categorical_dtype():
    series = pd.Series(pd.Categorical(["y", "x"], categories=["x", "y"]))
    resolved = resolve_palette(series)
    assert list(resolved.keys()) == ["x", "y"]


def test_resolve_palette_partial_mapping_fills_and_warns():
    palette = {"A": "#111111"}
    with pytest.warns(UserWarning, match="filled from"):
        resolved = resolve_palette(["A", "B", "C"], palette=palette, default="bioc")
    assert resolved["A"] == "#111111"
    assert resolved["B"] == list(BIOC_COLORS.values())[0]
    assert resolved["C"] == list(BIOC_COLORS.values())[1]


def test_resolve_palette_no_palette_uses_default_entirely():
    resolved = resolve_palette(["A", "B"], default="tableau10")
    assert resolved == {"A": list(TABLEAU_10.values())[0], "B": list(TABLEAU_10.values())[1]}


def test_resolve_palette_user_mapping_takes_precedence():
    palette = {"A": "#ABCDEF"}
    resolved = resolve_palette(["A"], palette=palette, default="bioc")
    assert resolved["A"] == "#ABCDEF"


def test_resolve_palette_does_not_mutate_caller_dict():
    palette = {"A": "#111111"}
    original = dict(palette)
    resolve_palette(["A", "B"], palette=palette, default="tableau10")
    assert palette == original


def test_resolve_palette_invalid_default_raises():
    with pytest.raises(ValueError, match="Unknown palette"):
        resolve_palette(["A"], default="not_a_real_palette")


def test_resolve_palette_invalid_color_value_raises():
    with pytest.raises(ValueError, match="Invalid"):
        resolve_palette(["A"], palette={"A": "#GGGGGG"})


def test_resolve_palette_non_string_color_raises():
    with pytest.raises(ValueError, match="Invalid"):
        resolve_palette(["A"], palette={"A": 123})


def test_resolve_palette_ignores_nan():
    import numpy as np
    resolved = resolve_palette(["A", np.nan, "B", None])
    assert list(resolved.keys()) == ["A", "B"]


def test_resolve_palette_overflow_cycles_when_filling_many():
    categories = [f"cat{i}" for i in range(len(TABLEAU_10) + 2)]
    resolved = resolve_palette(categories, default="tableau10")
    colors = list(TABLEAU_10.values())
    assert resolved[categories[-2]] == colors[0]
    assert resolved[categories[-1]] == colors[1]
