"""Tests for ggnomics._compose helpers."""

import pytest
from plotnine import ggplot, aes, geom_point
from plotnine.data import mtcars
from plotnine.composition import Compose

from ggnomics._compose import (
    hstack,
    vstack,
    grid,
    annotate_composition,
    save_composition,
    HAS_LAYOUT,
    _PLOTNINE_VERSION,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def p1():
    return ggplot(mtcars) + geom_point(aes("wt", "mpg"))


@pytest.fixture(scope="module")
def p2():
    return ggplot(mtcars) + geom_point(aes("wt", "disp"))


@pytest.fixture(scope="module")
def p3():
    return ggplot(mtcars) + geom_point(aes("disp", "mpg"))


# ---------------------------------------------------------------------------
# Version detection
# ---------------------------------------------------------------------------


def test_version_tuple():
    assert isinstance(_PLOTNINE_VERSION, tuple)
    assert len(_PLOTNINE_VERSION) == 2
    major, minor = _PLOTNINE_VERSION
    assert major == 0
    assert minor >= 15, "plotnine >= 0.15 required for composition"


def test_has_layout_flag():
    """HAS_LAYOUT should be True on the installed plotnine (0.16)."""
    assert HAS_LAYOUT is True


# ---------------------------------------------------------------------------
# hstack
# ---------------------------------------------------------------------------


def test_hstack_two(p1, p2):
    result = hstack(p1, p2)
    assert isinstance(result, Compose)


def test_hstack_three(p1, p2, p3):
    result = hstack(p1, p2, p3)
    assert isinstance(result, Compose)


def test_hstack_spacer(p1, p2):
    result = hstack(p1, p2, spacer=True)
    assert isinstance(result, Compose)


def test_hstack_single(p1):
    """Single plot: hstack with one argument should still work."""
    result = hstack(p1)
    # A single ggplot might not be Compose, but hstack returns it unchanged
    # so just verify it doesn't crash
    assert result is not None


# ---------------------------------------------------------------------------
# vstack
# ---------------------------------------------------------------------------


def test_vstack_two(p1, p2):
    result = vstack(p1, p2)
    assert isinstance(result, Compose)


def test_vstack_three(p1, p2, p3):
    result = vstack(p1, p2, p3)
    assert isinstance(result, Compose)


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------


def test_grid_two_plots_ncol2(p1, p2):
    result = grid([p1, p2], ncol=2)
    assert isinstance(result, Compose)


def test_grid_three_plots_ncol2(p1, p2, p3):
    """Odd count: last row gets a spacer on 0.15 fallback; on 0.16 plot_layout handles it."""
    result = grid([p1, p2, p3], ncol=2)
    assert isinstance(result, Compose)


def test_grid_four_plots_ncol2(p1, p2, p3):
    result = grid([p1, p2, p3, p1], ncol=2)
    assert isinstance(result, Compose)


def test_grid_single_plot(p1):
    """Single-element grid returns the plot unchanged (no phantom panels)."""
    result = grid([p1], ncol=1)
    assert result is p1


def test_grid_ncol1(p1, p2):
    result = grid([p1, p2], ncol=1)
    assert isinstance(result, Compose)


def test_grid_ncol3(p1, p2, p3):
    result = grid([p1, p2, p3], ncol=3)
    assert isinstance(result, Compose)


def test_grid_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        grid([], ncol=2)


def test_grid_widths(p1, p2):
    result = grid([p1, p2], ncol=2, widths=[2, 1])
    assert isinstance(result, Compose)


def test_grid_heights(p1, p2):
    result = grid([p1, p2], ncol=1, heights=[1, 2])
    assert isinstance(result, Compose)


# ---------------------------------------------------------------------------
# annotate_composition
# ---------------------------------------------------------------------------


def test_annotate_title(p1, p2):
    comp = p1 | p2
    result = annotate_composition(comp, title="My Figure")
    assert isinstance(result, Compose)


def test_annotate_subtitle(p1, p2):
    comp = p1 | p2
    result = annotate_composition(comp, subtitle="Subtitle text")
    assert isinstance(result, Compose)


def test_annotate_caption(p1, p2):
    comp = p1 | p2
    result = annotate_composition(comp, caption="Source: mtcars")
    assert isinstance(result, Compose)


def test_annotate_all(p1, p2):
    comp = p1 | p2
    result = annotate_composition(comp, title="T", subtitle="S", caption="C")
    assert isinstance(result, Compose)


def test_annotate_nothing(p1, p2):
    """No annotation kwargs: returns composition unchanged."""
    comp = p1 | p2
    result = annotate_composition(comp)
    assert isinstance(result, Compose)


# ---------------------------------------------------------------------------
# save_composition (smoke test — just ensure the call doesn't raise)
# ---------------------------------------------------------------------------


def test_save_composition(p1, p2, tmp_path):
    comp = p1 | p2
    out = str(tmp_path / "test_composition.png")
    # Should not raise
    save_composition(comp, out, dpi=72)
