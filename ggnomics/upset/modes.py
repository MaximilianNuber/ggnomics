"""Intersection-mode normalization and scale transformations."""

from __future__ import annotations

from typing import Literal

import numpy as np
from mizani.breaks import breaks_log
from mizani.transforms import trans, trans_new
from plotnine.mapping import after_stat

from ._types import ModeSpec

Mode = Literal[
    "exclusive_intersection",
    "distinct",
    "inclusive_intersection",
    "intersect",
    "inclusive_union",
    "union",
    "exclusive_union",
]

_MODE_ALIASES = {
    "distinct": "exclusive_intersection",
    "exclusive_intersection": "exclusive_intersection",
    "intersect": "inclusive_intersection",
    "inclusive_intersection": "inclusive_intersection",
    "union": "inclusive_union",
    "inclusive_union": "inclusive_union",
    "exclusive_union": "exclusive_union",
}


def normalize_mode(mode: Mode | str) -> str:
    """Convert one public mode or alias to its canonical name."""

    try:
        return _MODE_ALIASES[str(mode)]
    except KeyError as error:
        accepted = ", ".join(sorted(_MODE_ALIASES))
        raise ValueError(f"mode must be one of {accepted}; got {mode!r}") from error


def get_size_mode(mode: Mode, *, suffix: str = "_size") -> str:
    """Return the size-column name corresponding to one mode."""

    if not isinstance(suffix, str):
        raise TypeError(f"suffix must be str; got {type(suffix).__name__}")
    return f"{normalize_mode(mode)}{suffix}"


def upset_mode(mode: Mode) -> ModeSpec:
    """Construct a mode specification that can be added to an annotation."""

    return ModeSpec(normalize_mode(mode))


def upset_text_percentage(
    *, digits: int = 0, sep: str = "", mode: Mode = "distinct"
):
    """Return an after-stat expression for an intersection-to-union percentage."""

    if digits < 0:
        raise ValueError(f"digits must be non-negative; got {digits}")
    numerator = get_size_mode(mode)
    denominator = get_size_mode("inclusive_union")
    expression = (
        f"round(100 * {numerator} / {denominator}, {digits})"
        f".astype(str) + {('%' if not sep else sep + '%')!r}"
    )
    return after_stat(expression)


def aes_percentage(
    relative_to: Literal["intersection", "group", "all"],
    *,
    digits: int = 0,
    sep: str = "",
):
    """Return an after-stat expression for percentage labels.

    Must be used on a layer computed by ``stat_count`` (e.g. ``geom_bar``)
    with a discrete ``fill`` mapping (for example up/down direction), since
    the expression references the ``count``, ``x``, and ``fill`` variables
    that stat produces. Plotnine's internal ``group`` variable is assigned
    per unique ``(x, fill)`` combination, not per ``fill`` level, so it
    cannot be used to total counts for a group across intersections; ``fill``
    is used instead. This is a documented deviation from ComplexUpset, which
    operates on a materialized R data.frame rather than a Plotnine computed
    layer.

    ``relative_to`` controls the denominator:

    - ``"intersection"``: the total count at the same ``x`` position (i.e.
      the height of the whole stacked bar for that intersection).
    - ``"group"``: the total count for the same ``fill`` level across all
      intersections.
    - ``"all"``: the grand total across the whole layer.
    """

    if relative_to not in {"intersection", "group", "all"}:
        raise ValueError(
            "relative_to must be 'intersection', 'group', or 'all'; "
            f"got {relative_to!r}"
        )
    if digits < 0:
        raise ValueError(f"digits must be non-negative; got {digits}")
    denominator = {
        "intersection": "count.groupby(x).transform('sum')",
        "group": "count.groupby(fill).transform('sum')",
        "all": "count.sum()",
    }[relative_to]
    expression = (
        f"round(100 * count / ({denominator}), {digits}).astype(str)"
        f" + {('%' if not sep else sep + '%')!r}"
    )
    return after_stat(expression)


def reverse_log_trans(*, base: float = 10) -> trans:
    """Construct a reversed logarithmic transformation for set-size axes."""

    if base <= 1:
        raise ValueError(f"base must be greater than 1; got {base}")

    def transform(values):
        array = np.asarray(values, dtype=float)
        return -np.log(array) / np.log(base)

    def inverse(values):
        return np.power(base, -np.asarray(values, dtype=float))

    return trans_new(
        f"reverselog-{base:g}",
        transform,
        inverse,
        breaks_func=breaks_log(base=base),
        domain=(1e-100, np.inf),
    )
