from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class BracketSpec:
    """All geometry needed to draw one significance bracket.

    xmin, xmax    : x positions of the two groups (categorical axis maps group1→1.0 etc.)
    y_bracket     : y position of the horizontal bar of the bracket.
    y_tip_left    : y where the left vertical tip starts (points down from bar).
    y_tip_right   : y where the right vertical tip starts.
    label         : text label ("*", "**", "***", "****", "ns", or custom).
    """

    xmin: float
    xmax: float
    y_bracket: float
    y_tip_left: float
    y_tip_right: float
    label: str


def map_pvalue_to_stars(
    p: float,
    mapping: dict | None = None,
) -> str:
    """Convert a p-value to a significance label.

    Default thresholds match ggsignif:
        p <= 0.0001 → "****"
        p <= 0.001  → "***"
        p <= 0.01   → "**"
        p <= 0.05   → "*"
        p >  0.05   → "ns"

    Parameters
    ----------
    p :
        p-value.
    mapping :
        Custom {threshold: label} dict. Thresholds are tested in ascending order;
        the first threshold for which p <= threshold is returned.
    """
    if mapping is None:
        if p <= 0.0001:
            return "****"
        if p <= 0.001:
            return "***"
        if p <= 0.01:
            return "**"
        if p <= 0.05:
            return "*"
        return "ns"

    for threshold in sorted(mapping.keys()):
        if p <= threshold:
            return mapping[threshold]
    return "ns"


def compute_brackets(
    comparisons: list[tuple[str, str]],
    group_positions: dict[str, float],
    group_maxima: dict[str, float],
    labels: list[str],
    y_scale_range: float,
    margin_top: float = 0.05,
    step_increase: float = 0.1,
    tip_length: float | list[float] = 0.03,
) -> list[BracketSpec]:
    """Compute bracket geometry for a list of pairwise comparisons.

    Replicates ggsignif's core algorithm:

        y_bracket_i = max(group_max_g1, group_max_g2)
                      + y_scale_range * margin_top
                      + y_scale_range * step_increase * i

    Comparisons are sorted by span width before position assignment so that
    wider brackets (spanning more groups) are placed higher.

    Parameters
    ----------
    comparisons : list of (group_a, group_b)
    group_positions : {group_name: x_position}
        Typically {"A": 1, "B": 2, "C": 3, ...}
    group_maxima : {group_name: max_y_value}
    labels : one label string per comparison (already converted from p-values)
    y_scale_range : data y range (max - min)
    margin_top : fraction of y_scale_range added above data max before first bracket
    step_increase : additional fraction of y_scale_range per stacked bracket
    tip_length : fraction of y_scale_range for vertical tips; [left, right] for asymmetric
    """
    if not comparisons:
        return []

    if isinstance(tip_length, (int, float)):
        tip_l = tip_r = float(tip_length)
    else:
        tip_l, tip_r = float(tip_length[0]), float(tip_length[1])

    def _span(comp: tuple[str, str]) -> float:
        return abs(
            group_positions.get(comp[1], 0.0) - group_positions.get(comp[0], 0.0)
        )

    # Sort narrowest → widest so wider brackets get higher i and thus higher y
    indexed = sorted(zip(comparisons, labels), key=lambda x: _span(x[0]))

    brackets: list[BracketSpec] = []
    for i, (comp, label) in enumerate(indexed):
        g1, g2 = comp
        x1 = group_positions.get(g1, 1.0)
        x2 = group_positions.get(g2, 2.0)

        y_data_max = max(
            group_maxima.get(g1, 0.0),
            group_maxima.get(g2, 0.0),
        )
        y_bracket = (
            y_data_max
            + y_scale_range * margin_top
            + y_scale_range * step_increase * i
        )
        y_tip_left = y_bracket - y_scale_range * tip_l
        y_tip_right = y_bracket - y_scale_range * tip_r

        brackets.append(
            BracketSpec(
                xmin=min(x1, x2),
                xmax=max(x1, x2),
                y_bracket=y_bracket,
                y_tip_left=y_tip_left,
                y_tip_right=y_tip_right,
                label=label,
            )
        )

    return brackets
