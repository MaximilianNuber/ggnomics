"""Color palette registry for ggnomics.

Palettes are registered as :class:`PaletteSpec` entries in a single
explicit registry (`_REGISTRY`). :func:`get_palette` and
:func:`resolve_palette` are the two public entry points; individual plot
modules should prefer :func:`resolve_palette` over reconstructing
``breaks``/``values`` from a raw dict.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

# ── Palette color data ──────────────────────────────────────────────────────

# Tableau 10 categorical palette
TABLEAU_10: Dict[str, str] = {
    "0": "#4E79A7",
    "1": "#F28E2B",
    "2": "#E15759",
    "3": "#76B7B2",
    "4": "#59A14F",
    "5": "#EDC948",
    "6": "#B07AA1",
    "7": "#FF9DA7",
    "8": "#9C755F",
    "9": "#BAB0AC",
}

# Tableau 20 categorical palette
TABLEAU_20: Dict[str, str] = {
    "0": "#4E79A7",
    "1": "#A0CBE8",
    "2": "#F28E2B",
    "3": "#FFBE7D",
    "4": "#59A14F",
    "5": "#8CD17D",
    "6": "#B6992D",
    "7": "#F1CE63",
    "8": "#499894",
    "9": "#86BCB6",
    "10": "#E15759",
    "11": "#FF9D9A",
    "12": "#79706E",
    "13": "#BAB0AC",
    "14": "#D37295",
    "15": "#FABFD2",
    "16": "#B07AA1",
    "17": "#D4A6C8",
    "18": "#9D7660",
    "19": "#D7B5A6",
}

# scater-like Bioconductor palette (20 distinct colors)
BIOC_COLORS: Dict[str, str] = {
    "0": "#E41A1C",
    "1": "#377EB8",
    "2": "#4DAF4A",
    "3": "#984EA3",
    "4": "#FF7F00",
    "5": "#FFFF33",
    "6": "#A65628",
    "7": "#F781BF",
    "8": "#999999",
    "9": "#66C2A5",
    "10": "#FC8D62",
    "11": "#8DA0CB",
    "12": "#E78AC3",
    "13": "#A6D854",
    "14": "#FFD92F",
    "15": "#E5C494",
    "16": "#B3B3B3",
    "17": "#8DD3C7",
    "18": "#FFFFB3",
    "19": "#BEBADA",
}

# ── IGV palettes ─────────────────────────────────────────────────────────
# Colors originate from the Integrative Genomics Viewer (IGV) chromosome
# color convention. Values here were taken via the `ggsci` R package
# (`ggsci::pal_igv()`), which republishes the IGV palette for ggplot2.
# Cite: Robinson et al., "Integrative Genomics Viewer", Nature
# Biotechnology 29, 24-26 (2011). Preserve this attribution if these values
# are redistributed.

IGV_DEFAULT = {
    "chr1": "#5050FF",
    "chr2": "#CE3D32",
    "chr3": "#749B58",
    "chr4": "#F0E685",
    "chr5": "#466983",
    "chr6": "#BA6338",
    "chr7": "#5DB1DD",
    "chr8": "#802268",
    "chr9": "#6BD76B",
    "chr10": "#D595A7",
    "chr11": "#924822",
    "chr12": "#837B8D",
    "chr13": "#C75127",
    "chr14": "#D58F5C",
    "chr15": "#7A65A5",
    "chr16": "#E4AF69",
    "chr17": "#3B1B53",
    "chr18": "#CDDEB7",
    "chr19": "#612A79",
    "chr20": "#AE1F63",
    "chr21": "#E7C76F",
    "chr22": "#5A655E",
    "chrX": "#CC9900",
    "chrY": "#99CC00",
    "chrUn": "#A9A9A9",
    "chr23": "#CC9900",
    "chr24": "#99CC00",
    "chr25": "#33CC00",
    "chr26": "#00CC33",
    "chr27": "#00CC99",
    "chr28": "#0099CC",
    "chr29": "#0A47FF",
    "chr30": "#4775FF",
    "chr31": "#FFC20A",
    "chr32": "#FFD147",
    "chr33": "#990033",
    "chr34": "#991A00",
    "chr35": "#996600",
    "chr36": "#809900",
    "chr37": "#339900",
    "chr38": "#00991A",
    "chr39": "#009966",
    "chr40": "#008099",
    "chr41": "#003399",
    "chr42": "#1A0099",
    "chr43": "#660099",
    "chr44": "#990080",
    "chr45": "#D60047",
    "chr46": "#FF1463",
    "chr47": "#00D68F",
    "chr48": "#14FFB1",
}

IGV_ALTERNATING = {
    "even": "#5773CC",  # Indigo
    "odd": "#FFB900",  # Selective Yellow
}


# ── Palette registry ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class PaletteSpec:
    """Immutable description of one registered palette.

    Attributes:
        name: Canonical registry name.
        aliases: Additional names that resolve to this palette.
        colors: Ordered sequence of hex colors, used for both by-index
            lookup (:func:`get_palette`) and gap-filling
            (:func:`resolve_palette`).
        kind: ``"categorical"`` or ``"continuous"``.
        source: Human-readable attribution/provenance.
        max_categories: Recommended maximum distinct category count before
            colors repeat.
        overflow: Policy applied when more categories are requested than
            ``max_categories`` — currently always ``"cycle"``.
    """

    name: str
    aliases: Tuple[str, ...]
    colors: Tuple[str, ...]
    kind: str
    source: str
    max_categories: int
    overflow: str = "cycle"


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("-", "")


_REGISTRY: Dict[str, PaletteSpec] = {}


def _register(spec: PaletteSpec) -> None:
    for key in (spec.name, *spec.aliases):
        normalized = _normalize_name(key)
        existing = _REGISTRY.get(normalized)
        if existing is not None and existing.name != spec.name:
            raise RuntimeError(f"Palette alias {key!r} collides between {existing.name!r} and {spec.name!r}.")
        _REGISTRY[normalized] = spec


_register(
    PaletteSpec(
        name="tableau10",
        aliases=("tableau", "tab10"),
        colors=tuple(TABLEAU_10.values()),
        kind="categorical",
        source="Tableau 10 categorical palette",
        max_categories=10,
    )
)
_register(
    PaletteSpec(
        name="tableau20",
        aliases=("tab20",),
        colors=tuple(TABLEAU_20.values()),
        kind="categorical",
        source="Tableau 20 categorical palette",
        max_categories=20,
    )
)
_register(
    PaletteSpec(
        name="bioc",
        aliases=("bioconductor", "scater"),
        colors=tuple(BIOC_COLORS.values()),
        kind="categorical",
        source="scater-like Bioconductor categorical palette",
        max_categories=20,
    )
)
_register(
    PaletteSpec(
        name="igv_default",
        aliases=("igv",),
        colors=tuple(IGV_DEFAULT.values()),
        kind="categorical",
        source=(
            "Integrative Genomics Viewer chromosome colors, via the ggsci R "
            "package (ggsci::pal_igv()). Robinson et al., Nature Biotechnology "
            "29, 24-26 (2011)."
        ),
        max_categories=len(IGV_DEFAULT),
    )
)
_register(
    PaletteSpec(
        name="igv_alternating",
        aliases=(),
        colors=tuple(IGV_ALTERNATING.values()),
        kind="categorical",
        source="Integrative Genomics Viewer alternating band colors.",
        max_categories=2,
    )
)

_CANONICAL_NAMES = sorted({spec.name for spec in _REGISTRY.values()})


def _resolve_spec(name: str) -> PaletteSpec:
    key = _normalize_name(name)
    spec = _REGISTRY.get(key)
    if spec is None:
        raise ValueError(f"Unknown palette {name!r}. Choose from: {_CANONICAL_NAMES}")
    return spec


# ── Public API ────────────────────────────────────────────────────────────


def get_palette(n: int, name: str = "tableau") -> Dict[int, str]:
    """Return a palette mapping integer indices to hex colors.

    Args:
        n: Number of categories. Must be ``>= 0``.
        name: Palette name or alias (e.g. ``"tableau"``, ``"tableau20"``,
            ``"bioc"``, ``"igv_default"``, ``"igv_alternating"``).

    Returns:
        ``{0: "#hex", 1: "#hex", ...}`` dict with ``n`` entries.

    Raises:
        ValueError: If ``n < 0`` or ``name`` is not recognised.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}.")

    spec = _resolve_spec(name)
    colors = spec.colors
    m = len(colors)
    if n > m:
        warnings.warn(
            f"get_palette: requested {n} colors from palette {spec.name!r}, "
            f"which has only {m} distinct colors; colors will repeat.",
            UserWarning,
            stacklevel=2,
        )
    return {i: colors[i % m] for i in range(n)}


_HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")


def _validate_color(category: object, color: object) -> None:
    if not isinstance(color, str) or not color:
        raise ValueError(f"Invalid color for category {category!r}: {color!r}. Expected a non-empty color string.")
    if color.startswith("#") and not _HEX_COLOR_RE.match(color):
        raise ValueError(f"Invalid hex color for category {category!r}: {color!r}.")


def _ordered_unique_categories(categories) -> List:
    """Category order: declared order for pandas categoricals, else first-appearance."""

    if isinstance(categories, pd.CategoricalDtype):
        return list(categories.categories)
    if isinstance(categories, pd.Categorical):
        return list(categories.categories)
    if isinstance(categories, pd.Series) and isinstance(categories.dtype, pd.CategoricalDtype):
        return list(categories.cat.categories)

    seen: List = []
    seen_set = set()
    for value in categories:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            continue
        if value not in seen_set:
            seen_set.add(value)
            seen.append(value)
    return seen


def resolve_palette(
    categories: Union[pd.Series, pd.Categorical, Sequence],
    palette: Optional[Dict] = None,
    *,
    default: str = "tableau",
) -> Dict:
    """Resolve a ``{category: color}`` mapping for the observed categories.

    Args:
        categories: Observed category values. A pandas ``Categorical`` (or a
            ``Series`` with categorical dtype) contributes its *declared*
            category order; anything else is ordered by first appearance
            (never sorted).
        palette: Optional explicit ``{category: color}`` mapping. Takes
            precedence for every category it covers. Never mutated.
        default: Palette name used to fill any category missing from
            ``palette`` (or to color everything, when ``palette is None``).

    Returns:
        A new dict mapping every observed category to a color, in
        deterministic category order.

    Raises:
        ValueError: If ``default`` names an unknown palette, or a color in
            ``palette`` is not a plausible color string.
    """
    ordered = _ordered_unique_categories(categories)

    user_palette = palette or {}
    resolved: Dict = {}
    missing: List = []
    for category in ordered:
        if category in user_palette:
            color = user_palette[category]
            _validate_color(category, color)
            resolved[category] = color
        else:
            missing.append(category)

    if missing:
        spec = _resolve_spec(default)
        colors = spec.colors
        m = len(colors)
        for offset, category in enumerate(missing):
            resolved[category] = colors[offset % m]
        if palette is not None:
            warnings.warn(
                f"resolve_palette: {missing} not found in the given palette; "
                f"filled from the {spec.name!r} default palette.",
                UserWarning,
                stacklevel=2,
            )

    return resolved


__all__ = [
    "PaletteSpec",
    "TABLEAU_10",
    "TABLEAU_20",
    "BIOC_COLORS",
    "IGV_DEFAULT",
    "IGV_ALTERNATING",
    "get_palette",
    "resolve_palette",
]
