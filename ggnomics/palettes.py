"""Color palettes for ggnomics."""

from __future__ import annotations

from typing import Dict

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
    "0":  "#4E79A7",
    "1":  "#A0CBE8",
    "2":  "#F28E2B",
    "3":  "#FFBE7D",
    "4":  "#59A14F",
    "5":  "#8CD17D",
    "6":  "#B6992D",
    "7":  "#F1CE63",
    "8":  "#499894",
    "9":  "#86BCB6",
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
    "0":  "#E41A1C",
    "1":  "#377EB8",
    "2":  "#4DAF4A",
    "3":  "#984EA3",
    "4":  "#FF7F00",
    "5":  "#FFFF33",
    "6":  "#A65628",
    "7":  "#F781BF",
    "8":  "#999999",
    "9":  "#66C2A5",
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

_NAMED_PALETTES = {
    "tableau10": TABLEAU_10,
    "tableau": TABLEAU_10,
    "tableau20": TABLEAU_20,
    "bioc": BIOC_COLORS,
}

# Ordered list of colors for numeric indexing
_TABLEAU_10_LIST = list(TABLEAU_10.values())
_TABLEAU_20_LIST = list(TABLEAU_20.values())
_BIOC_LIST = list(BIOC_COLORS.values())


def get_palette(n: int, name: str = "tableau") -> Dict[int, str]:
    """Return a palette mapping integer indices to hex colors.

    Args:
        n: Number of categories.
        name: Palette family name — ``"tableau"`` / ``"tableau10"``
              (10 colors, cycled), ``"tableau20"`` (20 colors, cycled),
              or ``"bioc"`` (20 scater-like colors, cycled).

    Returns:
        ``{0: "#hex", 1: "#hex", ...}`` dict with ``n`` entries.

    Raises:
        ValueError: If ``name`` is not recognised.
    """
    key = name.lower().replace("-", "")
    if key not in _NAMED_PALETTES:
        raise ValueError(
            f"Unknown palette '{name}'. Choose from: {list(_NAMED_PALETTES.keys())}"
        )

    base = _NAMED_PALETTES[key]
    color_list = list(base.values())
    m = len(color_list)
    return {i: color_list[i % m] for i in range(n)}
