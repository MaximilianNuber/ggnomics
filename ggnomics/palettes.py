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

# ── IGV palettes ──────────────────────────────────────────────────────────────
# Source: Integrative Genomics Viewer chromosome colors, via ggsci R package
# Robinson et al., Nature Biotechnology 29, 24–26 (2011)

IGV_DEFAULT = {
    "chr1":  "#5050FF", "chr2":  "#CE3D32", "chr3":  "#749B58",
    "chr4":  "#F0E685", "chr5":  "#466983", "chr6":  "#BA6338",
    "chr7":  "#5DB1DD", "chr8":  "#802268", "chr9":  "#6BD76B",
    "chr10": "#D595A7", "chr11": "#924822", "chr12": "#837B8D",
    "chr13": "#C75127", "chr14": "#D58F5C", "chr15": "#7A65A5",
    "chr16": "#E4AF69", "chr17": "#3B1B53", "chr18": "#CDDEB7",
    "chr19": "#612A79", "chr20": "#AE1F63", "chr21": "#E7C76F",
    "chr22": "#5A655E", "chrX":  "#CC9900", "chrY":  "#99CC00",
    "chrUn": "#A9A9A9", "chr23": "#CC9900", "chr24": "#99CC00",
    "chr25": "#33CC00", "chr26": "#00CC33", "chr27": "#00CC99",
    "chr28": "#0099CC", "chr29": "#0A47FF", "chr30": "#4775FF",
    "chr31": "#FFC20A", "chr32": "#FFD147", "chr33": "#990033",
    "chr34": "#991A00", "chr35": "#996600", "chr36": "#809900",
    "chr37": "#339900", "chr38": "#00991A", "chr39": "#009966",
    "chr40": "#008099", "chr41": "#003399", "chr42": "#1A0099",
    "chr43": "#660099", "chr44": "#990080", "chr45": "#D60047",
    "chr46": "#FF1463", "chr47": "#00D68F", "chr48": "#14FFB1",
}

IGV_ALTERNATING = {
    "even": "#5773CC",  # Indigo
    "odd":  "#FFB900",  # Selective Yellow
}

_NAMED_PALETTES = {
    "tableau10": TABLEAU_10,
    "tableau": TABLEAU_10,
    "tableau20": TABLEAU_20,
    "bioc": BIOC_COLORS,
    "igv_default": IGV_DEFAULT,
    "igv_alternating": IGV_ALTERNATING,
}

# Ordered list of colors for numeric indexing
_TABLEAU_10_LIST = list(TABLEAU_10.values())
_TABLEAU_20_LIST = list(TABLEAU_20.values())
_BIOC_LIST = list(BIOC_COLORS.values())
_IGV_DEFAULT_LIST = list(IGV_DEFAULT.values())
_IGV_ALTERNATING_LIST = list(IGV_ALTERNATING.values())


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
