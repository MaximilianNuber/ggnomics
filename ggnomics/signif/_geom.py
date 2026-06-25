from __future__ import annotations

import pandas as pd
from plotnine import aes, geom_segment, geom_text


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_label(p: float, map_signif_level) -> str:
    """Convert a p-value to a display label according to map_signif_level."""
    from ._brackets import map_pvalue_to_stars

    if map_signif_level is True:
        return map_pvalue_to_stars(p)
    if map_signif_level is False:
        return f"p={p:.3f}"
    if isinstance(map_signif_level, dict):
        return map_pvalue_to_stars(p, map_signif_level)
    if callable(map_signif_level):
        return map_signif_level(p)
    return map_pvalue_to_stars(p)


def _vjust_to_va(vjust) -> str:
    """Convert a numeric vjust (0=bottom, 0.5=center, 1=top) to a plotnine va string."""
    if isinstance(vjust, str):
        return vjust
    if vjust < 0.33:
        return "bottom"
    if vjust < 0.67:
        return "center"
    return "top"


def _brackets_to_layers(brackets: list, params) -> list:
    """Convert BracketSpec objects into geom_segment + geom_text layers.

    Each bracket → 3 segments (left tip, horizontal bar, right tip) + 1 label.
    Returns [] if brackets is empty.
    """
    if not brackets:
        return []

    seg_rows = []
    txt_rows = []

    for b in brackets:
        # left vertical tip
        seg_rows.append(
            {"x": b.xmin, "xend": b.xmin, "y": b.y_tip_left, "yend": b.y_bracket}
        )
        # horizontal bar
        seg_rows.append(
            {"x": b.xmin, "xend": b.xmax, "y": b.y_bracket, "yend": b.y_bracket}
        )
        # right vertical tip
        seg_rows.append(
            {"x": b.xmax, "xend": b.xmax, "y": b.y_bracket, "yend": b.y_tip_right}
        )
        # label centered above bar
        txt_rows.append(
            {
                "x": (b.xmin + b.xmax) / 2.0,
                "y": b.y_bracket,
                "label": b.label,
            }
        )

    seg_df = pd.DataFrame(seg_rows)
    txt_df = pd.DataFrame(txt_rows)

    va = _vjust_to_va(getattr(params, "vjust", 0.4))

    return [
        geom_segment(
            aes(x="x", xend="xend", y="y", yend="yend"),
            data=seg_df,
            color=params.color,
            linetype=params.linetype,
            size=params.linewidth,
            inherit_aes=False,
        ),
        geom_text(
            aes(x="x", y="y", label="label"),
            data=txt_df,
            size=params.textsize,
            va=va,
            color=params.color,
            inherit_aes=False,
        ),
    ]


# ---------------------------------------------------------------------------
# Deferred layer (auto mode)
# ---------------------------------------------------------------------------


class _DeferredSignif:
    """Placeholder returned by geom_signif in automatic mode.

    Calling .resolve(df, x_col, y_col) runs stats, computes geometry,
    and returns concrete plotnine layers.  _last_brackets is populated
    as a side-effect so callers can read bracket y positions for axis expansion.
    """

    def __init__(
        self,
        comparisons,
        test,
        map_signif_level,
        p_adjust,
        margin_top,
        step_increase,
        tip_length,
        sig_only,
        textsize,
        vjust,
        color,
        linetype,
        linewidth,
        bracket_nudge_y,
        orientation,
    ):
        self.comparisons = comparisons
        self.test = test
        self.map_signif_level = map_signif_level
        self.p_adjust = p_adjust
        self.margin_top = margin_top
        self.step_increase = step_increase
        self.tip_length = tip_length
        self.sig_only = sig_only
        self.textsize = textsize
        self.vjust = vjust
        self.color = color
        self.linetype = linetype
        self.linewidth = linewidth
        self.bracket_nudge_y = bracket_nudge_y
        self.orientation = orientation
        self._last_brackets: list = []

    def resolve(self, df: pd.DataFrame, x_col: str, y_col: str) -> list:
        """Run stats on df and return concrete plotnine layers.

        Parameters
        ----------
        df :
            Flat DataFrame already assembled for plotting.
        x_col :
            Column name for the group variable.
        y_col :
            Column name for the numeric value.
        """
        from ._brackets import compute_brackets
        from ._stats import run_comparisons

        # Group order from categorical dtype (preserves plot order)
        cat = df[x_col].astype("category")
        group_order = list(cat.cat.categories)
        group_positions = {g: i + 1 for i, g in enumerate(group_order)}

        group_maxima = df.groupby(x_col, observed=True)[y_col].max().to_dict()

        stats_df = run_comparisons(
            df[y_col], df[x_col], self.comparisons, self.test, self.p_adjust
        )

        if stats_df.empty:
            self._last_brackets = []
            return []

        if self.sig_only:
            stats_df = stats_df[stats_df["padj"] < 0.05].reset_index(drop=True)

        if stats_df.empty:
            self._last_brackets = []
            return []

        labels = [_resolve_label(p, self.map_signif_level) for p in stats_df["padj"]]
        comparisons_filtered = list(zip(stats_df["group1"], stats_df["group2"]))

        y_range = float(df[y_col].max() - df[y_col].min())
        if y_range == 0.0:
            y_range = 1.0

        brackets = compute_brackets(
            comparisons=comparisons_filtered,
            group_positions=group_positions,
            group_maxima=group_maxima,
            labels=labels,
            y_scale_range=y_range,
            margin_top=self.margin_top,
            step_increase=self.step_increase,
            tip_length=self.tip_length,
        )

        self._last_brackets = brackets
        return _brackets_to_layers(brackets, self)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def geom_signif(
    comparisons=None,
    # manual mode
    annotations=None,
    y_position=None,
    xmin=None,
    xmax=None,
    # auto stats
    test="mannwhitney",
    test_args=None,
    map_signif_level=True,
    p_adjust="bonferroni",
    # appearance
    margin_top=0.05,
    step_increase=0.1,
    tip_length=0.03,
    bracket_nudge_y=0.0,
    textsize=15,
    vjust=-10,
    color="black",
    linetype="solid",
    linewidth=0.5,
    # filter
    sig_only=False,
    orientation="x",
):
    """Add significance brackets to a plotnine violin/box/bar plot.

    Returns a list of plotnine layers (manual mode) or a _DeferredSignif
    object (automatic mode) that can be resolved with .resolve(df, x_col, y_col).

    MANUAL MODE
    -----------
    When annotations + y_position + xmin + xmax are all provided, draws
    brackets at the exact positions given without running any statistics.

        geom_signif(
            annotations=["p = 0.003"],
            y_position=[8.5],
            xmin=[1], xmax=[3],
        )

    AUTOMATIC MODE
    --------------
    When only comparisons are given, returns a _DeferredSignif that runs
    statistical tests at resolve time.

        deferred = geom_signif(comparisons=[("A", "B"), ("A", "C")])
        layers = deferred.resolve(df, x_col="group", y_col="value")
        for layer in layers:
            p = p + layer

    Parameters
    ----------
    comparisons : list of (group_a, group_b)
    annotations : list of str — pre-computed labels (manual mode)
    y_position  : list of float — bracket bar y positions (manual mode)
    xmin, xmax  : list of float — bracket x spans, numeric (manual mode)
    test : "mannwhitney" | "wilcoxon" | "ttest" | "kruskal"
    map_signif_level : True | False | dict | callable
        True  → default star mapping
        False → "p={value:.3f}"
        dict  → {threshold: label}
        callable → f(p) → str
    p_adjust : "bonferroni" | "fdr_bh" | "none"
    margin_top : fraction of y range above data max before first bracket
    step_increase : fraction of y range added per stacked bracket
    tip_length : fraction of y range for vertical tips; [left, right] for asymmetric
    sig_only : if True, only draw brackets with adjusted p < 0.05
    orientation : "x" (vertical plots) | "y" (horizontal plots, experimental)
    """
    # ------------------------------------------------------------------
    # Manual mode: all four manual parameters provided
    # ------------------------------------------------------------------
    if (
        annotations is not None
        and y_position is not None
        and xmin is not None
        and xmax is not None
    ):
        from ._brackets import BracketSpec

        if isinstance(tip_length, (int, float)):
            tip_l = tip_r = float(tip_length)
        else:
            tip_l, tip_r = float(tip_length[0]), float(tip_length[1])

        brackets = []
        for ann, yp, xl, xr in zip(annotations, y_position, xmin, xmax):
            yp = float(yp)
            brackets.append(
                BracketSpec(
                    xmin=float(xl),
                    xmax=float(xr),
                    y_bracket=yp,
                    y_tip_left=yp - tip_l,
                    y_tip_right=yp - tip_r,
                    label=ann,
                )
            )

        class _ManualParams:
            pass

        p = _ManualParams()
        p.color = color
        p.linetype = linetype
        p.linewidth = linewidth
        p.textsize = textsize
        p.vjust = vjust

        return _brackets_to_layers(brackets, p)

    # ------------------------------------------------------------------
    # Automatic (deferred) mode
    # ------------------------------------------------------------------
    return _DeferredSignif(
        comparisons=comparisons,
        test=test,
        map_signif_level=map_signif_level,
        p_adjust=p_adjust,
        margin_top=margin_top,
        step_increase=step_increase,
        tip_length=tip_length,
        sig_only=sig_only,
        textsize=textsize,
        vjust=vjust,
        color=color,
        linetype=linetype,
        linewidth=linewidth,
        bracket_nudge_y=bracket_nudge_y,
        orientation=orientation,
    )
