from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def _mannwhitney(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    result = scipy_stats.mannwhitneyu(x, y, alternative="two-sided")
    return result.statistic, result.pvalue


def _wilcoxon(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    result = scipy_stats.wilcoxon(x, y)
    return result.statistic, result.pvalue


def _ttest(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    result = scipy_stats.ttest_ind(x, y, equal_var=False)  # Welch's t-test
    return result.statistic, result.pvalue


def _kruskal(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    result = scipy_stats.kruskal(x, y)
    return result.statistic, result.pvalue


TESTS: dict[str, callable] = {
    "mannwhitney": _mannwhitney,
    "wilcoxon": _wilcoxon,
    "ttest": _ttest,
    "kruskal": _kruskal,
}


def run_comparisons(
    data: pd.Series,
    groups: pd.Series,
    comparisons: list[tuple[str, str]] | None,
    test: str = "mannwhitney",
    p_adjust: str = "bonferroni",
) -> pd.DataFrame:
    """Run pairwise statistical tests for specified comparisons.

    Parameters
    ----------
    data :
        Numeric values.
    groups :
        Group labels, same index as data.
    comparisons :
        List of (g1, g2) pairs. None → all pairwise combinations.
    test :
        Key in TESTS dict.
    p_adjust :
        "bonferroni" | "fdr_bh" | "none"

    Returns
    -------
    DataFrame with columns: group1, group2, statistic, pvalue, padj, n1, n2
    """
    if test not in TESTS:
        raise ValueError(f"Unknown test {test!r}. Available: {list(TESTS.keys())}")

    test_fn = TESTS[test]
    unique_groups = groups.unique().tolist()

    if comparisons is None:
        n = len(unique_groups)
        if n > 10:
            warnings.warn(
                f"Auto-computing all pairwise comparisons for {n} groups "
                f"({n * (n - 1) // 2} comparisons). "
                "Consider passing explicit comparisons."
            )
        comparisons = list(combinations(unique_groups, 2))

    rows = []
    for g1, g2 in comparisons:
        x = data[groups == g1].dropna().values
        y = data[groups == g2].dropna().values
        if len(x) < 2 or len(y) < 2:
            continue
        try:
            stat, pval = test_fn(x, y)
        except Exception:
            continue
        rows.append(
            {
                "group1": g1,
                "group2": g2,
                "statistic": float(stat),
                "pvalue": float(pval),
                "n1": int(len(x)),
                "n2": int(len(y)),
            }
        )

    empty = pd.DataFrame(
        columns=["group1", "group2", "statistic", "pvalue", "padj", "n1", "n2"]
    )
    if not rows:
        return empty

    df = pd.DataFrame(rows)

    if p_adjust.lower() == "none" or len(df) <= 1:
        df["padj"] = df["pvalue"]
    else:
        try:
            from statsmodels.stats.multitest import multipletests

            method = "bonferroni" if p_adjust == "bonferroni" else "fdr_bh"
            _, padj_arr, _, _ = multipletests(df["pvalue"].values, method=method)
            df["padj"] = padj_arr
        except ImportError:
            df["padj"] = df["pvalue"]

    return df[["group1", "group2", "statistic", "pvalue", "padj", "n1", "n2"]]
