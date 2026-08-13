"""Statistical-comparison helper tests.

These helpers ask, for a chosen intersection-membership mode, whether a
covariate differs across intersections. They are only inferentially valid
when the *rows* supplied to the comparison represent appropriate independent
units (e.g. samples or patients) — per-gene or per-cell p-values from these
helpers must not be read as biological-replicate inference. See the
"Statistical-helper caution" section of vignettes/upset.qmd.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import ggnomics.upset as upset


@pytest.fixture
def stats_example():
    rng = np.random.default_rng(7)
    n = 150
    a = rng.random(n) < 0.5
    b = rng.random(n) < 0.5
    numeric = np.where(a & ~b, rng.normal(5, 1, n), rng.normal(0, 1, n))
    categorical = np.where(a, "x", "y")
    return pd.DataFrame(
        {
            "A": a,
            "B": b,
            "numeric_var": numeric,
            "categorical_var": categorical,
            "ignored_var": rng.normal(0, 1, n),
        }
    )


def test_default_numeric_path_uses_kruskal_wallis(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])
    result = upset.compute_intersection_tests(prepared, ignore=["categorical_var", "ignored_var"])
    row = result.loc[result["variable"] == "numeric_var"].iloc[0]
    assert row["test"] == "Kruskal-Wallis"
    assert math.isfinite(row["statistic"])
    assert 0 <= row["p_value"] <= 1


def test_default_categorical_path_uses_chi_square(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])
    result = upset.compute_intersection_tests(prepared, ignore=["numeric_var", "ignored_var"])
    row = result.loc[result["variable"] == "categorical_var"].iloc[0]
    assert row["test"] == "Pearson chi-square"
    assert math.isfinite(row["statistic"])
    assert 0 <= row["p_value"] <= 1


def test_per_variable_custom_test(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])
    calls = []

    def custom_test(*groups):
        calls.append(len(groups))
        return type("Result", (), {"statistic": 1.0, "pvalue": 0.5})()

    result = upset.compute_intersection_tests(
        prepared,
        tests={"numeric_var": custom_test},
        ignore=["categorical_var", "ignored_var"],
    )
    assert calls, "custom per-variable test must have been called"
    row = result.loc[result["variable"] == "numeric_var"].iloc[0]
    assert row["statistic"] == 1.0
    assert row["p_value"] == 0.5


def test_global_custom_test_applies_to_all_variables(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])

    def custom_test(*groups):
        return (2.0, 0.25)

    result = upset.compute_intersection_tests(
        prepared, test=custom_test, ignore=["categorical_var", "ignored_var"]
    )
    row = result.loc[result["variable"] == "numeric_var"].iloc[0]
    assert row["statistic"] == 2.0
    assert row["p_value"] == 0.25


def test_custom_result_with_statistic_and_pvalue_attributes(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])

    class Result:
        statistic = 3.5
        pvalue = 0.1

    result = upset.compute_intersection_tests(
        prepared, test=lambda *groups: Result(), ignore=["categorical_var", "ignored_var"]
    )
    row = result.loc[result["variable"] == "numeric_var"].iloc[0]
    assert row["statistic"] == 3.5
    assert row["p_value"] == 0.1


def test_custom_result_as_tuple(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])
    result = upset.compute_intersection_tests(
        prepared, test=lambda *groups: (4.0, 0.02), ignore=["categorical_var", "ignored_var"]
    )
    row = result.loc[result["variable"] == "numeric_var"].iloc[0]
    assert row["statistic"] == 4.0
    assert row["p_value"] == 0.02


def test_malformed_custom_result_raises_typeerror(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])
    with pytest.raises(TypeError):
        upset.compute_intersection_tests(
            prepared, test=lambda *groups: "not a result", ignore=["categorical_var", "ignored_var"]
        )


def test_ignored_variables_are_excluded(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])
    result = upset.compute_intersection_tests(
        prepared, ignore=["numeric_var", "categorical_var", "ignored_var"]
    )
    assert result.empty


def test_ignored_membership_columns_are_excluded_automatically(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])
    result = upset.compute_intersection_tests(prepared)
    assert "A" not in result["variable"].tolist()
    assert "B" not in result["variable"].tolist()


def test_ignored_mode_columns_are_excluded_by_default(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])
    result = upset.compute_intersection_tests(prepared, ignore_mode_columns=True)
    mode_like = [v for v in result["variable"] if v.startswith("in_") or v.endswith("_size")]
    assert mode_like == []


def test_min_group_size_drops_small_intersections():
    """A tiny, extreme-valued intersection group should be excluded from the
    test once min_group_size exceeds its size, changing the result — not
    silently included alongside the larger groups."""
    rng = np.random.default_rng(11)
    neither = pd.DataFrame({"A": False, "B": False, "numeric_var": rng.normal(0, 1, 40)})
    a_only = pd.DataFrame({"A": True, "B": False, "numeric_var": rng.normal(0, 1, 40)})
    b_only = pd.DataFrame({"A": False, "B": True, "numeric_var": rng.normal(0, 1, 40)})
    tiny_outlier = pd.DataFrame({"A": True, "B": True, "numeric_var": rng.normal(100, 1, 3)})
    data = pd.concat([neither, a_only, b_only, tiny_outlier], ignore_index=True)

    prepared = upset.upset_data(data, ["A", "B"])
    lenient = upset.compute_intersection_tests(prepared, min_group_size=1)
    strict = upset.compute_intersection_tests(prepared, min_group_size=10)

    lenient_row = lenient.loc[lenient["variable"] == "numeric_var"].iloc[0]
    strict_row = strict.loc[strict["variable"] == "numeric_var"].iloc[0]
    assert math.isfinite(lenient_row["statistic"])
    assert math.isfinite(strict_row["statistic"])
    # Excluding the extreme 3-row group must change the Kruskal-Wallis statistic.
    assert lenient_row["statistic"] != strict_row["statistic"]


def test_min_group_size_too_strict_raises(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])
    with pytest.raises(ValueError, match="at least two"):
        upset.compute_intersection_tests(prepared, min_group_size=1000)


def test_fewer_than_two_eligible_groups_raises(stats_example):
    tiny = stats_example.iloc[:3].copy()
    tiny["A"] = True
    tiny["B"] = True
    prepared = upset.upset_data(tiny, ["A", "B"])
    with pytest.raises(ValueError, match="at least two"):
        upset.compute_intersection_tests(prepared, min_group_size=1)


def test_non_exclusive_modes_emit_independence_warning(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])
    with pytest.warns(UserWarning, match="not independent"):
        upset.compute_intersection_tests(prepared, mode="union")


def test_exclusive_mode_does_not_warn(stats_example, recwarn):
    prepared = upset.upset_data(stats_example, ["A", "B"])
    upset.compute_intersection_tests(prepared, mode="distinct")
    assert not any("not independent" in str(w.message) for w in recwarn.list)


def test_fdr_correction_adds_column_and_preserves_order(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])
    raw = upset.compute_intersection_tests(prepared)
    adjusted = upset.adjust_intersection_tests(raw)
    assert "fdr" in adjusted.columns
    assert len(adjusted) == len(raw)
    valid = adjusted["fdr"].dropna()
    assert ((valid >= 0) & (valid <= 1)).all()


def test_na_pvalues_remain_na_through_fdr_correction(stats_example):
    raw = pd.DataFrame(
        {
            "variable": ["a", "b", "c"],
            "p_value": [0.01, np.nan, 0.5],
            "statistic": [1.0, np.nan, 2.0],
            "test": ["t"] * 3,
        }
    )
    adjusted = upset.adjust_intersection_tests(raw)
    assert pd.isna(adjusted.loc[adjusted["variable"] == "b", "fdr"]).all()
    assert not pd.isna(adjusted.loc[adjusted["variable"] == "a", "fdr"]).all()


def test_adjust_intersection_tests_missing_columns_raises_keyerror():
    incomplete = pd.DataFrame({"variable": ["a"], "p_value": [0.1]})
    with pytest.raises(KeyError):
        upset.adjust_intersection_tests(incomplete)


def test_compare_between_intersections_raw_dataframe(stats_example):
    result = upset.compare_between_intersections(stats_example, ["A", "B"])
    assert {"variable", "p_value", "statistic", "test"}.issubset(result.columns)


def test_compare_between_intersections_prepared_upsetdata(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])
    result = upset.compare_between_intersections(prepared)
    assert {"variable", "p_value", "statistic", "test"}.issubset(result.columns)


def test_compare_between_intersections_rejects_intersect_with_upsetdata(stats_example):
    prepared = upset.upset_data(stats_example, ["A", "B"])
    with pytest.raises(TypeError, match="omitted"):
        upset.compare_between_intersections(prepared, ["A", "B"])


def test_upset_test_orders_by_ascending_fdr(stats_example):
    result = upset.upset_test(stats_example, ["A", "B"])
    fdr = result["fdr"].dropna().tolist()
    assert fdr == sorted(fdr)


# ---------------------------------------------------------------------------
# Optional dependency behavior
# ---------------------------------------------------------------------------


def test_missing_scipy_yields_informative_import_error(stats_example, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.stats", None)
    prepared = upset.upset_data(stats_example, ["A", "B"])
    with pytest.raises(ImportError, match=r'pip install "ggnomics\[upset\]"'):
        upset.compute_intersection_tests(prepared, ignore=["categorical_var", "ignored_var"])


def test_missing_statsmodels_yields_informative_import_error(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "statsmodels", None)
    monkeypatch.setitem(sys.modules, "statsmodels.stats.multitest", None)
    raw = pd.DataFrame(
        {
            "variable": ["a"],
            "p_value": [0.01],
            "statistic": [1.0],
            "test": ["t"],
        }
    )
    with pytest.raises(ImportError, match=r'pip install "ggnomics\[upset\]"'):
        upset.adjust_intersection_tests(raw)
