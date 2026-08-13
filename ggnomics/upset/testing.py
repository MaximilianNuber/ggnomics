"""Between-intersection statistical comparisons."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from ._types import UpSetData
from .intersections import upset_data
from .modes import Mode, normalize_mode

_OPTIONAL_DEPENDENCY_MESSAGE = (
    "{module} is required for ggnomics.upset statistical helpers. "
    'Install with: pip install "ggnomics[upset]"'
)


def _scipy_stats():
    try:
        from scipy import stats
    except ImportError as exc:
        raise ImportError(
            _OPTIONAL_DEPENDENCY_MESSAGE.format(module="scipy")
        ) from exc
    return stats


def _multipletests():
    try:
        from statsmodels.stats.multitest import multipletests
    except ImportError as exc:
        raise ImportError(
            _OPTIONAL_DEPENDENCY_MESSAGE.format(module="statsmodels")
        ) from exc
    return multipletests


def _extract_test_result(result: Any) -> tuple[float, float, str]:
    statistic = getattr(result, "statistic", None)
    p_value = getattr(result, "pvalue", getattr(result, "p_value", None))
    if statistic is None or p_value is None:
        if isinstance(result, tuple) and len(result) >= 2:
            statistic, p_value = result[:2]
        else:
            raise TypeError(
                "test must return an object with statistic and pvalue attributes "
                "or a (statistic, p_value) tuple"
            )
    method = getattr(result, "method", getattr(result, "__class__", type(result)).__name__)
    return float(statistic), float(p_value), str(method)


def _default_variable_test(
    frame: pd.DataFrame,
    variable: str,
    group_column: str,
) -> tuple[float, float, str]:
    values = frame[variable]
    if pd.api.types.is_numeric_dtype(values.dtype):
        groups = [
            part[variable].dropna().to_numpy()
            for _, part in frame.groupby(group_column, observed=True)
        ]
        groups = [group for group in groups if len(group)]
        if len(groups) < 2:
            return np.nan, np.nan, "Kruskal-Wallis"
        result = _scipy_stats().kruskal(*groups)
        return float(result.statistic), float(result.pvalue), "Kruskal-Wallis"
    # Non-exclusive modes repeat observations across intersections, which
    # produces a frame with a duplicated index; pd.crosstab tries to
    # index-align its two inputs, so pass plain arrays instead.
    contingency = pd.crosstab(
        frame[group_column].to_numpy(), values.to_numpy(), dropna=True
    )
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return np.nan, np.nan, "Pearson chi-square"
    statistic, p_value, _, _ = _scipy_stats().chi2_contingency(contingency)
    return float(statistic), float(p_value), "Pearson chi-square"


def compute_intersection_tests(
    data: UpSetData,
    *,
    test: Callable[..., Any] | None = None,
    tests: Mapping[str, Callable[..., Any]] | None = None,
    ignore: Sequence[str] = (),
    ignore_mode_columns: bool = True,
    mode: Mode = "exclusive_intersection",
    min_group_size: int = 1,
) -> pd.DataFrame:
    """Compute one unadjusted test per eligible non-membership variable."""

    if not isinstance(data, UpSetData):
        raise TypeError("data must be prepared by upset_data()")
    if min_group_size < 1:
        raise ValueError("min_group_size must be at least 1")
    canonical_mode = normalize_mode(mode)
    if canonical_mode != "exclusive_intersection":
        warnings.warn(
            "Non-exclusive modes repeat observations across intersections; test groups "
            "are therefore not independent.",
            UserWarning,
            stacklevel=2,
        )
    from .plot import _mode_frame

    frame = _mode_frame(data, canonical_mode)
    group_sizes = frame.groupby("intersection", observed=True).size()
    keep = group_sizes.index[group_sizes >= min_group_size]
    frame = frame.loc[frame["intersection"].isin(keep)].copy()
    if len(keep) < 2:
        raise ValueError("at least two intersections with enough observations are required")

    ignored = set(ignore).union(data.statistics.sets).union({"intersection"})
    if ignore_mode_columns:
        ignored.update(
            column
            for column in frame.columns
            if column.startswith("in_") or column.endswith("_size")
        )
    variables = [column for column in frame.columns if column not in ignored]
    per_variable = dict(tests or {})
    rows: list[dict[str, Any]] = []
    for variable in variables:
        selected_test = per_variable.get(variable, test)
        if selected_test is None:
            statistic, p_value, method = _default_variable_test(
                frame, variable, "intersection"
            )
        else:
            groups = [
                part[variable].dropna().to_numpy()
                for _, part in frame.groupby("intersection", observed=True)
            ]
            groups = [group for group in groups if len(group) >= min_group_size]
            if len(groups) < 2:
                statistic, p_value, method = np.nan, np.nan, selected_test.__name__
            else:
                statistic, p_value, method = _extract_test_result(
                    selected_test(*groups)
                )
        rows.append(
            {
                "variable": str(variable),
                "p_value": p_value,
                "statistic": statistic,
                "test": method,
            }
        )
    return pd.DataFrame(rows, columns=["variable", "p_value", "statistic", "test"])


def adjust_intersection_tests(
    results: pd.DataFrame,
    *,
    method: str = "fdr_bh",
) -> pd.DataFrame:
    """Add multiple-testing-adjusted p-values to an intersection test table."""

    required = {"variable", "p_value", "statistic", "test"}
    missing = required.difference(results.columns)
    if missing:
        raise KeyError(f"results is missing required columns: {sorted(missing)}")
    adjusted = results.copy()
    valid = adjusted["p_value"].notna()
    adjusted["fdr"] = np.nan
    if valid.any():
        adjusted.loc[valid, "fdr"] = _multipletests()(
            adjusted.loc[valid, "p_value"].to_numpy(), method=method
        )[1]
    return adjusted


def compare_between_intersections(
    data,
    intersect: Sequence[str] | None = None,
    *,
    test: Callable[..., Any] | None = None,
    tests: Mapping[str, Callable[..., Any]] | None = None,
    ignore: Sequence[str] = (),
    ignore_mode_columns: bool = True,
    mode: Mode = "exclusive_intersection",
    min_group_size: int = 1,
    **upset_data_kwargs: Any,
) -> pd.DataFrame:
    """Compare covariates between intersections without FDR adjustment."""

    if isinstance(data, UpSetData):
        if intersect is not None:
            raise TypeError("intersect must be omitted when data is UpSetData")
        prepared = data
    else:
        if intersect is None:
            raise TypeError("intersect must be provided for a raw pandas.DataFrame")
        prepared = upset_data(data, intersect, mode=mode, **upset_data_kwargs)
    return compute_intersection_tests(
        prepared,
        test=test,
        tests=tests,
        ignore=ignore,
        ignore_mode_columns=ignore_mode_columns,
        mode=mode,
        min_group_size=min_group_size,
    )


def upset_test(
    data,
    intersect: Sequence[str] | None = None,
    *,
    test: Callable[..., Any] | None = None,
    tests: Mapping[str, Callable[..., Any]] | None = None,
    ignore: Sequence[str] = (),
    ignore_mode_columns: bool = True,
    mode: Mode = "exclusive_intersection",
    min_size: int = 1,
    correction: str = "fdr_bh",
    **upset_data_kwargs: Any,
) -> pd.DataFrame:
    """Test covariates and return results ordered by adjusted significance."""

    results = compare_between_intersections(
        data,
        intersect,
        test=test,
        tests=tests,
        ignore=ignore,
        ignore_mode_columns=ignore_mode_columns,
        mode=mode,
        min_group_size=min_size,
        **upset_data_kwargs,
    )
    return adjust_intersection_tests(results, method=correction).sort_values(
        "fdr", na_position="last", kind="stable"
    ).reset_index(drop=True)
