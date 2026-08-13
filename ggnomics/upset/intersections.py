"""Compute, select, and apply UpSet intersection representations."""

from __future__ import annotations

from itertools import chain, combinations
from math import inf
from types import MappingProxyType
from typing import Literal, Sequence
import warnings

import numpy as np
import pandas as pd

from ._types import IntersectionSelection, IntersectionStatistics, UpSetData
from .modes import Mode, normalize_mode

_CANONICAL_MODES = (
    "exclusive_intersection",
    "inclusive_intersection",
    "exclusive_union",
    "inclusive_union",
)


def _stack_frame(data: pd.DataFrame) -> pd.Series:
    """Stack across pandas 2.x and 3.x without changing Boolean values."""
    try:
        return data.stack(future_stack=True)
    except TypeError:
        return data.stack(dropna=False)


def _validate_frame(data: pd.DataFrame, intersect: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"data must be a pandas.DataFrame; got {type(data).__name__}")
    if not data.columns.is_unique:
        duplicated = data.columns[data.columns.duplicated()].tolist()
        raise ValueError(f"data columns must be unique; duplicated: {duplicated}")
    sets = tuple(intersect)
    if len(sets) < 2:
        raise ValueError(f"intersect must contain at least two columns; got {len(sets)}")
    if len(set(sets)) != len(sets):
        raise ValueError(f"intersect must contain unique names; got {sets!r}")
    missing = [name for name in sets if name not in data.columns]
    if missing:
        raise KeyError(f"intersect columns missing from data: {missing}")
    return sets


def _convert_memberships(
    data: pd.DataFrame,
    sets: tuple[str, ...],
    warn_when_converting: bool | Literal["auto"],
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    converted: list[str] = []
    columns: dict[str, pd.Series] = {}
    for name in sets:
        values = data[name]
        if values.isna().any():
            count = int(values.isna().sum())
            raise ValueError(
                f"data[{name!r}] must not contain missing memberships; got {count}"
            )
        if pd.api.types.is_bool_dtype(values.dtype):
            columns[name] = values.astype(bool)
            continue
        unique = set(pd.unique(values))
        if unique.issubset({0, 1, 0.0, 1.0}):
            columns[name] = values.astype(bool)
            converted.append(name)
            continue
        raise TypeError(
            f"data[{name!r}] must be Boolean or contain only 0/1; "
            f"got dtype={values.dtype} and values={sorted(map(str, unique))[:6]}"
        )
    if converted and warn_when_converting in {True, "auto"}:
        warnings.warn(
            "Converted 0/1 membership columns to Boolean: " + ", ".join(converted),
            UserWarning,
            stacklevel=3,
        )
    frame = pd.DataFrame(columns, index=data.index, copy=False)
    return frame, tuple(converted)


def _powerset(values: tuple[str, ...]) -> list[tuple[str, ...]]:
    return list(
        chain.from_iterable(combinations(values, degree) for degree in range(len(values) + 1))
    )


def _candidate_intersections(
    memberships: pd.DataFrame,
    sets: tuple[str, ...],
    intersections: Literal["observed", "all"] | Sequence[Sequence[str]],
    max_combinations_datapoints_n: int,
) -> list[tuple[str, ...]]:
    observed = [
        tuple(name for name, present in zip(sets, row, strict=True) if present)
        for row in memberships.to_numpy(dtype=bool)
    ]
    if isinstance(intersections, str):
        if intersections == "observed":
            return list(dict.fromkeys(observed))
        if intersections != "all":
            raise ValueError(
                "intersections must be 'observed', 'all', or a sequence of set-name sequences; "
                f"got {intersections!r}"
            )
        candidate_n = 2 ** len(sets)
        work_n = candidate_n * max(len(memberships), 1)
        if work_n > max_combinations_datapoints_n:
            raise ValueError(
                "intersections='all' would create "
                f"{candidate_n:,} combinations across {len(memberships):,} observations "
                f"({work_n:,} combination-observations), exceeding "
                f"max_combinations_datapoints_n={max_combinations_datapoints_n:,}"
            )
        return _powerset(sets)

    candidates: list[tuple[str, ...]] = []
    known = set(sets)
    for position, item in enumerate(intersections):
        requested = tuple(item)
        unknown = [name for name in requested if name not in known]
        if unknown:
            raise KeyError(
                f"intersections[{position}] contains names absent from intersect: {unknown}"
            )
        if len(set(requested)) != len(requested):
            raise ValueError(f"intersections[{position}] contains duplicate set names")
        requested_set = set(requested)
        candidates.append(tuple(name for name in sets if name in requested_set))
    if not candidates:
        raise ValueError("intersections must contain at least one candidate")
    return list(dict.fromkeys(candidates))


def _encode_intersections(
    candidates: Sequence[tuple[str, ...]],
) -> tuple[dict[str, tuple[str, ...]], dict[tuple[str, ...], str]]:
    by_id = {f"I{position}": members for position, members in enumerate(candidates)}
    by_members = {members: identifier for identifier, members in by_id.items()}
    return by_id, by_members


def _compute_sizes(
    memberships: pd.DataFrame,
    candidates: Sequence[tuple[str, ...]],
    sets: tuple[str, ...],
    identifiers: Sequence[str],
) -> pd.DataFrame:
    observed = memberships.to_numpy(dtype=np.int8, copy=False)
    candidate_matrix = np.asarray(
        [[name in members for name in sets] for members in candidates],
        dtype=np.int8,
    ).reshape(len(candidates), len(sets))
    overlap = observed @ candidate_matrix.T
    observed_degree = observed.sum(axis=1)
    candidate_degree = candidate_matrix.sum(axis=1)

    exact = (overlap == observed_degree[:, None]) & (
        overlap == candidate_degree[None, :]
    )
    inclusive_intersection = overlap >= candidate_degree[None, :]
    inclusive_union = overlap > 0
    exclusive_union = (overlap == observed_degree[:, None]) & (
        observed_degree[:, None] > 0
    )

    empty_candidates = candidate_degree == 0
    if empty_candidates.any():
        empty_observations = observed_degree == 0
        for matrix in (inclusive_intersection, inclusive_union, exclusive_union):
            matrix[:, empty_candidates] = empty_observations[:, None]

    result = pd.DataFrame(
        {
            "exclusive_intersection": exact.sum(axis=0),
            "inclusive_intersection": inclusive_intersection.sum(axis=0),
            "exclusive_union": exclusive_union.sum(axis=0),
            "inclusive_union": inclusive_union.sum(axis=0),
        },
        index=pd.Index(identifiers, name="intersection"),
        dtype="int64",
    )
    return result


def compute_intersections(
    data: pd.DataFrame,
    *,
    intersect: Sequence[str],
    intersections: Literal["observed", "all"] | Sequence[Sequence[str]] = "observed",
    warn_when_converting: bool | Literal["auto"] = "auto",
    encode_sets: bool = False,
    size_columns_suffix: str = "_size",
    max_combinations_datapoints_n: int = 10_000_000_000,
) -> IntersectionStatistics:
    """Compute unfiltered set memberships and four ComplexUpset size modes.

    The result is consumed by :func:`select_intersections`.  No caller-owned
    object is mutated.  ``encode_sets`` is accepted for API parity; opaque
    intersection identifiers make lossy set-name encoding unnecessary.
    """

    del encode_sets
    if not isinstance(size_columns_suffix, str) or not size_columns_suffix:
        raise ValueError("size_columns_suffix must be a non-empty string")
    if max_combinations_datapoints_n < 1:
        raise ValueError("max_combinations_datapoints_n must be positive")

    sets = _validate_frame(data, intersect)
    memberships, converted = _convert_memberships(data, sets, warn_when_converting)
    candidates = _candidate_intersections(
        memberships, sets, intersections, max_combinations_datapoints_n
    )
    by_id, by_members = _encode_intersections(candidates)
    identifiers = tuple(by_id)
    sizes = _compute_sizes(memberships, candidates, sets, identifiers)
    degrees = pd.Series(
        [len(by_id[identifier]) for identifier in identifiers],
        index=sizes.index,
        name="degree",
        dtype="int64",
    )
    set_sizes = memberships.sum(axis=0).astype("int64").rename("size")
    observed_members = [
        tuple(name for name, present in zip(sets, row, strict=True) if present)
        for row in memberships.to_numpy(dtype=bool, copy=False)
    ]
    element_ids = pd.Series(
        [by_members.get(members, "") for members in observed_members],
        index=data.index,
        name="intersection",
        dtype="object",
    )
    return IntersectionStatistics(
        data=data.copy(deep=False),
        sets=sets,
        memberships=memberships,
        element_intersections=element_ids,
        intersection_members=MappingProxyType(by_id),
        sizes=sizes,
        degrees=degrees,
        set_sizes=set_sizes,
        converted_columns=converted,
        size_columns_suffix=size_columns_suffix,
    )


def _validate_sort(value: str | bool, name: str) -> None:
    if value not in {"ascending", "descending", False}:
        raise ValueError(
            f"{name} must be 'ascending', 'descending', or False; got {value!r}"
        )


def select_intersections(
    statistics: IntersectionStatistics,
    *,
    min_size: int = 0,
    max_size: float = inf,
    min_degree: int = 0,
    max_degree: float = inf,
    n_intersections: int | None = None,
    keep_empty_groups: bool = False,
    sort_sets: Literal["ascending", "descending", False] = "descending",
    sort_intersections: Literal["ascending", "descending", False] = "descending",
    sort_intersections_by: Sequence[Literal["cardinality", "degree", "ratio"]] = (
        "cardinality",
    ),
    sort_ratio_numerator: Mode = "exclusive_intersection",
    sort_ratio_denominator: Mode = "inclusive_union",
    group_by: Literal["degree", "sets"] = "degree",
    mode: Mode = "exclusive_intersection",
) -> IntersectionSelection:
    """Derive a serializable filtering and ordering criterion."""

    if not isinstance(statistics, IntersectionStatistics):
        raise TypeError("statistics must be an IntersectionStatistics")
    if min_size < 0 or max_size < min_size:
        raise ValueError("size bounds must satisfy 0 <= min_size <= max_size")
    if min_degree < 0 or max_degree < min_degree:
        raise ValueError("degree bounds must satisfy 0 <= min_degree <= max_degree")
    if n_intersections is not None and n_intersections < 1:
        raise ValueError("n_intersections must be at least 1 or None")
    _validate_sort(sort_sets, "sort_sets")
    _validate_sort(sort_intersections, "sort_intersections")
    if group_by not in {"degree", "sets"}:
        raise ValueError("group_by must be 'degree' or 'sets'")
    sort_by = tuple(sort_intersections_by)
    invalid = [item for item in sort_by if item not in {"cardinality", "degree", "ratio"}]
    if invalid:
        raise ValueError(f"invalid sort_intersections_by values: {invalid}")

    canonical_mode = normalize_mode(mode)
    numerator = normalize_mode(sort_ratio_numerator)
    denominator = normalize_mode(sort_ratio_denominator)
    selected = statistics.sizes.index[
        statistics.sizes[canonical_mode].between(min_size, max_size)
        & statistics.degrees.between(min_degree, max_degree)
    ].tolist()
    if not selected:
        raise ValueError(
            "No intersections remain after filtering; loosen the size or degree bounds"
        )

    if n_intersections is not None and len(selected) > n_intersections:
        selected = (
            statistics.sizes.loc[selected, canonical_mode]
            .sort_values(ascending=False, kind="stable")
            .head(n_intersections)
            .index.tolist()
        )

    if sort_intersections is not False:
        table = pd.DataFrame(index=pd.Index(selected, name="intersection"))
        for position, criterion in enumerate(sort_by):
            if criterion == "cardinality":
                table[f"key_{position}"] = statistics.sizes.loc[selected, canonical_mode]
            elif criterion == "degree":
                table[f"key_{position}"] = statistics.degrees.loc[selected]
            else:
                den = statistics.sizes.loc[selected, denominator].replace(0, np.nan)
                table[f"key_{position}"] = statistics.sizes.loc[selected, numerator] / den
        ascending = sort_intersections == "ascending"
        selected = table.sort_values(
            list(table.columns), ascending=ascending, kind="stable", na_position="last"
        ).index.tolist()

    participating = {
        member
        for identifier in selected
        for member in statistics.intersection_members[identifier]
    }
    dropped = tuple(name for name in statistics.sets if name not in participating)
    kept_sets = statistics.sets if keep_empty_groups else tuple(
        name for name in statistics.sets if name in participating
    )
    if sort_sets is not False:
        ascending = sort_sets == "ascending"
        kept_sets = tuple(
            statistics.set_sizes.loc[list(kept_sets)]
            .sort_values(ascending=ascending, kind="stable")
            .index
        )

    if group_by == "sets":
        rank = {name: position for position, name in enumerate(kept_sets)}

        def group_key(identifier: str):
            members = statistics.intersection_members[identifier]
            lead = min((rank.get(name, len(rank)) for name in members), default=len(rank))
            return lead

        selected = sorted(selected, key=group_key)

    return IntersectionSelection(
        intersections=tuple(selected),
        sets=tuple(kept_sets),
        dropped_sets=dropped,
        min_size=min_size,
        max_size=max_size,
        min_degree=min_degree,
        max_degree=max_degree,
        n_intersections=n_intersections,
        sort_sets=sort_sets,
        sort_intersections=sort_intersections,
        sort_intersections_by=sort_by,
        sort_ratio_numerator=numerator,
        sort_ratio_denominator=denominator,
        group_by=group_by,
        mode=canonical_mode,
        origin_id=id(statistics),
    )


def report_dropped_sets(
    selection: IntersectionSelection,
    *,
    emit=warnings.warn,
) -> IntersectionSelection:
    """Report dropped sets and return the unchanged selection."""

    if selection.dropped_sets:
        names = ", ".join(selection.dropped_sets)
        emit(f"Dropping empty groups: {names}")
    return selection


def apply_intersection_selection(
    statistics: IntersectionStatistics,
    *,
    selection: IntersectionSelection,
) -> UpSetData:
    """Apply one criterion and construct aligned plot-ready representations."""

    if selection.origin_id != id(statistics):
        raise KeyError(
            "selection was not produced by select_intersections(statistics, ...) "
            "for this exact IntersectionStatistics object; apply_intersection_selection "
            "requires a statistics/selection pair that correspond to each other"
        )

    unknown = [
        identifier
        for identifier in selection.intersections
        if identifier not in statistics.intersection_members
    ]
    if unknown:
        raise KeyError(f"selection contains unknown intersection identifiers: {unknown}")
    unknown_sets = [name for name in selection.sets if name not in statistics.sets]
    if unknown_sets:
        raise KeyError(f"selection contains unknown sets: {unknown_sets}")

    intersections = selection.intersections
    sets = selection.sets
    intersection_dtype = pd.CategoricalDtype(intersections, ordered=True)
    set_dtype = pd.CategoricalDtype(sets, ordered=True)
    sizes = statistics.sizes.loc[list(intersections)].copy()
    sizes.index = pd.CategoricalIndex(
        sizes.index, dtype=intersection_dtype, name="intersection"
    )

    matrix = pd.DataFrame(
        {
            identifier: [
                name in statistics.intersection_members[identifier] for name in sets
            ]
            for identifier in intersections
        },
        index=pd.Index(sets, name="group"),
        dtype=bool,
    )
    matrix_frame = (
        _stack_frame(matrix.rename_axis(columns="intersection"))
        .rename("value")
        .reset_index()
    )
    matrix_frame["group"] = matrix_frame["group"].astype(set_dtype)
    matrix_frame["intersection"] = matrix_frame["intersection"].astype(
        intersection_dtype
    )
    set_positions = {name: position for position, name in enumerate(sets)}
    intersection_positions = {
        name: position for position, name in enumerate(intersections)
    }
    matrix_frame["group_position"] = matrix_frame["group"].map(set_positions).astype(int)
    matrix_frame["intersection_position"] = (
        matrix_frame["intersection"].map(intersection_positions).astype(int)
    )

    presence = (
        _stack_frame(
            statistics.memberships.loc[:, list(sets)].rename_axis(
                index="observation", columns="group"
            )
        )
        .rename("value")
        .reset_index()
    )
    presence = presence.loc[presence["value"]].reset_index(drop=True)
    presence["group"] = presence["group"].astype(set_dtype)

    with_sizes = statistics.data.copy()
    selected_elements = statistics.element_intersections.where(
        statistics.element_intersections.isin(intersections)
    )
    with_sizes["intersection"] = pd.Categorical(
        selected_elements, dtype=intersection_dtype
    )
    with_sizes = with_sizes.loc[with_sizes["intersection"].notna()].copy()
    for canonical_mode in _CANONICAL_MODES:
        column = f"{canonical_mode}{statistics.size_columns_suffix}"
        mapping = statistics.sizes[canonical_mode]
        with_sizes[column] = (
            with_sizes["intersection"].astype("object").map(mapping).astype("int64")
        )
        with_sizes[f"in_{canonical_mode}"] = True

    return UpSetData(
        statistics=statistics,
        selection=selection,
        with_sizes=with_sizes,
        presence=presence,
        matrix=matrix,
        matrix_frame=matrix_frame,
        sizes=sizes,
        sorted_sets=sets,
        sorted_intersections=intersections,
        plot_sets_subset=sets,
        plot_intersections_subset=intersections,
        intersection_members=statistics.intersection_members,
    )


def upset_data(
    data: pd.DataFrame,
    intersect: Sequence[str] | None = None,
    *,
    min_size: int = 0,
    max_size: float = inf,
    min_degree: int = 0,
    max_degree: float = inf,
    n_intersections: int | None = None,
    keep_empty_groups: bool = False,
    warn_when_dropping_groups: bool = False,
    warn_when_converting: bool | Literal["auto"] = "auto",
    sort_sets: Literal["ascending", "descending", False] = "descending",
    sort_intersections: Literal["ascending", "descending", False] = "descending",
    sort_intersections_by: Sequence[Literal["cardinality", "degree", "ratio"]] = (
        "cardinality",
    ),
    sort_ratio_numerator: Mode = "exclusive_intersection",
    sort_ratio_denominator: Mode = "inclusive_union",
    group_by: Literal["degree", "sets"] = "degree",
    mode: Mode = "exclusive_intersection",
    size_columns_suffix: str = "_size",
    encode_sets: bool = False,
    max_combinations_datapoints_n: int = 10_000_000_000,
    intersections: Literal["observed", "all"] | Sequence[Sequence[str]] = "observed",
) -> UpSetData:
    """Prepare data for UpSet plots using compute, select, then apply.

    ``intersect`` remains positional-or-keyword for direct ComplexUpset parity.
    """

    if intersect is None:
        raise TypeError("intersect must be provided")
    statistics = compute_intersections(
        data,
        intersect=intersect,
        intersections=intersections,
        warn_when_converting=warn_when_converting,
        encode_sets=encode_sets,
        size_columns_suffix=size_columns_suffix,
        max_combinations_datapoints_n=max_combinations_datapoints_n,
    )
    selection = select_intersections(
        statistics,
        min_size=min_size,
        max_size=max_size,
        min_degree=min_degree,
        max_degree=max_degree,
        n_intersections=n_intersections,
        keep_empty_groups=keep_empty_groups,
        sort_sets=sort_sets,
        sort_intersections=sort_intersections,
        sort_intersections_by=sort_intersections_by,
        sort_ratio_numerator=sort_ratio_numerator,
        sort_ratio_denominator=sort_ratio_denominator,
        group_by=group_by,
        mode=mode,
    )
    if warn_when_dropping_groups:
        selection = report_dropped_sets(selection)
    return apply_intersection_selection(statistics, selection=selection)
