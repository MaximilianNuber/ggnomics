"""Validated UpSet highlighting queries."""

from __future__ import annotations

import builtins
from types import MappingProxyType
from typing import Any, Sequence

from ._types import UpSetQuery


def upset_query(
    *,
    set: str | None = None,
    intersect: Sequence[str] | None = None,
    group: str | None = None,
    only_components: Sequence[str] | None = None,
    **aesthetics: Any,
) -> UpSetQuery:
    """Construct one set, intersection, or group highlighting query."""

    passed = sum(value is not None for value in (set, intersect, group))
    if passed != 1:
        raise ValueError("pass exactly one of set, intersect, or group")
    if not aesthetics:
        raise ValueError(
            "pass at least one highlight aesthetic, for example color='red' or fill='red'"
        )
    if set is not None and not isinstance(set, str):
        raise TypeError("set must be a string or None")
    if group is not None and not isinstance(group, str):
        raise TypeError("group must be a string or None")
    members = None if intersect is None else tuple(intersect)
    if members is not None and len(members) != len(builtins.set(members)):
        raise ValueError("intersect query contains duplicate set names")
    return UpSetQuery(
        set=set,
        intersect=members,
        group=group,
        only_components=None if only_components is None else tuple(only_components),
        aesthetics=MappingProxyType(dict(aesthetics)),
    )
