"""Query matching shared by the panel builders in annotations.py, matrix.py, and set_size.py."""

from __future__ import annotations

from typing import Sequence

from ._types import UpSetData, UpSetQuery


def validate_queries(data: UpSetData, queries: Sequence[UpSetQuery]) -> None:
    """Reject group queries and unknown set/intersection names explicitly.

    ``data`` must be prepared by :func:`~ggnomics.upset.upset_data`. Group-query
    highlighting is not implemented: no panel builder reads ``UpSetQuery.group``.
    Rather than silently accepting and ignoring it, this raises so callers learn
    immediately rather than seeing an unhighlighted plot.

    Returns nothing; raises ``NotImplementedError`` or ``KeyError`` on an
    invalid query. Consumed by :func:`~ggnomics.upset.compose_upset` before any
    panel is built.
    """

    known_sets = set(data.statistics.sets)
    for query in queries:
        if query.group is not None:
            raise NotImplementedError(
                "upset_query(group=...) highlighting is not implemented in "
                "ggnomics.upset; only set=... and intersect=[...] queries "
                "currently affect plot components."
            )
        if query.set is not None and query.set not in known_sets:
            raise KeyError(f"query set {query.set!r} is not among the prepared sets {sorted(known_sets)}")
        if query.intersect is not None:
            unknown = [name for name in query.intersect if name not in known_sets]
            if unknown:
                raise KeyError(f"query intersect contains unknown set names: {unknown}")


def find_query_intersection(data: UpSetData, query: UpSetQuery) -> str | None:
    """Find the intersection identifier that a query's ``intersect`` names.

    ``data`` must be prepared by :func:`~ggnomics.upset.upset_data`. Returns
    the identifier in ``data.intersection_members`` whose member sets exactly
    equal ``query.intersect``, or ``None`` when ``query.intersect`` is unset
    or matches no prepared intersection. Consumed by the panel builders in
    annotations.py and matrix.py to look up which rows a query highlights.
    """

    if query.intersect is None:
        return None
    requested = set(query.intersect)
    canonical = tuple(name for name in data.statistics.sets if name in requested)
    for identifier, members in data.intersection_members.items():
        if members == canonical:
            return identifier
    return None


def query_applies(query: UpSetQuery, component: str) -> bool:
    """Decide whether one query highlights a named plot component.

    Returns ``True`` when ``query.only_components`` is unset (the query
    applies everywhere) or contains ``component``. Consumed by every panel
    builder to filter the queries it should render.
    """

    return query.only_components is None or component in query.only_components
