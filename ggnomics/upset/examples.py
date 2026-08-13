"""Small deterministic datasets used in examples and tests."""

from __future__ import annotations

import pandas as pd


def create_upset_abc_example() -> pd.DataFrame:
    """Create the A/B/C membership example distributed with ComplexUpset."""

    rows = [
        ((True, False, False), 50),
        ((False, True, False), 50),
        ((False, False, True), 200),
        ((True, True, False), 10),
        ((True, False, True), 6),
        ((False, True, True), 6),
        ((True, True, True), 1),
        ((False, False, False), 2),
    ]
    values = [membership for membership, count in rows for _ in range(count)]
    return pd.DataFrame(values, columns=["A", "B", "C"], dtype=bool)
