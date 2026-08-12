"""Shared conversion helpers for BiocPy container backends."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def metadata_frame(
    data: object,
    attributes: Iterable[str],
    *,
    length: int,
) -> pd.DataFrame:
    """Return a BiocFrame-like metadata slot as a plain DataFrame."""

    metadata = None
    for attribute in attributes:
        metadata = getattr(data, attribute, None)
        if metadata is not None:
            break

    if metadata is None:
        return pd.DataFrame(index=range(length))

    if hasattr(metadata, "to_pandas"):
        frame = metadata.to_pandas().copy()
    else:
        frame = pd.DataFrame(metadata).copy()

    # BiocFrame.to_pandas() exposes row names as a regular column.
    if "rownames" in frame.columns:
        frame = frame.drop(columns="rownames")
    return frame.reset_index(drop=True)


def column_data_frame(data: object) -> pd.DataFrame:
    """Return ``column_data``/``col_data`` from a BiocPy container."""

    return metadata_frame(
        data,
        ("column_data", "col_data"),
        length=data.shape[1],
    )


def row_data_frame(data: object) -> pd.DataFrame:
    """Return ``row_data`` from a BiocPy container."""

    return metadata_frame(data, ("row_data",), length=data.shape[0])


__all__ = ["column_data_frame", "row_data_frame"]
