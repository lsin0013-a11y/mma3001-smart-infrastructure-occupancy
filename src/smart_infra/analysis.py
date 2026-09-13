"""Summary statistics and occupancy/environment relationship analysis.

These functions operate on cleaned DataFrames (see
:mod:`smart_infra.preprocessing`) and provide generic building blocks that
the specific engineering analysis (e.g. occupancy prediction, pattern
detection) is built on top of.
"""

from __future__ import annotations

import pandas as pd


def summarize_readings(
    df: pd.DataFrame,
    value_column: str,
    group_column: str | None = None,
) -> pd.DataFrame:
    """Compute count/mean/std/min/max summary statistics for a sensor column.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing ``value_column`` (and ``group_column`` if
        given).
    value_column : str
        Name of the numeric column to summarise, e.g. a CO2 or temperature
        reading.
    group_column : str, optional
        If given, statistics are computed per distinct value of this column
        (e.g. per room or zone identifier) instead of over the whole
        DataFrame.

    Returns
    -------
    pandas.DataFrame
        Summary statistics with columns ``count``, ``mean``, ``std``,
        ``min``, ``max``, indexed by ``group_column`` if provided, otherwise
        a single row.

    Raises
    ------
    KeyError
        If ``value_column`` (or ``group_column``) is not present in ``df``.
    """
    if value_column not in df.columns:
        raise KeyError(f"'{value_column}' not found in DataFrame columns")

    stats = ["count", "mean", "std", "min", "max"]

    if group_column is None:
        summary = df[value_column].agg(stats).to_frame().T
        summary.index = ["all"]
        return summary

    if group_column not in df.columns:
        raise KeyError(f"'{group_column}' not found in DataFrame columns")

    return df.groupby(group_column)[value_column].agg(stats)


def merge_occupancy_environment(
    occupancy_df: pd.DataFrame,
    environment_df: pd.DataFrame,
    tolerance: str = "5min",
) -> pd.DataFrame:
    """Align occupancy and environmental readings on their nearest timestamps.

    Occupancy and environmental sensors are rarely sampled on identical
    clocks, so an as-of (nearest-neighbour) merge within a tolerance window
    is used rather than an exact join.

    Parameters
    ----------
    occupancy_df : pandas.DataFrame
        Occupancy readings indexed by a sorted ``DatetimeIndex``.
    environment_df : pandas.DataFrame
        Environmental readings indexed by a sorted ``DatetimeIndex``.
    tolerance : str, default "5min"
        Maximum allowed time difference between matched occupancy and
        environmental observations, as a pandas offset alias.

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame indexed by the occupancy timestamps, with
        environmental columns attached where a match within ``tolerance``
        exists (NaN otherwise).

    Raises
    ------
    TypeError
        If either input is not indexed by a ``DatetimeIndex``.
    """
    if not isinstance(occupancy_df.index, pd.DatetimeIndex):
        raise TypeError("occupancy_df must be indexed by a DatetimeIndex")
    if not isinstance(environment_df.index, pd.DatetimeIndex):
        raise TypeError("environment_df must be indexed by a DatetimeIndex")

    return pd.merge_asof(
        occupancy_df.sort_index(),
        environment_df.sort_index(),
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )
