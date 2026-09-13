"""Cleaning and gap-auditing for time-stamped sensor data.

The raw Monash Smart Infrastructure exports contain missing, irregular or
potentially unreliable observations that must be audited before use (see
project brief, Dataset 2). This module provides small, composable functions
for that auditing step rather than silently imputing data.
"""

from __future__ import annotations

import pandas as pd


def detect_missing_intervals(df: pd.DataFrame, expected_freq: str) -> pd.DataFrame:
    """Identify gaps in a regularly-sampled time series.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by a ``DatetimeIndex`` (as returned by
        :func:`smart_infra.data_loader.load_sensor_csv`).
    expected_freq : str
        Pandas frequency string describing the expected sampling interval,
        e.g. ``"5min"`` or ``"1H"``.

    Returns
    -------
    pandas.DataFrame
        One row per detected gap with columns ``gap_start``, ``gap_end`` and
        ``missing_samples`` (number of expected samples absent between the
        two surrounding observations, exclusive of the observations
        themselves). Empty if no gaps are found.

    Raises
    ------
    TypeError
        If ``df`` is not indexed by a ``DatetimeIndex``.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("df must be indexed by a DatetimeIndex")

    if len(df) < 2:
        return pd.DataFrame(columns=["gap_start", "gap_end", "missing_samples"])

    step = pd.Timedelta(pd.tseries.frequencies.to_offset(expected_freq))
    deltas = df.index.to_series().diff().iloc[1:]

    # Any gap larger than 1.5x the expected step is treated as missing data
    # rather than normal jitter in sensor reporting.
    gap_mask = deltas > (step * 1.5)
    gaps = deltas[gap_mask]

    records = []
    for end_ts, delta in gaps.items():
        start_ts = end_ts - delta
        missing_samples = int(round(delta / step)) - 1
        records.append(
            {"gap_start": start_ts, "gap_end": end_ts, "missing_samples": missing_samples}
        )

    return pd.DataFrame.from_records(
        records, columns=["gap_start", "gap_end", "missing_samples"]
    )


def handle_missing_values(
    df: pd.DataFrame,
    method: str = "interpolate",
    limit: int | None = None,
) -> pd.DataFrame:
    """Fill or drop missing values in sensor columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data, typically already indexed by timestamp.
    method : {"interpolate", "ffill", "drop"}, default "interpolate"
        Strategy applied to numeric columns:

        - ``"interpolate"``: linear time-based interpolation.
        - ``"ffill"``: forward-fill the last valid observation.
        - ``"drop"``: drop rows containing any missing value.
    limit : int, optional
        Maximum number of consecutive missing values to fill for
        ``"interpolate"`` and ``"ffill"``. Unlimited if ``None``.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with missing values handled according to ``method``.

    Raises
    ------
    ValueError
        If ``method`` is not one of the supported strategies.
    """
    if method == "interpolate":
        return df.interpolate(method="time", limit=limit)
    if method == "ffill":
        return df.ffill(limit=limit)
    if method == "drop":
        return df.dropna()

    raise ValueError(f"Unsupported method '{method}'. Use 'interpolate', 'ffill' or 'drop'.")
