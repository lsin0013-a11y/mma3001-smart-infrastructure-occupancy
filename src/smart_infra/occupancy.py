"""Turn the raw occupancy event stream into a clean hourly occupied/empty table.

The occupancy export is an event stream, not a regular time series: rows
arrive about every 2 s while a space is occupied and none are written while
it is empty (see ``reports/data_audit_findings.md`` §2). This module cleans
those events and reduces them to one row per hour, which is the target series
for the occupancy forecast.

Each cleaning rule carries the ID used in the audit's cleaning plan (§7).
Suspect hours are flagged in their own column rather than deleted, so every
rule can be switched off for sensitivity analysis.
"""

from __future__ import annotations

import pandas as pd

LOCAL_TZ = "Australia/Melbourne"

SPACE_IDS = {
    "G.20": "cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b",
    "G.25": "227f1068-9e41-469f-bef0-275f27a0bdba",
}
"""``floorspaceid`` of the two occupancy spaces the metadata sheet maps to a room."""

EVENT_COLUMNS = ["id", "timestamp", "status", "headcount"]
"""Columns :func:`clean_events` expects, renamed from the raw export."""


def clean_events(events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove invalid and duplicated occupancy events for one space.

    Parameters
    ----------
    events : pandas.DataFrame
        Events for a single space with columns ``id`` (raw row id, increasing
        with arrival order), ``timestamp`` (timezone-aware), ``status``
        (``occupancystatus``) and ``headcount``. Other columns are dropped.

    Returns
    -------
    events : pandas.DataFrame
        Cleaned events sorted by ``timestamp``, with a fresh integer index.
    log : dict of str to int
        Number of rows removed by each rule:

        - ``OCC-RANGE-01``: headcount missing or negative.
        - ``OCC-DUP-01``: identical to another row apart from ``id``
          (the row with the lowest ``id`` is kept).
        - ``OCC-DUP-02``: same timestamp as another row but different values
          (the row with the highest ``id``, i.e. the latest, is kept).

    Raises
    ------
    ValueError
        If any of :data:`EVENT_COLUMNS` is missing.
    TypeError
        If ``timestamp`` is not timezone-aware. Naive times are rejected
        because hours would be ambiguous across daylight-saving changes.
    """
    missing = [c for c in EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing}")
    if not isinstance(events["timestamp"].dtype, pd.DatetimeTZDtype):
        raise TypeError("timestamp must be timezone-aware")

    log = {}
    df = events[EVENT_COLUMNS].sort_values(["timestamp", "id"])

    invalid = df["headcount"].isna() | (df["headcount"] < 0)
    log["OCC-RANGE-01"] = int(invalid.sum())
    df = df[~invalid]

    n = len(df)
    df = df.drop_duplicates(subset=["timestamp", "status", "headcount"], keep="first")
    log["OCC-DUP-01"] = n - len(df)

    n = len(df)
    df = df.drop_duplicates(subset="timestamp", keep="last")
    log["OCC-DUP-02"] = n - len(df)

    return df.reset_index(drop=True), log


def events_to_hourly(
    events: pd.DataFrame,
    occupied_threshold: int = 1,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Reduce cleaned events to one row per hour.

    Parameters
    ----------
    events : pandas.DataFrame
        Output of :func:`clean_events`.
    occupied_threshold : int, default 1
        An hour is occupied if its maximum headcount is at least this value.
    start, end : pandas.Timestamp, optional
        Timezone-aware first and last hour to include. Default to the hours of
        the first and last event.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``hour_start`` (hourly, Australia/Melbourne time, every hour
        from ``start`` to ``end`` inclusive) with columns:

        - ``n_events``: number of events in the hour (0 if none).
        - ``max_headcount``: highest headcount in the hour (0 if none).
        - ``occupied``: ``max_headcount >= occupied_threshold``.

        Hours with no events are treated as empty. Use :func:`flag_outages`
        to mark those that are more likely sensor outages.

    Raises
    ------
    ValueError
        If ``events`` is empty or ``occupied_threshold`` is below 1.
    """
    if events.empty:
        raise ValueError("No events to aggregate")
    if occupied_threshold < 1:
        raise ValueError("occupied_threshold must be at least 1")

    # Melbourne's UTC offset is a whole number of hours, so UTC hour boundaries
    # are also local hour boundaries. Working in UTC avoids the repeated hour
    # when daylight saving ends.
    hour = events["timestamp"].dt.tz_convert("UTC").dt.floor("h")
    hourly = events.groupby(hour)["headcount"].agg(n_events="size", max_headcount="max")

    first = hour.min() if start is None else start.tz_convert("UTC").floor("h")
    last = hour.max() if end is None else end.tz_convert("UTC").floor("h")
    index = pd.date_range(first, last, freq="h")
    hourly = hourly.reindex(index, fill_value=0)

    hourly["occupied"] = hourly["max_headcount"] >= occupied_threshold
    hourly.index = index.tz_convert(LOCAL_TZ).rename("hour_start")
    return hourly


def flag_outages(
    hourly: pd.DataFrame,
    events: pd.DataFrame,
    max_gap: str = "24h",
    working_hours: tuple[int, int] = (9, 17),
) -> pd.DataFrame:
    """Flag hours that fall in a likely sensor outage (rule ``OCC-GAP-01``).

    A gap between consecutive events is treated as a likely outage when it is
    longer than ``max_gap`` and covers at least one weekday working hour. A
    busy room going silent for a whole working day is more likely a sensor or
    network fault than an empty room, while a long silence over a weekend is
    plausible. Holiday closures are also flagged, since they can't be told
    apart from outages without a calendar.

    Parameters
    ----------
    hourly : pandas.DataFrame
        Output of :func:`events_to_hourly`.
    events : pandas.DataFrame
        The cleaned events ``hourly`` was built from.
    max_gap : str, default "24h"
        Shortest gap (pandas timedelta string) that can count as an outage.
    working_hours : tuple of int, default (9, 17)
        Local start (inclusive) and end (exclusive) hour of the working day.

    Returns
    -------
    pandas.DataFrame
        A copy of ``hourly`` with a boolean ``outage`` column. Only hours lying
        entirely inside a flagged gap are marked, so they never contain events.
    """
    hourly = hourly.copy()
    hourly["outage"] = False

    idx = hourly.index
    working = (idx.dayofweek < 5) & (idx.hour >= working_hours[0]) & (idx.hour < working_hours[1])

    t = events["timestamp"].sort_values()
    gaps = t.diff()
    long_gaps = gaps > pd.Timedelta(max_gap)
    for gap_end, gap in zip(t[long_gaps], gaps[long_gaps]):
        # Strictly after the gap's first event, so that event's own hour is excluded.
        inside = (idx > gap_end - gap) & (idx + pd.Timedelta("1h") <= gap_end)
        if (inside & working).any():
            hourly.loc[inside, "outage"] = True

    return hourly


def flag_night_detections(hourly: pd.DataFrame, night_hours: tuple[int, int] = (1, 5)) -> pd.DataFrame:
    """Flag occupied hours in the middle of the night (rule ``OCC-NIGHT-01``).

    The audit found detections between 01:00 and 05:00 on many nights. They
    may be false detections or genuine late-night use, so they are flagged for
    sensitivity analysis rather than removed.

    Parameters
    ----------
    hourly : pandas.DataFrame
        Output of :func:`events_to_hourly` (or :func:`flag_outages`).
    night_hours : tuple of int, default (1, 5)
        Local start (inclusive) and end (exclusive) hour of the night window.

    Returns
    -------
    pandas.DataFrame
        A copy of ``hourly`` with a boolean ``night_detection`` column.
    """
    hourly = hourly.copy()
    hour = hourly.index.hour
    hourly["night_detection"] = hourly["occupied"] & (hour >= night_hours[0]) & (hour < night_hours[1])
    return hourly


def build_hourly_table(
    events: pd.DataFrame,
    occupied_threshold: int = 1,
    max_gap: str = "24h",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Run the full cleaning pipeline: clean, aggregate to hours, flag.

    Parameters
    ----------
    events : pandas.DataFrame
        Raw events for one space, as accepted by :func:`clean_events`.
    occupied_threshold : int, default 1
        Passed to :func:`events_to_hourly`.
    max_gap : str, default "24h"
        Passed to :func:`flag_outages`.

    Returns
    -------
    hourly : pandas.DataFrame
        Hourly table with columns ``n_events``, ``max_headcount``,
        ``occupied``, ``outage`` and ``night_detection``.
    log : dict of str to int
        Rows removed per event rule (see :func:`clean_events`) plus hours
        flagged by ``OCC-GAP-01`` and ``OCC-NIGHT-01``.
    """
    clean, log = clean_events(events)
    hourly = events_to_hourly(clean, occupied_threshold=occupied_threshold)
    hourly = flag_outages(hourly, clean, max_gap=max_gap)
    hourly = flag_night_detections(hourly)
    log["OCC-GAP-01"] = int(hourly["outage"].sum())
    log["OCC-NIGHT-01"] = int(hourly["night_detection"].sum())
    return hourly, log
