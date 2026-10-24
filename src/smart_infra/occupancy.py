"""Turn the raw occupancy event stream into a clean occupied/empty time series.

The occupancy export is an event stream, not a regular time series: rows
arrive about every 2 s while a space is occupied and none are written while
it is empty (see ``reports/data_audit_findings.md`` §2). This module cleans
those events and reduces them to one row per fixed interval (10 min, 30 min or
1 h; see :data:`RESOLUTIONS`), which is the target series for the occupancy
forecast.

Each cleaning rule carries the ID used in the audit's cleaning plan (§7).
Suspect intervals are flagged in their own column rather than deleted, so
every rule can be switched off for sensitivity analysis.
"""

from __future__ import annotations

import pandas as pd

LOCAL_TZ = "Australia/Melbourne"

RESOLUTIONS = ("10min", "30min", "1h")
"""Supported interval lengths. Each divides an hour evenly, so intervals
always line up with clock hours."""

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
        because intervals would be ambiguous across daylight-saving changes.
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


def _check_resolution(resolution: str) -> pd.Timedelta:
    """Return ``resolution`` as a Timedelta, or raise if it is not supported."""
    if resolution not in RESOLUTIONS:
        raise ValueError(f"Unsupported resolution '{resolution}'. Use one of {RESOLUTIONS}.")
    return pd.Timedelta(resolution)


def events_to_intervals(
    events: pd.DataFrame,
    resolution: str = "1h",
    occupied_threshold: int = 1,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Reduce cleaned events to one row per fixed interval.

    Parameters
    ----------
    events : pandas.DataFrame
        Output of :func:`clean_events`.
    resolution : {"10min", "30min", "1h"}, default "1h"
        Interval length.
    occupied_threshold : int, default 1
        An interval is occupied if its maximum headcount is at least this value.
    start, end : pandas.Timestamp, optional
        Timezone-aware first and last interval to include. Default to the
        intervals of the first and last event.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``interval_start`` (Australia/Melbourne time, every interval
        from ``start`` to ``end`` inclusive) with columns:

        - ``n_events``: number of events in the interval (0 if none).
        - ``max_headcount``: highest headcount in the interval (0 if none).
        - ``occupied``: ``max_headcount >= occupied_threshold``.

        Intervals with no events are treated as empty. Use
        :func:`flag_outages` to mark those that are more likely sensor outages.

    Raises
    ------
    ValueError
        If ``events`` is empty, ``resolution`` is not supported, or
        ``occupied_threshold`` is below 1.
    """
    _check_resolution(resolution)
    if events.empty:
        raise ValueError("No events to aggregate")
    if occupied_threshold < 1:
        raise ValueError("occupied_threshold must be at least 1")

    # Melbourne's UTC offset is a whole number of hours, so UTC interval
    # boundaries are also local ones. Working in UTC avoids the repeated hour
    # when daylight saving ends.
    interval = events["timestamp"].dt.tz_convert("UTC").dt.floor(resolution)
    table = events.groupby(interval)["headcount"].agg(n_events="size", max_headcount="max")

    first = interval.min() if start is None else start.tz_convert("UTC").floor(resolution)
    last = interval.max() if end is None else end.tz_convert("UTC").floor(resolution)
    index = pd.date_range(first, last, freq=resolution)
    table = table.reindex(index, fill_value=0)

    table["occupied"] = table["max_headcount"] >= occupied_threshold
    table.index = index.tz_convert(LOCAL_TZ).rename("interval_start")
    return table


def flag_outages(
    table: pd.DataFrame,
    events: pd.DataFrame,
    resolution: str = "1h",
    max_gap: str = "24h",
    working_hours: tuple[int, int] = (9, 17),
) -> pd.DataFrame:
    """Flag intervals that fall in a likely sensor outage (rule ``OCC-GAP-01``).

    A gap between consecutive events is treated as a likely outage when it is
    longer than ``max_gap`` and covers at least one weekday working hour. A
    busy room going silent for a whole working day is more likely a sensor or
    network fault than an empty room, while a long silence over a weekend is
    plausible. Holiday closures are also flagged, since they can't be told
    apart from outages without a calendar.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`events_to_intervals`.
    events : pandas.DataFrame
        The cleaned events ``table`` was built from.
    resolution : {"10min", "30min", "1h"}, default "1h"
        The resolution ``table`` was built with.
    max_gap : str, default "24h"
        Shortest gap (pandas timedelta string) that can count as an outage.
    working_hours : tuple of int, default (9, 17)
        Local start (inclusive) and end (exclusive) hour of the working day.

    Returns
    -------
    pandas.DataFrame
        A copy of ``table`` with a boolean ``outage`` column. Only intervals
        lying entirely inside a flagged gap are marked, so they never contain
        events.
    """
    step = _check_resolution(resolution)
    table = table.copy()
    table["outage"] = False

    idx = table.index
    t = events["timestamp"].sort_values()
    gaps = t.diff()
    long_gaps = gaps > pd.Timedelta(max_gap)
    for gap_end, gap in zip(t[long_gaps], gaps[long_gaps]):
        gap_start = gap_end - gap
        # Decide from the silence itself, minute by minute, so the decision is
        # the same at every resolution.
        minutes = pd.date_range(gap_start.ceil("min"), gap_end, freq="min", inclusive="left").tz_convert(LOCAL_TZ)
        covers_working = ((minutes.dayofweek < 5) & (minutes.hour >= working_hours[0])
                          & (minutes.hour < working_hours[1])).any()
        if covers_working:
            # Strictly after the gap's first event, so that event's own interval is excluded.
            inside = (idx > gap_start) & (idx + step <= gap_end)
            table.loc[inside, "outage"] = True

    return table


def flag_night_detections(table: pd.DataFrame, night_hours: tuple[int, int] = (1, 5)) -> pd.DataFrame:
    """Flag occupied intervals in the middle of the night (rule ``OCC-NIGHT-01``).

    The audit found detections between 01:00 and 05:00 on many nights. They
    may be false detections or genuine late-night use, so they are flagged for
    sensitivity analysis rather than removed.

    Parameters
    ----------
    table : pandas.DataFrame
        Output of :func:`events_to_intervals` (or :func:`flag_outages`).
    night_hours : tuple of int, default (1, 5)
        Local start (inclusive) and end (exclusive) hour of the night window.

    Returns
    -------
    pandas.DataFrame
        A copy of ``table`` with a boolean ``night_detection`` column.
    """
    table = table.copy()
    hour = table.index.hour
    table["night_detection"] = table["occupied"] & (hour >= night_hours[0]) & (hour < night_hours[1])
    return table


def build_occupancy_table(
    events: pd.DataFrame,
    resolution: str = "1h",
    occupied_threshold: int = 1,
    max_gap: str = "24h",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Run the full cleaning pipeline: clean, aggregate to intervals, flag.

    Parameters
    ----------
    events : pandas.DataFrame
        Raw events for one space, as accepted by :func:`clean_events`.
    resolution : {"10min", "30min", "1h"}, default "1h"
        Interval length, passed to :func:`events_to_intervals`.
    occupied_threshold : int, default 1
        Passed to :func:`events_to_intervals`.
    max_gap : str, default "24h"
        Passed to :func:`flag_outages`.

    Returns
    -------
    table : pandas.DataFrame
        Table with columns ``n_events``, ``max_headcount``, ``occupied``,
        ``outage`` and ``night_detection``.
    log : dict of str to int
        Rows removed per event rule (see :func:`clean_events`) plus intervals
        flagged by ``OCC-GAP-01`` and ``OCC-NIGHT-01``.
    """
    clean, log = clean_events(events)
    table = events_to_intervals(clean, resolution=resolution, occupied_threshold=occupied_threshold)
    table = flag_outages(table, clean, resolution=resolution, max_gap=max_gap)
    table = flag_night_detections(table)
    log["OCC-GAP-01"] = int(table["outage"].sum())
    log["OCC-NIGHT-01"] = int(table["night_detection"].sum())
    return table, log
