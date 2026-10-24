"""Turn the raw occupancy event stream into a clean occupied/empty time series.

The occupancy export is an event stream, not a regular time series: rows
arrive about every 2 s while a space is occupied and none are written while
it is empty (see ``reports/data_audit_findings.md`` §2). This module cleans
those events and reduces them to one row per fixed interval (2 min to 1 h; see
:data:`RESOLUTIONS`), which is the target series for the occupancy forecast.

Occupancy is treated as a *state*: each event's status holds until the next
event, for at most ``max_hold``. Counting only the instants at which events
arrive would make short intervals that fall between two reports look empty,
so the occupied share would keep falling towards zero as the resolution is
refined instead of converging.

Each cleaning rule carries the ID used in the audit's cleaning plan (§7).
Suspect intervals are flagged in their own column rather than deleted, so
every rule can be switched off for sensitivity analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LOCAL_TZ = "Australia/Melbourne"

RESOLUTIONS = ("2min", "5min", "10min", "30min", "1h")
"""Interval lengths the tables are built at. Any length that divides an hour
evenly is accepted (so intervals line up with clock hours); finer ones are used
only in the resolution convergence study."""

SPACE_IDS = {
    "G.20": "cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b",
    "G.25": "227f1068-9e41-469f-bef0-275f27a0bdba",
}
"""``floorspaceid`` of the two occupancy spaces the metadata sheet maps to a room."""

DEFAULT_MAX_HOLD = "5min"
"""Longest time an event's status is assumed to hold without a new event.
On G.20, 99.9% of gaps between consecutive occupied events are under 340 s."""

DEFAULT_CAPACITY = 20
"""Most people G.20 can physically hold: about 30 m2 (the student's estimate)
at 1.5 m2 per person. Higher readings are treated as sensor errors."""

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
    """Return ``resolution`` as a Timedelta, or raise if it doesn't divide an hour evenly."""
    try:
        step = pd.Timedelta(resolution)
    except ValueError:
        raise ValueError(f"Unparseable resolution '{resolution}'") from None
    if step <= pd.Timedelta(0) or pd.Timedelta("1h") % step != pd.Timedelta(0):
        raise ValueError(f"Resolution '{resolution}' must divide one hour evenly, e.g. {RESOLUTIONS}")
    return step


def _to_ns(timestamps: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Return timezone-aware timestamps as int64 nanoseconds since the epoch (UTC)."""
    utc = pd.DatetimeIndex(timestamps).tz_convert("UTC").tz_localize(None).as_unit("ns")
    return utc.to_numpy().view("int64")


def events_to_intervals(
    events: pd.DataFrame,
    resolution: str = "1h",
    occupied_threshold: int = 1,
    max_hold: str = DEFAULT_MAX_HOLD,
    capacity: int | None = None,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Reduce cleaned events to one row per fixed interval.

    Each event's headcount is held from its timestamp until the next event,
    or for at most ``max_hold`` if the next event is further away. The room is
    occupied whenever the held headcount is at least ``occupied_threshold``.

    Parameters
    ----------
    events : pandas.DataFrame
        Output of :func:`clean_events`.
    resolution : str, default "1h"
        Interval length; must divide one hour evenly (see :data:`RESOLUTIONS`).
    occupied_threshold : int, default 1
        Smallest headcount that counts as occupied.
    max_hold : str, default "5min"
        Longest time (pandas timedelta string) an event's status is held.
    capacity : int, optional
        Most people the room can physically hold (rule ``OCC-CAP-01``).
        Headcounts above it are treated as sensor errors: they are capped at
        ``capacity`` and the interval is flagged in ``over_capacity``.
        No cap if ``None``.
    start, end : pandas.Timestamp, optional
        Timezone-aware first and last interval to include. Default to the
        intervals of the first and last event.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``interval_start`` (Australia/Melbourne time, every interval
        from ``start`` to ``end`` inclusive) with columns:

        - ``n_events``: number of events in the interval (0 if none).
        - ``max_headcount``: highest headcount held at any time in the interval.
        - ``mean_headcount``: time-weighted average headcount over the
          interval (person-seconds per second), the quantity ventilation
          depends on. Like ``occupied_fraction``, its mean over a period does
          not depend on the resolution.
        - ``occupied_fraction``: share of the interval, 0 to 1, during which
          the room was occupied. Its mean over a period does not depend on the
          resolution.
        - ``occupied``: ``occupied_fraction > 0``, i.e. occupied at any time
          in the interval.
        - ``over_capacity`` (only if ``capacity`` is given): the uncapped
          headcount exceeded ``capacity`` at some time in the interval.

        Time with no event held is treated as empty. Use :func:`flag_outages`
        to mark intervals that are more likely sensor outages.

    Raises
    ------
    ValueError
        If ``events`` is empty, ``resolution`` is not supported, or
        ``occupied_threshold`` is below 1.
    """
    step = _check_resolution(resolution)
    if events.empty:
        raise ValueError("No events to aggregate")
    if occupied_threshold < 1:
        raise ValueError("occupied_threshold must be at least 1")

    # Melbourne's UTC offset is a whole number of hours, so UTC interval
    # boundaries are also local ones. Working in UTC avoids the repeated hour
    # when daylight saving ends.
    interval = events["timestamp"].dt.tz_convert("UTC").dt.floor(resolution)
    first = interval.min() if start is None else start.tz_convert("UTC").floor(resolution)
    last = interval.max() if end is None else end.tz_convert("UTC").floor(resolution)
    index = pd.date_range(first, last, freq=resolution)

    # Hold each event's headcount until the next event, capped at max_hold.
    t = _to_ns(events["timestamp"])
    raw_headcount = events["headcount"].to_numpy()
    headcount = raw_headcount if capacity is None else np.minimum(raw_headcount, capacity)
    hold_end = np.minimum(np.append(t[1:], np.iinfo(np.int64).max), t + pd.Timedelta(max_hold).value)
    spell_len = hold_end - t

    bounds = _to_ns(index)
    occupied = headcount >= occupied_threshold
    occupied_ns = _integrate(t[occupied], spell_len[occupied], np.ones(occupied.sum(), dtype=np.int64),
                             bounds, step.value)
    person_ns = _integrate(t, spell_len, headcount.astype(float), bounds, step.value)

    # Headcount held over from an event before the interval started.
    j = np.searchsorted(t, bounds, side="left") - 1
    jj = np.maximum(j, 0)
    held = (j >= 0) & (bounds < hold_end[jj])

    grouped = pd.DataFrame({"headcount": headcount, "raw": raw_headcount}, index=events.index).groupby(interval)
    table = grouped.agg(n_events=("headcount", "size"), max_headcount=("headcount", "max"),
                        raw_max=("raw", "max"))
    table = table.reindex(index, fill_value=0)
    table["max_headcount"] = np.maximum(table["max_headcount"].to_numpy(), np.where(held, headcount[jj], 0))
    table["mean_headcount"] = person_ns / step.value
    table["occupied_fraction"] = occupied_ns / step.value
    table["occupied"] = occupied_ns > 0
    if capacity is not None:
        raw_max = np.maximum(table["raw_max"].to_numpy(), np.where(held, raw_headcount[jj], 0))
        table["over_capacity"] = raw_max > capacity
    table = table.drop(columns="raw_max")
    table.index = index.tz_convert(LOCAL_TZ).rename("interval_start")
    return table


def _integrate(starts: np.ndarray, lengths: np.ndarray, weights: np.ndarray,
               bounds: np.ndarray, step: int) -> np.ndarray:
    """Integrate a piecewise-constant signal over each interval ``[bound, bound + step)``.

    The signal equals ``weights[i]`` on ``[starts[i], starts[i] + lengths[i])``
    and 0 elsewhere; the pieces must be sorted and must not overlap. With
    G(x) = integral of the signal up to x, the integral over [a, b) is
    G(b) - G(a). G at any x is a cumulative sum up to the last piece starting
    at or before x, plus the covered part of that piece, so every boundary is
    evaluated with one binary search (O(log n)) instead of a scan.
    """
    if len(starts) == 0:
        return np.zeros(len(bounds), dtype=weights.dtype)
    cumulative = np.concatenate([[0], np.cumsum(lengths * weights)])

    def integral_before(x: np.ndarray) -> np.ndarray:
        k = np.searchsorted(starts, x, side="right") - 1
        kk = np.maximum(k, 0)
        partial = np.minimum(x - starts[kk], lengths[kk]) * weights[kk]
        return np.where(k >= 0, cumulative[kk] + partial, 0)

    return integral_before(bounds + step) - integral_before(bounds)


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
    resolution : str, default "1h"
        The resolution ``table`` was built with.
    max_gap : str, default "24h"
        Shortest gap (pandas timedelta string) that can count as an outage.
    working_hours : tuple of int, default (9, 17)
        Local start (inclusive) and end (exclusive) hour of the working day.

    Returns
    -------
    pandas.DataFrame
        A copy of ``table`` with a boolean ``outage`` column. Only unoccupied
        intervals lying entirely inside a flagged gap are marked, so they never
        contain events.
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
            # An interval just after the gap starts can still hold the last
            # event's status (see max_hold); it is real data, not an outage.
            table.loc[inside & ~table["occupied"], "outage"] = True

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
    max_hold: str = DEFAULT_MAX_HOLD,
    capacity: int | None = DEFAULT_CAPACITY,
    max_gap: str = "24h",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Run the full cleaning pipeline: clean, aggregate to intervals, flag.

    Parameters
    ----------
    events : pandas.DataFrame
        Raw events for one space, as accepted by :func:`clean_events`.
    resolution : str, default "1h"
        Interval length, passed to :func:`events_to_intervals`.
    occupied_threshold : int, default 1
        Passed to :func:`events_to_intervals`.
    max_hold : str, default "5min"
        Passed to :func:`events_to_intervals`.
    capacity : int or None, default 20
        Passed to :func:`events_to_intervals`; ``None`` disables ``OCC-CAP-01``.
    max_gap : str, default "24h"
        Passed to :func:`flag_outages`.

    Returns
    -------
    table : pandas.DataFrame
        Table with columns ``n_events``, ``max_headcount``, ``mean_headcount``,
        ``occupied_fraction``, ``occupied``, ``over_capacity`` (if
        ``capacity`` is set), ``outage`` and ``night_detection``.
    log : dict of str to int
        Rows removed per event rule (see :func:`clean_events`) plus intervals
        flagged by ``OCC-CAP-01``, ``OCC-GAP-01`` and ``OCC-NIGHT-01``.
    """
    clean, log = clean_events(events)
    table = events_to_intervals(clean, resolution=resolution, occupied_threshold=occupied_threshold,
                                max_hold=max_hold, capacity=capacity)
    table = flag_outages(table, clean, resolution=resolution, max_gap=max_gap)
    table = flag_night_detections(table)
    log["OCC-CAP-01"] = int(table["over_capacity"].sum()) if capacity is not None else 0
    log["OCC-GAP-01"] = int(table["outage"].sum())
    log["OCC-NIGHT-01"] = int(table["night_detection"].sum())
    return table, log


def read_table(path: str) -> pd.DataFrame:
    """Read an occupancy table written by ``scripts/build_occupancy_tables.py``.

    Parameters
    ----------
    path : str or Path
        CSV with an ``interval_start`` column holding times with UTC offsets.

    Returns
    -------
    pandas.DataFrame
        The table indexed by ``interval_start`` in Australia/Melbourne time.
    """
    table = pd.read_csv(path)
    table.index = pd.DatetimeIndex(pd.to_datetime(table.pop("interval_start"), utc=True)).tz_convert(LOCAL_TZ)
    table.index.name = "interval_start"
    return table
