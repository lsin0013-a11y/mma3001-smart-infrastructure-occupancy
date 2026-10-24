"""Calendar inputs for the occupancy forecast: semester periods and public holidays.

Room use in a university building follows the academic calendar much more
than the weather, so each interval is tagged with where it falls in the
semester and whether it is a public holiday. The dates themselves are kept in
``data/calendar/`` (see the README there for sources).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

CALENDAR_DIR = Path(__file__).resolve().parents[2] / "data" / "calendar"

PERIODS = ("teaching", "mid_semester_break", "swot_vac", "exams", "non_semester")
"""Calendar periods. ``non_semester`` covers every date outside a listed period
(summer and winter breaks, and the gap between semesters)."""


def load_teaching_periods(path: str | Path = CALENDAR_DIR / "monash_teaching_periods.csv") -> pd.DataFrame:
    """Load semester periods, one row per continuous period.

    Parameters
    ----------
    path : str or Path
        CSV with columns ``period``, ``start`` and ``end`` (dates, inclusive).

    Returns
    -------
    pandas.DataFrame
        Columns ``period``, ``start`` and ``end`` (``datetime64``), sorted by
        ``start``.

    Raises
    ------
    ValueError
        If a period name is not in :data:`PERIODS`, a period ends before it
        starts, or two periods overlap.
    """
    df = pd.read_csv(path, parse_dates=["start", "end"]).sort_values("start", ignore_index=True)
    unknown = set(df["period"]) - set(PERIODS)
    if unknown:
        raise ValueError(f"Unknown periods {sorted(unknown)}; expected {PERIODS}")
    if (df["end"] < df["start"]).any():
        raise ValueError("A period ends before it starts")
    if (df["start"].iloc[1:].to_numpy() <= df["end"].iloc[:-1].to_numpy()).any():
        raise ValueError("Teaching periods overlap")
    return df[["period", "start", "end"]]


def load_public_holidays(path: str | Path = CALENDAR_DIR / "vic_public_holidays.csv") -> pd.DatetimeIndex:
    """Load public holiday dates.

    Parameters
    ----------
    path : str or Path
        CSV with a ``date`` column.

    Returns
    -------
    pandas.DatetimeIndex
        Holiday dates (midnight, timezone-naive).
    """
    return pd.DatetimeIndex(pd.read_csv(path, parse_dates=["date"])["date"]).normalize()


def calendar_features(
    index: pd.DatetimeIndex,
    periods: pd.DataFrame | None = None,
    holidays: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Tag each interval with calendar information known in advance.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Timezone-aware interval starts (e.g. the index of an occupancy table).
        Features use the local date and time of each interval.
    periods : pandas.DataFrame, optional
        Output of :func:`load_teaching_periods`. Loaded from the default file
        if ``None``.
    holidays : pandas.DatetimeIndex, optional
        Output of :func:`load_public_holidays`. Loaded from the default file
        if ``None``.

    Returns
    -------
    pandas.DataFrame
        Same index, with columns:

        - ``minute_of_day``: minutes since local midnight (0 to 1439).
        - ``day_of_week``: 0 = Monday to 6 = Sunday.
        - ``weekend``: Saturday or Sunday.
        - ``public_holiday``: the local date is a public holiday.
        - ``period``: one of :data:`PERIODS` (categorical).

    Raises
    ------
    TypeError
        If ``index`` is not timezone-aware.
    """
    if not isinstance(index, pd.DatetimeIndex) or index.tz is None:
        raise TypeError("index must be a timezone-aware DatetimeIndex")
    periods = load_teaching_periods() if periods is None else periods
    holidays = load_public_holidays() if holidays is None else holidays

    date = index.tz_localize(None).normalize()
    period = pd.Series("non_semester", index=index, dtype=object)
    for row in periods.itertuples():
        period[(date >= row.start) & (date <= row.end)] = row.period

    return pd.DataFrame({
        "minute_of_day": index.hour * 60 + index.minute,
        "day_of_week": index.dayofweek,
        "weekend": index.dayofweek >= 5,
        "public_holiday": date.isin(holidays),
        "period": pd.Categorical(period, categories=PERIODS),
    }, index=index)
