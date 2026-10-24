"""Tests for smart_infra.occupancy."""

import pandas as pd
import pytest

from smart_infra.occupancy import (
    build_hourly_table,
    clean_events,
    events_to_hourly,
    flag_night_detections,
    flag_outages,
)

TZ = "Australia/Melbourne"


def make_events(rows):
    """Build an events frame from (id, local time, status, headcount) tuples."""
    df = pd.DataFrame(rows, columns=["id", "timestamp", "status", "headcount"])
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(TZ)
    return df


def test_clean_events_drops_exact_duplicates():
    events = make_events([
        (1, "2024-05-01 10:00:00", "CurrentlyOccupied", 2),
        (2, "2024-05-01 10:00:00", "CurrentlyOccupied", 2),
        (3, "2024-05-01 10:00:02", "CurrentlyOccupied", 3),
    ])

    clean, log = clean_events(events)

    assert list(clean["id"]) == [1, 3]
    assert log["OCC-DUP-01"] == 1


def test_clean_events_keeps_latest_row_for_conflicting_timestamp():
    events = make_events([
        (7, "2024-05-01 10:00:00", "CurrentlyOccupied", 4),
        (5, "2024-05-01 10:00:00", "CurrentlyOccupied", 2),
    ])

    clean, log = clean_events(events)

    assert list(clean["id"]) == [7]
    assert log["OCC-DUP-02"] == 1


def test_clean_events_drops_negative_headcount():
    events = make_events([
        (1, "2024-05-01 10:00:00", "CurrentlyOccupied", -1),
        (2, "2024-05-01 10:00:02", "CurrentlyOccupied", 1),
    ])

    clean, log = clean_events(events)

    assert list(clean["id"]) == [2]
    assert log["OCC-RANGE-01"] == 1


def test_clean_events_rejects_naive_timestamps():
    events = make_events([(1, "2024-05-01 10:00:00", "CurrentlyOccupied", 1)])
    events["timestamp"] = events["timestamp"].dt.tz_localize(None)

    with pytest.raises(TypeError):
        clean_events(events)


def test_clean_events_rejects_missing_columns():
    with pytest.raises(ValueError):
        clean_events(pd.DataFrame({"id": [1]}))


def test_events_to_hourly_fills_empty_hours():
    events = make_events([
        (1, "2024-05-01 09:15:00", "CurrentlyOccupied", 2),
        (2, "2024-05-01 09:15:02", "CurrentlyOccupied", 3),
        (3, "2024-05-01 11:30:00", "RecentlyOccupied", 0),
    ])

    hourly = events_to_hourly(events)

    assert list(hourly.index.hour) == [9, 10, 11]
    assert list(hourly["n_events"]) == [2, 0, 1]
    assert list(hourly["max_headcount"]) == [3, 0, 0]
    assert list(hourly["occupied"]) == [True, False, False]


def test_events_to_hourly_applies_threshold():
    events = make_events([(1, "2024-05-01 09:15:00", "CurrentlyOccupied", 1)])

    assert not events_to_hourly(events, occupied_threshold=2)["occupied"].iloc[0]
    with pytest.raises(ValueError):
        events_to_hourly(events, occupied_threshold=0)


def test_events_to_hourly_keeps_repeated_hour_when_daylight_saving_ends():
    # Clocks go back from 03:00 AEDT to 02:00 AEST on 7 April 2024, so local
    # 02:00-03:00 happens twice and the day has 25 hours.
    events = pd.DataFrame({
        "id": [1, 2],
        "timestamp": pd.to_datetime(["2024-04-06 13:00", "2024-04-07 13:00"]).tz_localize("UTC").tz_convert(TZ),
        "status": "CurrentlyOccupied",
        "headcount": [1, 1],
    })

    hourly = events_to_hourly(events)

    assert len(hourly) == 25
    assert hourly.index.is_unique
    assert list(hourly.index.hour).count(2) == 2


def test_events_to_hourly_rejects_empty_input():
    with pytest.raises(ValueError):
        events_to_hourly(make_events([]))


def test_flag_outages_flags_long_weekday_gap():
    # Tuesday 18:00 to Thursday 08:00 covers all of Wednesday's working day.
    events = make_events([
        (1, "2024-05-07 18:00:00", "CurrentlyOccupied", 1),
        (2, "2024-05-09 08:00:00", "CurrentlyOccupied", 1),
    ])
    hourly = flag_outages(events_to_hourly(events), events)

    assert hourly["outage"].sum() == 37  # 19:00 Tue to 07:00 Thu
    assert not hourly["outage"].iloc[0] and not hourly["outage"].iloc[-1]


def test_flag_outages_ignores_weekend_gap():
    # Friday 19:00 to Sunday 20:00: long, but no weekday working hours.
    events = make_events([
        (1, "2024-05-10 19:00:00", "CurrentlyOccupied", 1),
        (2, "2024-05-12 20:00:00", "CurrentlyOccupied", 1),
    ])

    assert not flag_outages(events_to_hourly(events), events)["outage"].any()


def test_flag_outages_ignores_short_gap():
    events = make_events([
        (1, "2024-05-07 08:00:00", "CurrentlyOccupied", 1),
        (2, "2024-05-07 18:00:00", "CurrentlyOccupied", 1),
    ])

    assert not flag_outages(events_to_hourly(events), events)["outage"].any()


def test_flag_night_detections_marks_only_occupied_night_hours():
    events = make_events([
        (1, "2024-05-01 02:10:00", "CurrentlyOccupied", 1),
        (2, "2024-05-01 03:10:00", "RecentlyOccupied", 0),
        (3, "2024-05-01 10:10:00", "CurrentlyOccupied", 1),
    ])
    hourly = flag_night_detections(events_to_hourly(events))

    assert list(hourly.index[hourly["night_detection"]].hour) == [2]


def test_build_hourly_table_reports_every_rule():
    events = make_events([
        (1, "2024-05-01 10:00:00", "CurrentlyOccupied", 1),
        (2, "2024-05-01 10:00:00", "CurrentlyOccupied", 1),
        (3, "2024-05-01 12:00:00", "CurrentlyOccupied", 2),
    ])

    hourly, log = build_hourly_table(events)

    assert set(log) == {"OCC-RANGE-01", "OCC-DUP-01", "OCC-DUP-02", "OCC-GAP-01", "OCC-NIGHT-01"}
    assert log["OCC-DUP-01"] == 1
    assert list(hourly.columns) == ["n_events", "max_headcount", "occupied", "outage", "night_detection"]
