"""Tests for smart_infra.occupancy."""

import pandas as pd
import pytest

from smart_infra.occupancy import (
    build_occupancy_table,
    clean_events,
    events_to_intervals,
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


def test_events_to_intervals_fills_empty_hours_at_1h():
    events = make_events([
        (1, "2024-05-01 09:15:00", "CurrentlyOccupied", 2),
        (2, "2024-05-01 09:15:02", "CurrentlyOccupied", 3),
        (3, "2024-05-01 11:30:00", "RecentlyOccupied", 0),
    ])

    table = events_to_intervals(events)

    assert list(table.index.hour) == [9, 10, 11]
    assert list(table["n_events"]) == [2, 0, 1]
    assert list(table["max_headcount"]) == [3, 0, 0]
    assert list(table["occupied"]) == [True, False, False]


def test_events_to_intervals_at_10min_splits_the_hour():
    events = make_events([
        (1, "2024-05-01 09:05:00", "CurrentlyOccupied", 2),
        (2, "2024-05-01 09:45:00", "CurrentlyOccupied", 1),
    ])

    table = events_to_intervals(events, resolution="10min")

    assert list(table.index.minute) == [0, 10, 20, 30, 40]
    assert list(table["occupied"]) == [True, False, False, False, True]


def test_events_to_intervals_same_events_at_each_resolution():
    # One short visit: the whole hour is occupied at 1h, but only 1 of 2
    # half-hours and 1 of 6 ten-minute intervals.
    events = make_events([
        (1, "2024-05-01 09:05:00", "CurrentlyOccupied", 1),
        (2, "2024-05-01 09:55:00", "RecentlyOccupied", 0),
    ])

    shares = {r: events_to_intervals(events, resolution=r)["occupied"].mean() for r in ("1h", "30min", "10min")}

    assert shares == {"1h": 1.0, "30min": 0.5, "10min": pytest.approx(1 / 6)}


def test_events_to_intervals_holds_state_between_sparse_events():
    # Reports 5 min apart: the 09:02 interval has no event of its own but the
    # room is still occupied, with the headcount carried over.
    events = make_events([
        (1, "2024-05-01 09:00:00", "CurrentlyOccupied", 2),
        (2, "2024-05-01 09:05:00", "CurrentlyOccupied", 2),
        (3, "2024-05-01 09:06:00", "RecentlyOccupied", 0),
    ])

    table = events_to_intervals(events, resolution="2min")

    assert list(table["n_events"]) == [1, 0, 1, 1]
    assert list(table["occupied"]) == [True, True, True, False]
    assert list(table["max_headcount"]) == [2, 2, 2, 0]
    assert list(table["occupied_fraction"]) == [1.0, 1.0, 1.0, 0.0]


def test_events_to_intervals_stops_holding_after_max_hold():
    # A lone occupied report followed by 20 min of silence counts as occupied
    # for max_hold (5 min) only.
    events = make_events([
        (1, "2024-05-01 09:00:00", "CurrentlyOccupied", 1),
        (2, "2024-05-01 09:20:00", "RecentlyOccupied", 0),
    ])

    table = events_to_intervals(events, resolution="5min")

    assert list(table["occupied"]) == [True, False, False, False, False]
    assert table["occupied_fraction"].iloc[0] == 1.0


def test_occupied_share_converges_as_resolution_is_refined():
    # Two visits within one hour. The share of intervals marked occupied must
    # fall (or stay level) as intervals shrink, and never drop below the
    # time-weighted share, which is the same at every resolution.
    events = make_events([
        (1, "2024-05-01 09:01:00", "CurrentlyOccupied", 1),
        (2, "2024-05-01 09:04:00", "RecentlyOccupied", 0),
        (3, "2024-05-01 09:33:00", "CurrentlyOccupied", 1),
        (4, "2024-05-01 09:36:00", "CurrentlyOccupied", 1),
        (5, "2024-05-01 09:41:00", "RecentlyOccupied", 0),
        (6, "2024-05-01 09:58:00", "NotOccupied", 0),
    ])
    tables = {r: events_to_intervals(events, resolution=r) for r in ("1h", "30min", "10min", "5min", "2min")}
    share = [t["occupied"].mean() for t in tables.values()]
    weighted = [t["occupied_fraction"].mean() for t in tables.values()]

    assert share == sorted(share, reverse=True)
    assert all(w == pytest.approx(11 / 60) for w in weighted)  # 3 + 8 occupied minutes
    assert share[-1] >= weighted[-1]


def test_events_to_intervals_rejects_unsupported_resolution():
    events = make_events([(1, "2024-05-01 09:15:00", "CurrentlyOccupied", 1)])

    with pytest.raises(ValueError):
        events_to_intervals(events, resolution="7min")


def test_events_to_intervals_applies_threshold():
    events = make_events([(1, "2024-05-01 09:15:00", "CurrentlyOccupied", 1)])

    assert not events_to_intervals(events, occupied_threshold=2)["occupied"].iloc[0]
    with pytest.raises(ValueError):
        events_to_intervals(events, occupied_threshold=0)


def test_events_to_intervals_keeps_repeated_hour_when_daylight_saving_ends():
    # Clocks go back from 03:00 AEDT to 02:00 AEST on 7 April 2024, so local
    # 02:00-03:00 happens twice and the day has 25 hours.
    events = pd.DataFrame({
        "id": [1, 2],
        "timestamp": pd.to_datetime(["2024-04-06 13:00", "2024-04-07 13:00"]).tz_localize("UTC").tz_convert(TZ),
        "status": "CurrentlyOccupied",
        "headcount": [1, 1],
    })

    table = events_to_intervals(events)

    assert len(table) == 25
    assert table.index.is_unique
    assert list(table.index.hour).count(2) == 2


def test_events_to_intervals_rejects_empty_input():
    with pytest.raises(ValueError):
        events_to_intervals(make_events([]))


def test_flag_outages_flags_long_weekday_gap():
    # Tuesday 18:00 to Thursday 08:00 covers all of Wednesday's working day.
    events = make_events([
        (1, "2024-05-07 18:00:00", "CurrentlyOccupied", 1),
        (2, "2024-05-09 08:00:00", "CurrentlyOccupied", 1),
    ])
    table = flag_outages(events_to_intervals(events), events)

    assert table["outage"].sum() == 37  # 19:00 Tue to 07:00 Thu
    assert not table["outage"].iloc[0] and not table["outage"].iloc[-1]


def test_flag_outages_at_30min_excludes_event_intervals():
    events = make_events([
        (1, "2024-05-07 18:00:00", "CurrentlyOccupied", 1),
        (2, "2024-05-09 08:00:00", "CurrentlyOccupied", 1),
    ])
    table = flag_outages(events_to_intervals(events, resolution="30min"), events, resolution="30min")

    assert table["outage"].sum() == 75  # 18:30 Tue to 07:30 Thu: 11 + 48 + 16
    assert not table["outage"].iloc[0] and not table["outage"].iloc[-1]


@pytest.mark.parametrize("resolution", ["10min", "30min", "1h"])
def test_flag_outages_decision_does_not_depend_on_resolution(resolution):
    # Friday 16:10 to Saturday 22:00 covers only 50 min of working time, all
    # inside the hour of the first event. It must still be flagged at 1h.
    events = make_events([
        (1, "2024-05-10 16:10:00", "CurrentlyOccupied", 1),
        (2, "2024-05-11 22:00:00", "CurrentlyOccupied", 1),
    ])
    table = flag_outages(events_to_intervals(events, resolution=resolution), events, resolution=resolution)

    assert table["outage"].any()


def test_flag_outages_ignores_weekend_gap():
    # Friday 19:00 to Sunday 20:00: long, but no weekday working hours.
    events = make_events([
        (1, "2024-05-10 19:00:00", "CurrentlyOccupied", 1),
        (2, "2024-05-12 20:00:00", "CurrentlyOccupied", 1),
    ])

    assert not flag_outages(events_to_intervals(events), events)["outage"].any()


def test_flag_outages_ignores_short_gap():
    events = make_events([
        (1, "2024-05-07 08:00:00", "CurrentlyOccupied", 1),
        (2, "2024-05-07 18:00:00", "CurrentlyOccupied", 1),
    ])

    assert not flag_outages(events_to_intervals(events), events)["outage"].any()


def test_flag_night_detections_marks_only_occupied_night_hours():
    events = make_events([
        (1, "2024-05-01 02:10:00", "CurrentlyOccupied", 1),
        (2, "2024-05-01 03:10:00", "RecentlyOccupied", 0),
        (3, "2024-05-01 10:10:00", "CurrentlyOccupied", 1),
    ])
    table = flag_night_detections(events_to_intervals(events))

    assert list(table.index[table["night_detection"]].hour) == [2]


def test_build_occupancy_table_reports_every_rule():
    events = make_events([
        (1, "2024-05-01 10:00:00", "CurrentlyOccupied", 1),
        (2, "2024-05-01 10:00:00", "CurrentlyOccupied", 1),
        (3, "2024-05-01 12:00:00", "CurrentlyOccupied", 2),
    ])

    table, log = build_occupancy_table(events)

    assert set(log) == {"OCC-RANGE-01", "OCC-DUP-01", "OCC-DUP-02", "OCC-GAP-01", "OCC-NIGHT-01"}
    assert log["OCC-DUP-01"] == 1
    assert list(table.columns) == [
        "n_events", "max_headcount", "occupied_fraction", "occupied", "outage", "night_detection",
    ]
