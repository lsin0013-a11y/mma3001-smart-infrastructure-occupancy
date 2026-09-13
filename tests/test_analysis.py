"""Tests for smart_infra.analysis."""

import pandas as pd
import pytest

from smart_infra.analysis import merge_occupancy_environment, summarize_readings


def test_summarize_readings_overall():
    df = pd.DataFrame({"co2_ppm": [400, 420, 440]})

    summary = summarize_readings(df, value_column="co2_ppm")

    assert summary.loc["all", "count"] == 3
    assert summary.loc["all", "mean"] == pytest.approx(420.0)


def test_summarize_readings_grouped():
    df = pd.DataFrame(
        {
            "room_id": ["A", "A", "B"],
            "co2_ppm": [400, 420, 500],
        }
    )

    summary = summarize_readings(df, value_column="co2_ppm", group_column="room_id")

    assert summary.loc["A", "mean"] == pytest.approx(410.0)
    assert summary.loc["B", "count"] == 1


def test_summarize_readings_missing_column_raises():
    df = pd.DataFrame({"co2_ppm": [400]})

    with pytest.raises(KeyError):
        summarize_readings(df, value_column="temperature_c")


def test_merge_occupancy_environment_matches_within_tolerance():
    occupancy = pd.DataFrame(
        {"occupied": [1, 0]},
        index=pd.DatetimeIndex(["2024-05-01 00:00", "2024-05-01 00:10"], name="timestamp"),
    )
    environment = pd.DataFrame(
        {"co2_ppm": [450, 460]},
        index=pd.DatetimeIndex(["2024-05-01 00:02", "2024-05-01 00:09"], name="timestamp"),
    )

    merged = merge_occupancy_environment(occupancy, environment, tolerance="5min")

    assert merged.loc["2024-05-01 00:00", "co2_ppm"] == 450
    assert merged.loc["2024-05-01 00:10", "co2_ppm"] == 460


def test_merge_occupancy_environment_requires_datetime_index():
    occupancy = pd.DataFrame({"occupied": [1]})
    environment = pd.DataFrame(
        {"co2_ppm": [450]}, index=pd.DatetimeIndex(["2024-05-01 00:00"])
    )

    with pytest.raises(TypeError):
        merge_occupancy_environment(occupancy, environment)
