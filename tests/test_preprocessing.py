"""Tests for smart_infra.preprocessing."""

import numpy as np
import pandas as pd
import pytest

from smart_infra.preprocessing import detect_missing_intervals, handle_missing_values


def make_series(timestamps, values):
    return pd.DataFrame({"co2_ppm": values}, index=pd.DatetimeIndex(timestamps, name="timestamp"))


def test_detect_missing_intervals_finds_single_gap():
    timestamps = [
        "2024-05-01 00:00",
        "2024-05-01 00:05",
        "2024-05-01 00:20",  # 15 min gap -> 2 missing 5-min samples
        "2024-05-01 00:25",
    ]
    df = make_series(timestamps, [1, 2, 3, 4])

    gaps = detect_missing_intervals(df, expected_freq="5min")

    assert len(gaps) == 1
    assert gaps.iloc[0]["missing_samples"] == 2


def test_detect_missing_intervals_no_gaps_returns_empty():
    timestamps = ["2024-05-01 00:00", "2024-05-01 00:05", "2024-05-01 00:10"]
    df = make_series(timestamps, [1, 2, 3])

    gaps = detect_missing_intervals(df, expected_freq="5min")

    assert gaps.empty


def test_detect_missing_intervals_requires_datetime_index():
    df = pd.DataFrame({"co2_ppm": [1, 2, 3]})

    with pytest.raises(TypeError):
        detect_missing_intervals(df, expected_freq="5min")


def test_handle_missing_values_interpolate_fills_gap():
    timestamps = ["2024-05-01 00:00", "2024-05-01 00:05", "2024-05-01 00:10"]
    df = make_series(timestamps, [1.0, np.nan, 3.0])

    filled = handle_missing_values(df, method="interpolate")

    assert filled["co2_ppm"].iloc[1] == pytest.approx(2.0)


def test_handle_missing_values_drop_removes_rows():
    timestamps = ["2024-05-01 00:00", "2024-05-01 00:05", "2024-05-01 00:10"]
    df = make_series(timestamps, [1.0, np.nan, 3.0])

    filled = handle_missing_values(df, method="drop")

    assert len(filled) == 2


def test_handle_missing_values_invalid_method_raises():
    df = make_series(["2024-05-01 00:00"], [1.0])

    with pytest.raises(ValueError):
        handle_missing_values(df, method="not_a_method")
