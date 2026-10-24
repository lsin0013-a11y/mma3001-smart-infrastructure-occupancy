"""Tests for smart_infra.academic_calendar."""

import pandas as pd
import pytest

from smart_infra.academic_calendar import calendar_features, load_public_holidays, load_teaching_periods

PERIODS = pd.DataFrame({
    "period": ["teaching", "mid_semester_break"],
    "start": pd.to_datetime(["2024-03-04", "2024-04-01"]),
    "end": pd.to_datetime(["2024-03-31", "2024-04-05"]),
})
HOLIDAYS = pd.DatetimeIndex(["2024-04-01"])


def test_calendar_features_tags_periods_holidays_and_time():
    index = pd.DatetimeIndex(["2024-03-04 09:30", "2024-04-01 14:05", "2024-04-06 23:55"], tz="Australia/Melbourne")

    features = calendar_features(index, PERIODS, HOLIDAYS)

    assert list(features["period"]) == ["teaching", "mid_semester_break", "non_semester"]
    assert list(features["public_holiday"]) == [False, True, False]
    assert list(features["minute_of_day"]) == [570, 845, 1435]
    assert list(features["weekend"]) == [False, False, True]


def test_calendar_features_uses_local_date_not_utc():
    # 00:30 local on 1 April is still 31 March in UTC.
    index = pd.DatetimeIndex(["2024-04-01 00:30"], tz="Australia/Melbourne")

    assert calendar_features(index, PERIODS, HOLIDAYS)["public_holiday"].iloc[0]


def test_calendar_features_rejects_naive_index():
    with pytest.raises(TypeError):
        calendar_features(pd.DatetimeIndex(["2024-03-04 09:30"]), PERIODS, HOLIDAYS)


def test_shipped_calendar_files_are_valid():
    periods = load_teaching_periods()
    holidays = load_public_holidays()

    assert periods["start"].min() <= pd.Timestamp("2024-03-03")  # covers the G.20 data
    assert pd.Timestamp("2025-12-25") in holidays


def test_load_teaching_periods_rejects_overlap(tmp_path):
    path = tmp_path / "periods.csv"
    path.write_text("period,start,end\nteaching,2024-03-01,2024-03-10\nexams,2024-03-10,2024-03-20\n")

    with pytest.raises(ValueError):
        load_teaching_periods(path)
