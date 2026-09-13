"""Tests for smart_infra.data_loader."""

import pandas as pd
import pytest

from smart_infra.data_loader import load_sensor_csv


@pytest.fixture
def sample_csv(tmp_path):
    csv_path = tmp_path / "env_sample.csv"
    csv_path.write_text(
        "timestamp,room_id,co2_ppm,temperature_c\n"
        "2024-05-01 00:00:00,69_G57,450,21.5\n"
        "2024-05-01 00:05:00,69_G57,452,21.6\n"
        "2024-05-01 00:05:00,69_G57,999,99.9\n"  # duplicate timestamp, should be dropped
        "2024-05-01 00:10:00,69_G57,455,21.7\n"
    )
    return csv_path


def test_load_sensor_csv_sorts_and_indexes_by_timestamp(sample_csv):
    df = load_sensor_csv(sample_csv)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.index) == sorted(df.index)


def test_load_sensor_csv_drops_duplicate_timestamps(sample_csv):
    df = load_sensor_csv(sample_csv)

    assert len(df) == 3
    assert df.loc["2024-05-01 00:05:00", "co2_ppm"] == 452


def test_load_sensor_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sensor_csv(tmp_path / "does_not_exist.csv")


def test_load_sensor_csv_missing_timestamp_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("time,co2_ppm\n2024-05-01 00:00:00,450\n")

    with pytest.raises(ValueError, match="timestamp"):
        load_sensor_csv(csv_path)


def test_load_sensor_csv_missing_required_column_raises(sample_csv):
    with pytest.raises(ValueError, match="humidity_pct"):
        load_sensor_csv(sample_csv, required_columns=["co2_ppm", "humidity_pct"])
