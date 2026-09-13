"""Loading and basic validation of raw sensor CSV files.

Functions in this module are the single entry point for reading the
Monash Smart Infrastructure occupancy and environmental CSV exports into
pandas DataFrames with a validated, sorted, timezone-naive datetime index.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_sensor_csv(
    filepath: str | Path,
    timestamp_column: str = "timestamp",
    required_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load a single sensor/occupancy CSV export into a validated DataFrame.

    Parameters
    ----------
    filepath : str or Path
        Path to a CSV file exported from the sensor platform (e.g. one
        room/zone environmental log or occupancy log).
    timestamp_column : str, default "timestamp"
        Name of the column holding the measurement timestamp. It is parsed
        to ``datetime64[ns]``, set as the index, and used to sort rows.
    required_columns : list of str, optional
        Column names (excluding ``timestamp_column``) that must be present.
        If any are missing a ``ValueError`` is raised, since a computational
        method cannot be evaluated on data with ambiguous inputs.

    Returns
    -------
    pandas.DataFrame
        The loaded data, indexed by ``timestamp_column`` in ascending order,
        with duplicate timestamps removed (first occurrence kept).

    Raises
    ------
    FileNotFoundError
        If ``filepath`` does not exist.
    ValueError
        If ``timestamp_column`` or any ``required_columns`` are missing from
        the file, or if no rows have a parseable timestamp.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Sensor file not found: {path}")

    df = pd.read_csv(path)

    if timestamp_column not in df.columns:
        raise ValueError(
            f"Expected timestamp column '{timestamp_column}' not found in "
            f"{path.name}. Available columns: {list(df.columns)}"
        )

    missing = [c for c in (required_columns or []) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {path.name}")

    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce")
    n_unparsed = df[timestamp_column].isna().sum()
    if n_unparsed == len(df):
        raise ValueError(f"No parseable timestamps found in column '{timestamp_column}'")

    df = df.dropna(subset=[timestamp_column])
    df = df.drop_duplicates(subset=[timestamp_column], keep="first")
    df = df.set_index(timestamp_column).sort_index()

    return df
