# Data

Raw sensor exports are **not tracked in this repository** (see `.gitignore`) because they are large (tens to hundreds of MB) and are supplied separately by the unit teaching team.

## Dataset

**Monash Smart Infrastructure Occupancy and Environmental Data** (Project Brief, Dataset 2).

Place the extracted files locally as:

```
Datasets/
├── 5EnvSensor_MayToDec2024_180kRows/       # environmental sensor readings (CO2, PM1/PM2.5/PM10, temperature, humidity, pressure)
├── 5occupancySensor_MayToDec2024_9MRows/   # time-stamped occupancy readings
└── Sensor ID and Locations.xlsx            # sensor/room/zone metadata
```

`smart_infra.data_loader.load_sensor_csv` expects a `timestamp` column (or an equivalent passed via `timestamp_column`) in each CSV.

## Notes on data quality

Per the project brief, the raw exports contain **missing, irregular or potentially unreliable observations that require auditing** before use — see `smart_infra.preprocessing.detect_missing_intervals` and `handle_missing_values`. Do not assume the data is analysis-ready without first checking for gaps, duplicate timestamps, and out-of-range sensor values.
