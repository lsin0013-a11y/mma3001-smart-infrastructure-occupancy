# Data

Raw sensor exports are **not tracked in this repository** (see `.gitignore`) because they are large (the occupancy export alone is about 1.6 GB) and are supplied separately by the unit teaching team.

## Dataset

**Monash Smart Infrastructure Occupancy and Environmental Data** (Project Brief, Dataset 2).

Place the extracted files locally, relative to the repository root, as:

```
Datasets/
├── 5occupancySensor_MayToDec2024_9MRows/
│   └── 5occupancySensor_MayToDec2024_9MRows.csv   # occupancy events (used by this project)
├── 5EnvSensor_MayToDec2024_180kRows/
│   └── 5EnvSensor_MayToDec2024_180kRows.csv       # environmental readings (audited, not used for modelling)
└── Sensor ID and Locations.xlsx                   # sensor/room metadata
```

Despite the file names, the occupancy data runs from 24 November 2023 to 22 April 2026.

## What this project uses

Only the **occupancy export**, filtered to room **G.20** (`floorspaceid` `cca1ac3e-f81f-4ebb-aa2b-8b0bf3e9e53b`). The relevant columns are:

| Column | Meaning |
|---|---|
| `id` | row identifier (increases with arrival order) |
| `floorspaceid` | the monitored space |
| `occupancystatus` | `CurrentlyOccupied`, `RecentlyOccupied` or `NotOccupied` |
| `headcount` | people detected (≥ 1 only when `CurrentlyOccupied`) |
| `collecteddate` | event time, with UTC offset (e.g. `2024-03-03 15:06:30+11`) |

The environmental export was audited but is excluded from modelling; see [`reports/data_audit_findings.md`](../reports/data_audit_findings.md) §3 for why.

## Notes on data quality

The occupancy export is an **event stream**: rows arrive about every 2 s while a space is occupied, and **no rows** are written while it is empty. A sensor outage therefore looks the same as an empty room. Duplicates, night-time detections and other issues found in the audit, and the cleaning rule for each, are listed in [`reports/data_audit_findings.md`](../reports/data_audit_findings.md) §2.3 and §7. Rows and hours are flagged rather than deleted, so each rule can be switched off for sensitivity analysis.
