# Dataset 2 — Raw Data Audit Findings

**Date of audit:** 2026-10-03
**Scope:** the raw files supplied as Dataset 2 (Monash Smart Infrastructure Occupancy and Environmental Data), evaluated against the selected problem of **predicting room occupancy**.
**Reproduce:** every figure below is printed by `python scripts/profile_raw_data.py` (see [How to reproduce](#how-to-reproduce)).

| File | Rows | Notes |
|---|---|---|
| `5occupancySensor_MayToDec2024_9MRows.csv` | 9,091,732 | 1.6 GB, one row per occupancy event |
| `5EnvSensor_MayToDec2024_180kRows.csv` | 179,447 | readings nested in a `jsondata` JSON column |
| `Sensor ID and Locations.xlsx` | 78 env + 89 occupancy entries | occupancy sheet is titled *"OCC (To be confirmed)"* |

---

## 1. Summary

**Enough data to predict occupancy?** Yes, if occupancy is predicted from its own history plus calendar (and optionally weather) features. **No** for estimating occupancy from environmental readings taken in the same room, because:

1. 51% of environmental rows come from three **frozen sensors** that repeat one or two readings throughout the file.
2. The only two live environmental sensors are live in **windows that don't overlap**, and there are **no environmental rows at all from Nov 2024 to Aug 2025**.
3. **No environmental sensor can be placed in the same room as an occupancy space** using the supplied metadata.

**Recommended problem framing:** forecast each space's hourly headcount (or probability of being occupied) 1–24 h ahead, using past occupancy, the academic calendar and weather. Use the hour-of-week average profile as the baseline. This is the information needed for occupancy-based ventilation and for pre-heating/pre-cooling rooms (HVAC energy), which gives the mechanical-engineering relevance.

---

## 2. Occupancy data

### 2.1 Structure and meaning

- Columns: `id, deviceid, floorspaceid, occupancystatus, headcount, collecteddate, occupancystatuschangedate, previousoccupancystatus`.
- **One `deviceid`** (`56923747-…`) for all rows, which looks like a gateway. Spaces are identified by `floorspaceid`, and there are **5 of them**.
- **It is an event stream, not a regular time series.** Rows arrive about every 2 s while a space is occupied (median gap 2–4 s). An empty room produces no rows, so **a sensor outage can't be told apart from an empty room**. Any regular-interval series has to be built by carrying the last status forward, plus a rule for when a gap counts as an outage.
- Status and headcount are consistent: `CurrentlyOccupied` always has headcount ≥ 1, and `RecentlyOccupied` / `NotOccupied` always have headcount = 0.

| Status | Previous status | Rows |
|---|---|---|
| CurrentlyOccupied | RecentlyOccupied | 4,509,436 |
| CurrentlyOccupied | NotOccupied | 3,703,402 |
| RecentlyOccupied | CurrentlyOccupied | 420,133 |
| NotOccupied | RecentlyOccupied | 265,666 |
| CurrentlyOccupied | Unknown | 193,056 |
| NotOccupied | CurrentlyOccupied | 37 |
| NotOccupied | Unknown | 2 |

`previousoccupancystatus` is the status *before the last status change*, not the status in the previous row. It usually differs from the previous row's status, and that is expected. `Unknown` most likely marks a sensor restart.

### 2.2 Coverage per space

| floorspaceid | Room (metadata) | Rows | First | Last | Mean / max headcount |
|---|---|---|---|---|---|
| `0f3f37f8…` | **not in metadata** | 2,169,461 | 2023-11-24 | 2026-04-22 | 2.29 / 40 |
| `941f0352…` | **not in metadata** | 1,975,423 | 2023-11-24 | 2026-04-22 | 2.28 / 15 |
| `cca1ac3e…` | G.20 (Bldg 60) | 1,894,871 | 2024-03-03 | 2026-04-22 | 4.48 / 72 |
| `3e5a14df…` | **not in metadata** | 1,551,583 | 2024-05-01 | 2024-12-30 | 4.08 / 36 |
| `227f1068…` | G.25 (Bldg 60) | 1,500,394 | 2024-03-03 | 2026-04-22 | 3.05 / 38 |

- **The data covers far more than "May–Dec 2024".** It runs from 2023-11-24 to 2026-04-22 (about 2.4 years). Only `3e5a14df` matches the file name's window.
- Clear daily and weekly patterns. In G.20, mean headcount rises from about 1.5 at 03:00 to about 5 at 13:00–18:00, and Mon–Fri is clearly busier than the weekend.

### 2.3 Problems found

| Check | Result |
|---|---|
| Duplicate rows (identical apart from `id`) | 4 |
| Same space and timestamp appearing twice | 9 keys (18 rows) |
| Gaps > 3 days | `0f3f37f8` and `941f0352`: 2023-12-23 → 2024-01-08 (15.6 d); `0f3f37f8`: 2024-12-23 → 12-26 (3.1 d); `cca1ac3e`: 2025-12-24 → 12-27 (3.6 d). These match Christmas closures, but could also be outages. |
| Gaps > 1 h | 541–2,561 per space. Most are probably empty rooms, not outages. |
| Night-time detections (01:00–04:59, headcount > 0) | `941f0352`: 649 of 860 days; `227f1068`: 659/781; `0f3f37f8`: 581/861; `cca1ac3e`: 252/777; `3e5a14df`: 153/244. **Suspicious:** either false detections or genuinely 24-hour spaces. Needs confirming. |
| Headcount extremes | Maximum 72 (G.20, Sunday 2024-08-04, 11:14–11:16). The values change smoothly (54 → 63 → 69 …), which looks like a real event (possibly a campus open day), not an error. 99.9th percentile per space: 9–46. |
| Jumps of ≥ 10 people between consecutive rows | ≤ 0.003% of rows |
| Status changes straight from CurrentlyOccupied to NotOccupied (skipping RecentlyOccupied) | 37 rows |

---

## 3. Environmental data

### 3.1 Structure

- Each row has `sensorid`, `createdate` / `processdate` (when the platform polled the sensor) and `jsondata`, a list of variables, each with its own measurement `time` (Unix epoch) and `value`.
- Variables: CO2 (ppm), temperature (°C), humidity (%RH), pressure (mb), PM1 / PM2.5 / PM4 / PM10 (µg/m³), formaldehyde (µg/m³), light VOCs (ppb), battery (V).
- `status` is `Processed` for every row, and `processing_errors` is empty for every row.
- **Polling cadence:** about 640 s (`createdate`). **Sensor measurement cadence:** about 600 s (JSON `time`).
- **`createdate` is not the measurement time.** The platform re-stores the last value it holds on every poll, so the same measurement appears at least twice, and indefinitely once a sensor stops updating. The real timestamp is the JSON `time`, which can be months older than `createdate` (median lag 0–342 days depending on the sensor).

### 3.2 Staleness: only two sensors ever carry live readings

| sensorid | Room (metadata) | Rows | Distinct measurement times | Distinct readings | Status |
|---|---|---|---|---|---|
| 6012002000326 | **not in metadata** | 58,909 | 29,039 | 24,263 | Frozen Feb–Oct 2024 (one reading); **live 2025-09-07 → 2026-04-22** |
| 6012002000241 | **not in metadata** | 58,908 | 2 | 2 | **Frozen throughout** |
| 6012002000869 | G.38 Compactus_upper | 29,341 | 13,800 | 11,075 | **Live 2024-02-29 → 2024-06-14**, then frozen to Oct 2024 |
| 6012002000227 | **not in metadata** | 21,363 | 1 | 1 | **Frozen throughout** (CO2 666 ppm, 24.0 °C) |
| 6012002000777 | **not in metadata** | 10,926 | 1 | 1 | **Frozen throughout** (CO2 425 ppm, 18.8 °C, timestamp 2023-05-06) |

- Frozen sensors 227, 241 and 777 make up 91,197 rows, **51% of the file**.
- **No environmental rows exist from Nov 2024 to Aug 2025.**
- Within the "May–Dec 2024" window in the file names, the only live environmental data is sensor 869 from 1 May to 14 Jun 2024 (about 6 weeks).

### 3.3 Value ranges (all rows, frozen ones included)

| Variable | Min | P1 | Median | P99 | Max | Comment |
|---|---|---|---|---|---|---|
| CO2 (ppm) | 400 | 415 | 520 | 666 | 1387 | Hard floor at 400, which looks like the sensor's lower limit |
| Temperature (°C) | 11.8 | 15.5 | 23.8 | 27.5 | 37.5 | 37.5 °C is implausible indoors; check |
| Humidity (%RH) | 17 | 35 | 59 | 69 | 84 | |
| Pressure (mb) | 988 | 999 | 1012 | 1024 | 1028 | Plausible for the weather |
| Formaldehyde (µg/m³) | 1 | 1 | 5 | 55 | 55 | **Maxes out at 55**; only 66,067 non-null |
| Light VOCs (ppb) | 0 | 0 | 7 | 71 | 8051 | Extreme spike; check |
| PM10 (µg/m³) | 0 | 0 | 4 | 10 | 190 | Spikes possibly from workshop activity |
| Battery (V) | 3.97 | 4.03 | 4.19 | 4.19 | 4.25 | Healthy |

### 3.4 Does live CO2 follow occupancy?

Hourly mean CO2 was correlated with the hourly maximum headcount of each space. A *partial* r removes the hour-of-week average from both series, so the shared building schedule doesn't inflate the result.

| Sensor (live window) | Hourly CO2 profile | r with each space | Partial r |
|---|---|---|---|
| 869 (Mar–Jun 2024) | 454 ppm (07:00) → 499 ppm (12:00) | +0.46 to +0.68 | +0.23 to **+0.55 (`941f0352`)** |
| 326 (Sep 2025–Apr 2026) | 455 ppm (07:00) → 426 ppm (15:00), i.e. **falls during the day** | −0.16 to −0.21 | ≈ 0.00 |

- **869:** responds to building activity, and most strongly to `941f0352`. But its daily swing is only about 45 ppm, which points to a large or well-ventilated zone, and the metadata places it in G.38, where there is no occupancy sensor. This is weak evidence of co-location.
- **326:** no relationship with any space once the schedule is removed. Its CO2 falls in the daytime, which suggests a space ventilated during the day with no occupants. It is not co-located with any of the occupancy spaces.

---

## 4. Problems in the metadata sheet

- Occupancy sheet titled **"To be confirmed"**. Several IDs are marked `(?)`, one is "missing", one note is `#REF!`, and one ID is probably a typo (`9034-285a-5dcf` where `8034-…` is expected).
- The occupancy sheet lists device IDs (`8034-…`) that never appear in the data. The data can only be matched through `SpaceId` = `floorspaceid`, and only for 2 of the 5 spaces.
- The same `SpaceId` is given to rooms on different floors (e.g. `9306da1f…` for 1.65, the 1.66 corridor and G62; `2387067e…` for 1.39 and G.23).
- **The same env ID `6012003000691` is listed for both ES2G14 (G.52) and ES2G01 (G.61).**
- The env sheet lists sensors (`6012003…`) in the rooms that do have occupancy data (G.20: ES1G01–03; G.25: ES1G09–14), but **none of them are in the extract**. Four of the five extracted env sensors are not in the sheet.
- One env row has Building 69 with Floor/Room `69`.

---

## 5. Strengths and weaknesses for occupancy prediction

**Strengths**
- Measured headcount available to validate against, at about 2 s resolution.
- About 2.4 years of data, covering several semesters, breaks and two Christmas closures.
- Five spaces with different usage levels, so models can be compared across rooms and tested on rooms they weren't trained on.
- Status and headcount values are internally consistent; no nulls.
- When the env sensors are live, they measure a rich set of variables every 10 minutes.

**Weaknesses**
- Missing data and an empty room look the same (no rows either way).
- The file names don't describe the time span correctly.
- Environmental data is mostly frozen, duplicated, and timestamped by when it was polled, not when it was measured.
- There is no reliable link from sensors to rooms; the metadata is unconfirmed and inconsistent.
- No room floor areas, volumes, capacities or HVAC/ventilation information, all of which a CO2 mass-balance model needs.
- Sensor readings at their limits (CO2 floor, formaldehyde maximum), plus unexplained spikes.

---

## 6. Additional data needed

**From the teaching team (highest priority)**
- A confirmed list of which room each `floorspaceid` and each env `sensorid` is in.
- Environmental data from sensors actually in occupancy-monitored rooms (e.g. G.20 / G.25).
- Room floor areas, volumes and capacities; HVAC schedules or outside-air rates from the building management system.

**Public sources**

| Need | Source |
|---|---|
| Teaching weeks, exams, breaks | Monash University academic calendar (key dates) |
| Public holidays | Business Victoria — Victorian public holidays |
| Outdoor temperature, rain, solar | BoM Climate Data Online, Moorabbin Airport (086077). For hourly history: Open-Meteo historical API (ECMWF ERA5 reanalysis) |
| Outdoor CO2 baseline (for a mass-balance model) | CSIRO/BoM Cape Grim baseline CO2 record |
| Outdoor PM2.5 | EPA Victoria AirWatch |

Calendar features are expected to matter much more than weather for predicting occupancy.

---

## 7. Plan for traceable cleaning

1. **Raw files are read-only.** Record their SHA-256 hashes in `data/manifest.json` so the exact extract can be identified.
2. **Process in stages:** raw → parsed (typed and unpacked, values unchanged) → flagged → cleaned → features. Each stage is a script in `src/smart_infra/` and records the git commit and config version that produced it.
3. **Numbered rules in one config file.** Each rule has an ID, a description, a threshold and a justification. Examples:

   | Rule ID | Rule | Planned action |
   |---|---|---|
   | `ENV-STALE-01` | Measurement time hasn't advanced for more than 30 min | Flag as stale; exclude from modelling |
   | `ENV-DUP-01` | Repeated (sensor, measurement time) pair | Keep the first |
   | `ENV-RANGE-*` | Value at or beyond the sensor's limits (CO2 ≤ 400, formaldehyde = 55, temperature > 35 °C indoors, VOC spikes) | Flag as at-limit or outlier |
   | `OCC-DUP-01` | Rows identical apart from `id` | Drop duplicates |
   | `OCC-DUP-02` | Same space and timestamp, different values | Keep the last by `id`; log it |
   | `OCC-GAP-01` | No rows for more than N h during working hours on a teaching day | Flag as a possible outage; mark that period missing, not empty |
   | `OCC-NIGHT-01` | Headcount > 0 between 01:00 and 05:00 | Flag for review / sensitivity analysis |
   | `META-MAP-01` | ID not in the metadata, or matching more than one room | Flag as unmapped |

4. **Flag rather than delete.** A QC-flag column per row; the cleaning step applies actions based on the flags. Rules can be switched on or off for sensitivity analysis.
5. **Issues log and summary report.** Every flagged row goes to an issues log (rule ID, file, row ID, sensor, timestamp, value, action), and each pipeline run produces a summary report with counts per rule in `reports/`.
6. **Unit tests per rule** using small synthetic data with planted errors (pytest, following the existing tests).

---

## 8. Patterns and prediction

- The occupancy sensors measure occupancy directly, so daily and weekly patterns can be described **without** predicting anything (hour-of-week profiles, clustering of daily profiles, seasonal decomposition). Prediction would only be needed first to describe rooms that have env sensors but no occupancy sensors.
- The dependency runs the other way: describing the patterns comes first, because the hour-of-week profile is the baseline any prediction model has to beat.
- Environmental data mainly shows the *consequences* of occupancy (CO2 build-up, comfort, whether ventilation keeps up), not how rooms are used.

---

## How to reproduce

```bash
pip install duckdb openpyxl          # in addition to requirements.txt
python scripts/profile_raw_data.py   # expects the raw exports in ./Datasets (see data/README.md)
```

The first run builds `data/profile_cache.duckdb` (untracked, about 20 s); later runs reuse it (`--rebuild` forces a reload). Timestamps are reported in Australia/Melbourne local time.
