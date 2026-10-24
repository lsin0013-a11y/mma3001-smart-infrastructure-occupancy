# Forecasting Room Occupancy for Demand-Based HVAC Scheduling

Individual project for **MMA3001: Numerical Methods and Machine Learning** (Monash University), using **Dataset 2: Monash Smart Infrastructure Occupancy and Environmental Data**.

## Status

| Step | State |
|---|---|
| 1. Raw-data audit ([`reports/data_audit_findings.md`](reports/data_audit_findings.md)) | Done |
| 2. Problem definition (this README; draft report sections 1–2) | Done |
| 3. Clean occupied/empty tables for room G.20 at 10 min, 30 min and 1 h ([`reports/g20_occupancy_summary.md`](reports/g20_occupancy_summary.md)) | Done |
| 4. Calendar features (teaching weeks, breaks, public holidays) | Planned |
| 5. Baseline (time-of-week profile) and models (logistic regression, tree-based) | Planned |
| 6. Validation on withheld 2026 data, HVAC schedule comparison, sensitivity analysis | Planned |

## Engineering problem

HVAC (heating, ventilation and air-conditioning) is one of the largest energy uses in university buildings, and most systems run on fixed timetables (for example Monday–Friday, 07:00–19:00) whether or not rooms are in use. Conditioning an empty room wastes energy; switching off while a room is in use leaves occupants uncomfortable. If the building management system knew in advance when a room would be occupied, it could pre-heat or pre-cool only when needed.

**Problem: predict whether room G.20 (Building 60, Monash Clayton) will be occupied in each time interval (10 min, 30 min or 1 h), from calendar information alone, so its HVAC can be scheduled on demand instead of on a fixed timetable.**

Why this scope:

- **G.20 is one of only two occupancy spaces the metadata maps to a real room**, has about two years of data (March 2024 – April 2026), and is the busiest space in the dataset. Room G.25 may be used at the end to test whether the method generalises.
- **Occupancy data only.** The raw-data audit found the environmental data can't support this problem: 51% of it comes from frozen sensors, the two live sensors are live in windows that don't overlap, and no environmental sensor can be placed in G.20.
- **Fixed intervals, yes/no.** The resolution is a setting (10 min, 30 min or 1 h), compared as part of the optimisation work. A binary output is simpler to model, validate and explain than headcount.

### Inputs and outputs

| | Item | Format / domain |
|---|---|---|
| Input | Hour of day | integer, 0–23 (Australia/Melbourne local time) |
| Input | Day of week | Monday–Sunday |
| Input | Teaching period | teaching week / exam period / break (Monash academic calendar) |
| Input | Public holiday | true / false (Victorian public holidays) |
| Input (optional) | Occupied in the same interval last week | true / false |
| Output | Occupancy probability | 0–1 per interval |
| Output | Occupancy prediction | true / false per interval, used as the HVAC on/off schedule |
| Output | Schedule comparison | hours conditioned while empty, and occupied hours missed, compared with a fixed schedule |

**Ground truth:** an interval is *occupied* if G.20's headcount is 1 or more at any time in that interval.

### Limitations of the scope

- One room; binary output; intervals no finer than 10 min.
- Calendar inputs only, so the model can't anticipate one-off events or timetable changes.
- An empty room and a sensor outage both produce no data. Long weekday silences are flagged as likely outages, but some missing data will unavoidably be labelled as empty.
- Savings are estimated in hours of conditioning, not kWh: the dataset has no HVAC capacity, room volume or ventilation data.
- The sensor-to-room mapping comes from a metadata sheet marked "To be confirmed" by the data provider.

## Repository structure

```
.
├── src/smart_infra/       # installable Python package (source code)
│   ├── data_loader.py     # CSV loading + validation
│   ├── preprocessing.py   # gap detection, missing-value handling
│   ├── occupancy.py       # event stream -> clean occupied/empty table (10 min / 30 min / 1 h)
│   └── analysis.py        # summary statistics, occupancy/environment merging
├── tests/                  # pytest test suite
├── scripts/
│   ├── profile_raw_data.py        # reproducible raw-data audit (DuckDB)
│   └── build_occupancy_tables.py  # builds the occupancy tables for one room
├── reports/
│   ├── data_audit_findings.md  # raw-data audit: coverage, errors, suitability
│   ├── g20_occupancy_summary.md  # G.20 tables: coverage and cleaning counts per resolution
│   └── test_report.html   # generated pytest HTML test report
├── docs/                   # generated HTML API documentation (pdoc)
├── data/                   # not tracked in git — see data/README.md
├── notebooks/               # exploratory analysis notebooks
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

## Getting started

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # install smart_infra as an editable package
```

Place the raw dataset locally per [`data/README.md`](data/README.md) (the raw files are large and are not committed to this repository).

## Auditing the raw data

```bash
python scripts/profile_raw_data.py
```

This prints every figure quoted in [`reports/data_audit_findings.md`](reports/data_audit_findings.md). It reads the raw exports and never modifies them, and it caches parsed tables in `data/profile_cache.duckdb` (untracked).

## Building the occupancy tables

```bash
python scripts/build_occupancy_tables.py                     # G.20, all three resolutions
python scripts/build_occupancy_tables.py --resolution 10min  # one resolution only
```

This reads G.20's ~1.9 million raw events and writes one table per resolution (10 min, 30 min, 1 h) to `data/processed/g20_<resolution>.csv` (untracked), with one row per interval and columns:

| Column | Meaning |
|---|---|
| `interval_start` | start of the interval, Australia/Melbourne time with UTC offset |
| `n_events` | raw events in the interval (0 = no detections) |
| `max_headcount` | highest headcount in the interval |
| `occupied` | **target**: `max_headcount >= 1` |
| `outage` | interval lies inside a silence of more than 24 h covering weekday working hours, so it is more likely a sensor outage than an empty room (rule `OCC-GAP-01`); excluded from modelling |
| `night_detection` | occupied interval between 01:00 and 05:00 (rule `OCC-NIGHT-01`); kept, but tested in the sensitivity analysis |

It also writes [`reports/g20_occupancy_summary.md`](reports/g20_occupancy_summary.md): coverage and occupied share at each resolution, the number of rows or intervals affected by each cleaning rule, and the outage periods found. `--threshold` and `--max-gap` change the occupancy threshold and outage rule for sensitivity analysis.

**Why three resolutions:** a short visit marks a whole interval as occupied, so coarser intervals overstate how much the room is used. On G.20 the occupied share is 39.7% at 10 min, 48.3% at 30 min and 55.5% at 1 h. Comparing them is part of the optimisation work (the brief's "input resolution"): finer intervals describe use more accurately and suit ventilation control, while heating and cooling need 30–60 min to condition a room and calendar inputs can't predict the exact minutes someone arrives.

## Running the tests

```bash
pytest --html=reports/test_report.html --self-contained-html
```

The tests use small synthetic datasets rather than the raw data, so they run without the dataset. The generated report is at `reports/test_report.html`.

## Building the documentation

HTML API documentation is generated from the docstrings with [`pdoc`](https://pdoc.dev/):

```bash
pdoc smart_infra --output-dir docs -d numpy
```

Open `docs/smart_infra.html` in a browser to view it.

## Computational approach (planned)

1. **Pre-processing:** reduce G.20's ~2-second event stream to one row per interval (10 min, 30 min or 1 h; occupied or not), flagging likely sensor outages rather than treating them as empty.
2. **Baseline:** time-of-week profile, i.e. for each interval of the week (168 at 1 h, 1,008 at 10 min), the fraction of training weeks in which the room was occupied.
3. **Main model:** logistic regression on calendar features; chosen because it gives probabilities, interpretable coefficients and fast training.
4. **Alternative:** a decision tree or random forest, to test whether capturing interactions (e.g. Friday afternoons in exam periods) is worth the extra complexity.

## Validation (planned)

- **Chronological split:** train on 2024–2025, test on 2026 data the models never see. A random split would leak future information.
- **Metrics:** accuracy, F1 score for the occupied class, Brier score for probability quality.
- **Engineering evaluation:** compare a fixed Monday–Friday 07:00–19:00 schedule with the prediction-driven schedule, counting hours conditioned while empty and occupied hours missed.
- **Sensitivity analysis:** time resolution (10 min / 30 min / 1 h), the occupancy threshold, the gap length treated as an outage, and including or excluding night-time detections.

## AI use and reflection

AI assistance (Claude) has been used for: scaffolding the repository structure; drafting the initial data-loading/preprocessing/analysis utility functions and their docstrings; configuring the testing and documentation tooling (pytest, pdoc); profiling the raw dataset (writing the DuckDB audit script and drafting `reports/data_audit_findings.md`); proposing the problem framing; and drafting this README. All audit figures come from running that script on the raw files and can be regenerated. All generated code was reviewed and run locally (`pytest`, `pdoc`) before being committed. The choice of engineering problem, dataset interpretation and all analytical decisions remain the student's responsibility and will be detailed in the written report.

## License

Source code is released under the [MIT License](LICENSE). The underlying dataset is supplied by Monash University for MMA3001 coursework and is governed by the unit's own data usage terms.
