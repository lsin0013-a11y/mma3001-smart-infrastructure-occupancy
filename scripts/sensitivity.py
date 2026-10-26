"""Sensitivity of the results to each assumption, changed one at a time.

For every variant the whole chain is rerun at 5-min resolution: occupancy
table -> forecast (gradient boosting with the tuned settings, plus
persistence) -> CO2 simulation of the main strategies over the 2026 test
period. The base case uses the defaults; each row changes one setting.

Writes ``reports/sensitivity.csv`` and ``reports/sensitivity.md``.

Usage (from the repository root)::

    python scripts/sensitivity.py [--room G.20] [--datasets Datasets]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from build_occupancy_tables import OCC_CSV, load_space_events
from smart_infra.forecasting import TARGET, evaluate, make_dataset, make_models, split_by_date
from smart_infra.occupancy import DEFAULT_CAPACITY, SPACE_IDS, build_occupancy_table
from smart_infra.ventilation import (
    CO2_GENERATION,
    CO2_LIMIT,
    INFILTRATION_ACH,
    RATE_PER_PERSON,
    ROOM_AREA,
    ROOM_HEIGHT,
    assess,
    design_occupancy,
    infiltration_airflow,
    simulate_co2,
)

BOOSTING = {"learning_rate": 0.05, "max_leaf_nodes": 15, "max_iter": 300}  # from run_forecast.py tuning (5 min)
BASE = {"max_hold": "5min", "capacity": DEFAULT_CAPACITY, "max_gap": "24h", "drop_night": False,
        "rate": RATE_PER_PERSON, "ach": INFILTRATION_ACH, "generation": CO2_GENERATION,
        "area": ROOM_AREA, "limit": CO2_LIMIT}
VARIANTS = [
    ("Base case", {}),
    ("Hold limit 2 min", {"max_hold": "2min"}),
    ("Hold limit 10 min", {"max_hold": "10min"}),
    ("Hold limit uncapped", {"max_hold": "24h"}),
    ("Capacity cap 20 (first room estimate)", {"capacity": 20}),
    ("Outage gap 12 h", {"max_gap": "12h"}),
    ("Outage gap 48 h", {"max_gap": "48h"}),
    ("Night detections removed", {"drop_night": True}),
    ("7.5 L/s per person", {"rate": 7.5}),
    ("12.5 L/s per person", {"rate": 12.5}),
    ("No infiltration", {"ach": 0.0}),
    ("Infiltration 1.0 ACH", {"ach": 1.0}),
    ("CO2 generation -20%", {"generation": CO2_GENERATION * 0.8}),
    ("CO2 generation +20%", {"generation": CO2_GENERATION * 1.2}),
    ("Room 200 m2", {"area": 200.0}),
    ("Room 400 m2", {"area": 400.0}),
    ("Room 30 m2 (first estimate)", {"area": 30.0}),
    ("Limit 800 ppm", {"limit": 800.0}),
]


def table_for(events: pd.DataFrame, s: dict, resolution: str) -> pd.DataFrame:
    """Occupancy table under the labelling settings in ``s``."""
    table, _ = build_occupancy_table(events, resolution=resolution, max_hold=s["max_hold"],
                                     capacity=s["capacity"], max_gap=s["max_gap"])
    if s["drop_night"]:
        # Treat 01:00-05:00 detections as false: no one present.
        night = (table.index.hour >= 1) & (table.index.hour < 5)
        table.loc[night, ["mean_headcount", "occupied_fraction"]] = 0.0
    return table


def run_variant(events: pd.DataFrame, s: dict) -> dict:
    table = table_for(events, s, "5min")
    usable = table[~table["outage"]]
    train, test = split_by_date(make_dataset(table))
    X_train, y_train, X_test, y_test = (train.drop(columns=TARGET), train[TARGET],
                                        test.drop(columns=TARGET), test[TARGET])
    models = make_models(BOOSTING)
    forecast = pd.Series(models["boosting"].fit(X_train, y_train).predict(X_test), index=test.index)
    acc = evaluate(y_test, forecast.to_numpy())
    persist = evaluate(y_test, models["persistence_baseline"].predict(X_test))

    # True occupancy at 1-min resolution; the fixed schedule is sized for the
    # design occupancy of this variant's training period.
    full = table_for(events, s, "1min")
    before = full[(full.index < test.index.min()) & ~full["outage"]]
    people_design = design_occupancy(before["mean_headcount"].to_numpy())
    minute = full[full.index >= test.index.min()]
    people = minute["mean_headcount"].where(~minute["outage"], 0.0).to_numpy()
    slot = minute.index.tz_convert("UTC").floor("5min").tz_convert(minute.index.tz)
    valid = slot.isin(test.index) & ~minute["outage"].to_numpy()
    measured = table["mean_headcount"].where(~table["outage"])
    live = measured.shift(1).reindex(test.index)
    heads = {"live_sensor": live, "hybrid": np.maximum(live, forecast)}
    on = (minute.index.dayofweek < 5) & (minute.index.hour >= 7) & (minute.index.hour < 19)
    volume = s["area"] * ROOM_HEIGHT
    leak = infiltration_airflow(s["ach"], volume)
    row = {
        "time_occupied": usable["occupied_fraction"].mean(),
        "mean_headcount": usable["mean_headcount"].mean(),
        "forecast_mae": acc["mae"],
        "persistence_mae": persist["mae"],
    }
    row["design_people"] = people_design
    supplies = {"fixed": np.where(on, s["rate"] * people_design, 0.0),
                **{k: (s["rate"] * v.clip(lower=0)).reindex(slot).fillna(0.0).to_numpy() for k, v in heads.items()}}
    for name, airflow in supplies.items():
        co2 = simulate_co2(people, airflow + leak, 60.0, volume=volume, generation=s["generation"])
        result = assess(people, airflow, co2, 60.0, limit=s["limit"], mask=valid)
        row[f"{name}_air_m3"] = result["air_m3"]
        row[f"{name}_ph_over"] = result["person_hours_over_limit"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--room", default="G.20", choices=sorted(SPACE_IDS))
    parser.add_argument("--datasets", type=Path, default=Path("Datasets"))
    args = parser.parse_args()

    events = load_space_events(args.datasets / OCC_CSV, SPACE_IDS[args.room])
    rows = []
    for name, change in VARIANTS:
        rows.append({"variant": name, **run_variant(events, {**BASE, **change})})
        r = rows[-1]
        print(f"{name:<28} time occ {r['time_occupied']:.1%}  MAE {r['forecast_mae']:.3f}  "
              f"live {r['live_sensor_ph_over']:.0f} p-h  hybrid {r['hybrid_ph_over']:.0f} p-h  "
              f"fixed {r['fixed_ph_over']:.0f} p-h", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv("reports/sensitivity.csv", index=False, float_format="%.4f")

    lines = [
        f"# {args.room} sensitivity analysis",
        "",
        "Each row changes one assumption from the base case and reruns the whole chain at 5-min resolution "
        "(occupancy table, gradient-boosting forecast, CO2 simulation over the 2026 test period). "
        "*p-h* = person-hours above the CO2 limit. Air is outdoor air supplied, thousand m³.",
        "",
        "Generated by `python scripts/sensitivity.py`.",
        "",
        "| Variant | Time occupied | Mean headcount | Forecast MAE | Persistence MAE | Design people | "
        "Fixed: air / p-h | Live sensor: air / p-h | Hybrid: air / p-h |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results.itertuples():
        lines.append(
            f"| {r.variant} | {r.time_occupied:.1%} | {r.mean_headcount:.3f} | {r.forecast_mae:.3f} | "
            f"{r.persistence_mae:.3f} | {r.design_people} | {r.fixed_air_m3 / 1e3:.0f}k / {r.fixed_ph_over:.0f} | "
            f"{r.live_sensor_air_m3 / 1e3:.0f}k / {r.live_sensor_ph_over:.0f} | "
            f"{r.hybrid_air_m3 / 1e3:.0f}k / {r.hybrid_ph_over:.0f} |")
    Path("reports/sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote reports/sensitivity.csv and reports/sensitivity.md")


if __name__ == "__main__":
    main()
