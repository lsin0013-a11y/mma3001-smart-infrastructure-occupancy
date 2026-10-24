"""Compare ventilation strategies for one room over the 2026 test period.

The true occupancy is rebuilt at 1-min resolution from the raw events. Each
strategy sets the outdoor-air supply once per control interval (5 min, 10 min,
30 min or 1 h), and the room's CO2 is simulated minute by minute with the
exact solution of the mass balance in :mod:`smart_infra.ventilation`.

Strategies (supply = rate per person x headcount used):

- ``fixed_schedule``: designed for full capacity (20 people), Mon-Fri 07:00-19:00;
- ``oracle``: the interval's actual mean headcount (perfect knowledge; the
  least air that meets the per-person rate, not achievable in practice);
- ``live_sensor``: the previous interval's measured headcount (reacts after
  the fact, one interval late);
- ``persistence_30min``: the headcount 30 min earlier;
- ``forecast``: gradient-boosting forecast issued 30 min ahead;
- ``forecast_calendar_only``: the same without recent headcounts;
- ``forecast_q90``: 90th-percentile forecast (errs towards more air);
- ``hybrid``: the larger of the live-sensor headcount and the forecast, so
  the forecast can raise the air ahead of expected arrivals.

Every strategy also gets background infiltration (0.3 air changes per hour,
an assumption), which is counted in the CO2 simulation but not as supplied air.

Writes ``reports/ventilation_results.csv``, ``reports/ventilation_results.md``
and two figures in ``reports/figures/``.

Usage (from the repository root, after ``run_forecast.py``)::

    python scripts/ventilation_study.py [--room G.20] [--datasets Datasets]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from build_occupancy_tables import OCC_CSV, load_space_events  # noqa: E402
from smart_infra.occupancy import DEFAULT_CAPACITY, SPACE_IDS, build_occupancy_table, read_table  # noqa: E402
from smart_infra.ventilation import (  # noqa: E402
    CO2_LIMIT,
    INFILTRATION_ACH,
    RATE_PER_PERSON,
    assess,
    infiltration_airflow,
    simulate_co2,
    steady_state_co2,
)  # noqa: E402

CONTROL_RESOLUTIONS = ["5min", "10min", "30min", "1h"]
TEST_START = "2026-01-01"
DT = 60.0

# Default chart palette (dataviz skill) and ink.
COLORS = {"fixed_schedule": "#52514e", "oracle": "#1baf7a", "live_sensor": "#eda100",
          "forecast": "#2a78d6", "forecast_q90": "#eb6834", "hybrid": "#6250d6"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {"fixed_schedule": "Fixed schedule (20 people, 07-19)", "oracle": "Oracle (actual headcount)",
          "live_sensor": "Live sensor (previous interval)", "persistence_30min": "Persistence (30 min old)",
          "forecast": "Forecast (mean, 30 min ahead)", "forecast_calendar_only": "Forecast (calendar only)",
          "forecast_q90": "Forecast (90th percentile)", "hybrid": "Hybrid (max of live sensor, forecast)"}


def control_headcounts(stem: str, res: str) -> pd.DataFrame:
    """Headcount each strategy acts on, per control interval of the test period."""
    forecasts = read_table(Path("data/processed") / f"{stem}_forecast_{res}.csv")
    table = read_table(Path("data/processed") / f"{stem}_{res}.csv")
    measured = table["mean_headcount"].where(~table["outage"])
    return pd.DataFrame({
        "oracle": forecasts["observed"],
        "live_sensor": measured.shift(1).reindex(forecasts.index),
        "persistence_30min": forecasts["persistence_baseline"],
        "forecast": forecasts["boosting"],
        "forecast_calendar_only": forecasts["boosting_calendar_only"],
        "forecast_q90": forecasts["boosting_q90"],
        "hybrid": np.maximum(measured.shift(1).reindex(forecasts.index), forecasts["boosting"]),
    })


def fixed_schedule(index: pd.DatetimeIndex) -> np.ndarray:
    """Design airflow for full capacity on weekdays 07:00-19:00, off otherwise."""
    on = (index.dayofweek < 5) & (index.hour >= 7) & (index.hour < 19)
    return np.where(on, RATE_PER_PERSON * DEFAULT_CAPACITY, 0.0)


def run(minute: pd.DataFrame, stem: str) -> tuple[pd.DataFrame, dict]:
    """Simulate every strategy at every control resolution."""
    people = minute["mean_headcount"].where(~minute["outage"], 0.0).to_numpy()
    rows, traces = [], {}
    fixed = fixed_schedule(minute.index)
    for res in CONTROL_RESOLUTIONS:
        heads = control_headcounts(stem, res)
        # Only minutes whose control interval has a valid forecast are scored.
        # Floor in UTC: the local clock repeats an hour when daylight saving ends.
        slot = minute.index.tz_convert("UTC").floor(res).tz_convert(minute.index.tz)
        valid = slot.isin(heads.dropna().index) & ~minute["outage"].to_numpy()
        strategies = {name: (RATE_PER_PERSON * heads[name].clip(lower=0)).reindex(slot).fillna(0.0).to_numpy()
                      for name in heads.columns}
        strategies = {"fixed_schedule": fixed, **strategies}
        for name, airflow in strategies.items():
            co2 = simulate_co2(people, airflow + infiltration_airflow(), DT)
            rows.append({"control": res, "strategy": name, **assess(people, airflow, co2, DT, mask=valid)})
            if res == "5min":
                traces[name] = co2
    return pd.DataFrame(rows), {"people": people, "co2": traces, "index": minute.index}


def plot_tradeoff(results: pd.DataFrame, path: Path) -> None:
    """Outdoor air supplied vs person-hours above the CO2 limit, 5-min control."""
    r = results[results["control"] == "5min"].set_index("strategy")
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    markers = {"fixed_schedule": "s", "oracle": "D", "live_sensor": "^", "persistence_30min": "v",
               "forecast": "o", "forecast_calendar_only": "P", "forecast_q90": "o", "hybrid": "X"}
    for name, row in r.iterrows():
        color = COLORS.get(name, "#9e9d98")
        ax.scatter(row["air_m3"] / 1000, row["person_hours_over_limit"], s=70, marker=markers[name],
                   color=color, edgecolor="white", linewidth=1.2, zorder=3,
                   label=f"{LABELS[name]}: {row['air_m3'] / 1000:.0f}k m³, {row['person_hours_over_limit']:.0f} p-h")
    ax.set_xlabel("Outdoor air supplied over test period (thousand m³)", color=MUTED)
    ax.set_ylabel(f"Person-hours above {CO2_LIMIT:.0f} ppm", color=MUTED)
    ax.set_title("Ventilation strategies, 5-min control: air used vs air-quality shortfall",
                 color=INK, fontsize=11, loc="left")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-30)
    # Eight strategies, several close together near the origin: a legend reads
    # better than direct labels.
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK, handletextpad=0.4)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")


def plot_day(traces: dict, path: Path) -> None:
    """CO2 through the busiest test-period weekday under four strategies."""
    index = traces["index"]
    people = pd.Series(traces["people"], index=index)
    # Busiest weekday with no capped (likely erroneous) readings.
    daily = people.groupby(index.normalize())
    clean_days = daily.max()[daily.max() < DEFAULT_CAPACITY].index
    totals = daily.sum()
    day = totals[totals.index.isin(clean_days) & (totals.index.dayofweek < 5)].idxmax()
    m = (index >= day) & (index < day + pd.Timedelta("1D"))
    hours = (index[m] - day).total_seconds() / 3600

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.0, 5.0), dpi=200, sharex=True,
                                      gridspec_kw={"height_ratios": [1, 2.2]})
    top.fill_between(hours, people[m], step="post", color="#86b6ef", lw=0)
    top.set_ylabel("People", color=MUTED)
    top.set_title(f"G.20 on {day:%a %d %b %Y}: occupancy and simulated CO2", color=INK, fontsize=11, loc="left")
    for name in ("fixed_schedule", "live_sensor", "forecast", "forecast_q90"):
        bottom.plot(hours, traces["co2"][name][m], color=COLORS[name], lw=1.6, label=LABELS[name])
    bottom.axhline(CO2_LIMIT, color=MUTED, lw=1, ls=(0, (4, 3)))
    bottom.annotate(f"{CO2_LIMIT:.0f} ppm limit", (0.3, CO2_LIMIT), xytext=(0, 4), textcoords="offset points",
                    color=MUTED, fontsize=8)
    bottom.set_ylabel("CO2 (ppm)", color=MUTED)
    bottom.set_xlabel("Hour of day", color=MUTED)
    bottom.set_xticks(range(0, 25, 3))
    bottom.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK)
    for ax in (top, bottom):
        style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")


def style(ax) -> None:
    """Recessive grid and axes."""
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)


def write_markdown(results: pd.DataFrame, path: Path, room: str) -> None:
    """Results table per control resolution, relative to the fixed schedule."""
    lines = [
        f"# {room} ventilation strategies: 2026 test period",
        "",
        f"Outdoor air at {RATE_PER_PERSON:.0f} L/s per person; CO2 simulated minute by minute (well-mixed room, "
        f"81 m³, 0.0052 L/s CO2 per person, 420 ppm outdoors; steady state at the design rate "
        f"{steady_state_co2():.0f} ppm; background infiltration {INFILTRATION_ACH} air changes per hour). Sensor outages excluded from scoring. *Air vs fixed* is outdoor air "
        "supplied relative to the fixed schedule (a proxy for fan and conditioning energy).",
        "",
        f"Generated by `python scripts/ventilation_study.py --room {room}`.",
        "",
    ]
    fixed_air = results.loc[results["strategy"] == "fixed_schedule", "air_m3"].iloc[0]
    for res, group in results.groupby("control", sort=False):
        lines += [f"## Control every {res}", "",
                  "| Strategy | Air (m³) | Air vs fixed | Person-hours > limit | Occupied time > limit | "
                  "Mean CO2 occupied (ppm) | Peak (ppm) |", "|---|---|---|---|---|---|---|"]
        for r in group.itertuples():
            lines.append(f"| {LABELS[r.strategy]} | {r.air_m3:,.0f} | {r.air_m3 / fixed_air:.0%} | "
                         f"{r.person_hours_over_limit:,.1f} | {r.share_occupied_time_over_limit:.1%} | "
                         f"{r.mean_occupied_ppm:.0f} | {r.peak_ppm:.0f} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--room", default="G.20", choices=sorted(SPACE_IDS))
    parser.add_argument("--datasets", type=Path, default=Path("Datasets"))
    args = parser.parse_args()
    stem = args.room.replace(".", "").lower()

    events = load_space_events(args.datasets / OCC_CSV, SPACE_IDS[args.room])
    minute, _ = build_occupancy_table(events, resolution="1min")
    minute = minute[minute.index >= pd.Timestamp(TEST_START, tz=minute.index.tz)]
    results, traces = run(minute, stem)

    results.to_csv("reports/ventilation_results.csv", index=False, float_format="%.4f")
    write_markdown(results, Path("reports/ventilation_results.md"), args.room)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plot_tradeoff(results, Path("reports/figures/ventilation_tradeoff.png"))
    plot_day(traces, Path("reports/figures/ventilation_day.png"))
    print(results.round(3).to_string())


if __name__ == "__main__":
    main()
