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
- ``recent_max``: the larger of the last two measured intervals (no forecast);
- ``forecast``: gradient-boosting forecast issued 10 min ahead;
- ``hybrid``: the larger of the live-sensor headcount and the forecast;
- ``live_margin``: the live-sensor headcount plus a safety margin. The margin
  is chosen on 2025 (before the test period): the smallest in
  :data:`MARGINS` that cuts person-hours above the limit by at least
  :data:`MARGIN_TARGET` relative to no margin.

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
TUNE_START, TEST_START = "2025-01-01", "2026-01-01"
DT = 60.0
MARGINS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
MARGIN_TARGET = 0.9

# Default chart palette (dataviz skill) and ink.
COLORS = {"fixed_schedule": "#52514e", "oracle": "#1baf7a", "live_sensor": "#eda100", "recent_max": "#9e9d98",
          "forecast": "#2a78d6", "hybrid": "#6250d6", "live_margin": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABELS = {"fixed_schedule": "Fixed schedule (20 people, 07-19)", "oracle": "Oracle (actual headcount)",
          "live_sensor": "Live sensor (previous interval)", "recent_max": "Larger of last two readings",
          "forecast": "Forecast (10 min ahead)", "hybrid": "Hybrid (larger of live sensor, forecast)",
          "live_margin": "Live sensor + safety margin"}


def measured_headcount(stem: str, res: str) -> pd.Series:
    """Measured mean headcount per interval, with outages unknown."""
    table = read_table(Path("data/processed") / f"{stem}_{res}.csv")
    return table["mean_headcount"].where(~table["outage"])


def control_headcounts(stem: str, res: str, margin: float) -> pd.DataFrame:
    """Headcount each strategy acts on, per control interval of the test period."""
    forecasts = read_table(Path("data/processed") / f"{stem}_forecast_{res}.csv")
    measured = measured_headcount(stem, res)
    live = measured.shift(1).reindex(forecasts.index)
    return pd.DataFrame({
        "oracle": forecasts["observed"],
        "live_sensor": live,
        "recent_max": np.maximum(live, measured.shift(2).reindex(forecasts.index)),
        "forecast": forecasts["boosting"],
        "hybrid": np.maximum(live, forecasts["boosting"]),
        "live_margin": (1 + margin) * live,
    })


def fixed_schedule(index: pd.DatetimeIndex) -> np.ndarray:
    """Design airflow for full capacity on weekdays 07:00-19:00, off otherwise."""
    on = (index.dayofweek < 5) & (index.hour >= 7) & (index.hour < 19)
    return np.where(on, RATE_PER_PERSON * DEFAULT_CAPACITY, 0.0)


def slots(index: pd.DatetimeIndex, res: str) -> pd.DatetimeIndex:
    """Control interval of each minute. Floored in UTC: the local clock repeats
    an hour when daylight saving ends."""
    return index.tz_convert("UTC").floor(res).tz_convert(index.tz)


def choose_margin(minute: pd.DataFrame, stem: str) -> tuple[float, pd.DataFrame]:
    """Pick the live-sensor safety margin on the tuning period (5-min control)."""
    people = minute["mean_headcount"].where(~minute["outage"], 0.0).to_numpy()
    live = measured_headcount(stem, "5min").shift(1)
    slot = slots(minute.index, "5min")
    valid = slot.isin(live.dropna().index) & ~minute["outage"].to_numpy()
    rows = []
    for margin in MARGINS:
        airflow = (RATE_PER_PERSON * (1 + margin) * live).reindex(slot).fillna(0.0).to_numpy()
        co2 = simulate_co2(people, airflow + infiltration_airflow(), DT)
        rows.append({"margin": margin, **assess(people, airflow, co2, DT, mask=valid)})
    sweep = pd.DataFrame(rows)
    ok = sweep["person_hours_over_limit"] <= (1 - MARGIN_TARGET) * sweep["person_hours_over_limit"].iloc[0]
    return float(sweep.loc[ok, "margin"].iloc[0]), sweep


def run(minute: pd.DataFrame, stem: str, margin: float) -> tuple[pd.DataFrame, dict]:
    """Simulate every strategy at every control resolution."""
    people = minute["mean_headcount"].where(~minute["outage"], 0.0).to_numpy()
    rows, traces = [], {"co2": {}, "air": {}}
    fixed = fixed_schedule(minute.index)
    for res in CONTROL_RESOLUTIONS:
        heads = control_headcounts(stem, res, margin)
        # Only minutes whose control interval has a valid forecast are scored.
        slot = slots(minute.index, res)
        valid = slot.isin(heads.dropna().index) & ~minute["outage"].to_numpy()
        strategies = {name: (RATE_PER_PERSON * heads[name].clip(lower=0)).reindex(slot).fillna(0.0).to_numpy()
                      for name in heads.columns}
        strategies = {"fixed_schedule": fixed, **strategies}
        for name, airflow in strategies.items():
            co2 = simulate_co2(people, airflow + infiltration_airflow(), DT)
            rows.append({"control": res, "strategy": name, **assess(people, airflow, co2, DT, mask=valid)})
            if res == "5min":
                traces["co2"][name], traces["air"][name] = co2, airflow
    return pd.DataFrame(rows), {"people": people, "index": minute.index, **traces}


def plot_tradeoff(results: pd.DataFrame, path: Path) -> None:
    """Outdoor air supplied vs person-hours above the CO2 limit, 5-min control."""
    r = results[results["control"] == "5min"].set_index("strategy")
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    markers = {"fixed_schedule": "s", "oracle": "D", "live_sensor": "^", "recent_max": "v", "forecast": "o",
               "hybrid": "X", "live_margin": "P"}
    for name, row in r.iterrows():
        ax.scatter(row["air_m3"] / 1000, row["person_hours_over_limit"], s=70, marker=markers[name],
                   color=COLORS[name], edgecolor="white", linewidth=1.2, zorder=3,
                   label=f"{LABELS[name]}: {row['air_m3'] / 1000:.0f}k m³, {row['person_hours_over_limit']:.0f} p-h")
    ax.set_xlabel("Outdoor air supplied over test period (thousand m³)", color=MUTED)
    ax.set_ylabel(f"Person-hours above {CO2_LIMIT:.0f} ppm (modelled)", color=MUTED)
    ax.set_title("Ventilation strategies, 5-min control: air used vs air-quality shortfall",
                 color=INK, fontsize=11, loc="left")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-15)
    # Several strategies sit close together near the origin: a legend reads
    # better than direct labels.
    ax.legend(frameon=False, fontsize=7.5, loc="upper right", labelcolor=INK, handletextpad=0.4)
    style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")


def plot_day(traces: dict, path: Path) -> None:
    """Measured occupancy, air required vs supplied, and modelled CO2 on the
    busiest test-period weekday."""
    index = traces["index"]
    people = pd.Series(traces["people"], index=index)
    # Busiest weekday with no capped (likely erroneous) readings.
    daily = people.groupby(index.normalize())
    clean_days = daily.max()[daily.max() < DEFAULT_CAPACITY].index
    totals = daily.sum()
    day = totals[totals.index.isin(clean_days) & (totals.index.dayofweek < 5)].idxmax()
    m = (index >= day) & (index < day + pd.Timedelta("1D"))
    hours = (index[m] - day).total_seconds() / 3600
    required = RATE_PER_PERSON * people[m].to_numpy()
    fixed_air = traces["air"]["fixed_schedule"][m]
    short = (fixed_air < required) & (required > 0)

    fig, (top, mid, bottom) = plt.subplots(3, 1, figsize=(7.0, 6.6), dpi=200, sharex=True,
                                           gridspec_kw={"height_ratios": [1, 1.5, 1.6]})
    top.fill_between(hours, people[m], step="post", color="#86b6ef", lw=0)
    top.set_ylabel("People", color=MUTED)
    top.set_title(f"G.20 on {day:%a %d %b %Y}", color=INK, fontsize=11, loc="left")
    top.text(0.01, 0.92, "Measured: people counters", transform=top.transAxes, color=MUTED, fontsize=8, va="top")

    mid.fill_between(hours, required, where=~short, step="post", color="#86b6ef", lw=0,
                     label="Required: 10 L/s x measured people")
    mid.fill_between(hours, required, where=short, step="post", color="#e5484d", alpha=0.6, lw=0,
                     label="Required but not supplied by the fixed schedule")
    for name in ("fixed_schedule", "live_margin"):
        mid.step(hours, traces["air"][name][m], where="post", color=COLORS[name], lw=1.4,
                 label=f"Supplied: {LABELS[name]}")
    mid.set_ylabel("Outdoor air (L/s)", color=MUTED)
    mid.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=INK)

    bottom.axvspan(19, 24, color=GRID, alpha=0.6, lw=0, label="Fixed schedule off (assumed, from 19:00)")
    for name in ("fixed_schedule", "live_margin"):
        bottom.plot(hours, traces["co2"][name][m], color=COLORS[name], lw=1.5, ls=(0, (5, 2)), label=LABELS[name])
    bottom.axhline(CO2_LIMIT, color=MUTED, lw=1, ls=(0, (1, 2)))
    bottom.annotate(f"{CO2_LIMIT:.0f} ppm limit", (0.3, CO2_LIMIT), xytext=(0, 4), textcoords="offset points",
                    color=MUTED, fontsize=8)
    bottom.set_ylabel("CO2 (ppm)", color=MUTED)
    bottom.text(0.01, 0.97, "Modelled, not measured (mass-balance simulation)", transform=bottom.transAxes,
                color=MUTED, fontsize=8, va="top", style="italic")
    bottom.legend(frameon=False, fontsize=7.5, loc="center left", labelcolor=INK)
    bottom.set_xlabel("Hour of day", color=MUTED)
    bottom.set_xticks(range(0, 25, 3))
    bottom.set_xlim(0, 24)
    for ax in (top, mid, bottom):
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


def write_markdown(results: pd.DataFrame, sweep: pd.DataFrame, margin: float, path: Path, room: str) -> None:
    """Margin selection and results per control resolution, relative to the fixed schedule."""
    lines = [
        f"# {room} ventilation strategies: 2026 test period",
        "",
        f"Outdoor air at {RATE_PER_PERSON:.0f} L/s per person; CO2 simulated minute by minute (well-mixed room, "
        f"81 m³, 0.0052 L/s CO2 per person, 420 ppm outdoors; steady state at the design rate "
        f"{steady_state_co2():.0f} ppm; background infiltration {INFILTRATION_ACH} air changes per hour). "
        "All CO2 values are modelled, not measured. Sensor outages excluded from scoring. *Air vs fixed* is "
        "outdoor air supplied relative to the fixed schedule (a proxy for fan and conditioning energy).",
        "",
        f"Generated by `python scripts/ventilation_study.py --room {room}`.",
        "",
        f"## Safety margin, chosen on {TUNE_START[:4]} (live sensor, 5-min control)",
        "",
        f"Smallest margin that cuts person-hours above the limit by at least {MARGIN_TARGET:.0%}: "
        f"**{margin:.0%}** ({RATE_PER_PERSON * (1 + margin):.1f} L/s per measured person).",
        "",
        "| Margin | Air (m³) | Person-hours > limit |", "|---|---|---|",
    ]
    lines += [f"| {r.margin:.0%} | {r.air_m3:,.0f} | {r.person_hours_over_limit:,.1f} |" for r in sweep.itertuples()]
    lines.append("")
    fixed_air = results.loc[results["strategy"] == "fixed_schedule", "air_m3"].iloc[0]
    for res, group in results.groupby("control", sort=False):
        lines += [f"## Control every {res}", "",
                  "| Strategy | Air (m³) | Air vs fixed | Person-hours > limit | Occupied time > limit | "
                  "Mean CO2 occupied (ppm) | Peak (ppm) |", "|---|---|---|---|---|---|---|"]
        for r in group.itertuples():
            lines.append(f"| {LABELS[r.strategy]} | {r.air_m3:,.0f} | {r.air_m3 / fixed_air:.1%} | "
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
    tune_start, test_start = (pd.Timestamp(t, tz=minute.index.tz) for t in (TUNE_START, TEST_START))
    margin, sweep = choose_margin(minute[(minute.index >= tune_start) & (minute.index < test_start)], stem)
    print(f"safety margin chosen on {TUNE_START[:4]}: {margin:.0%}")
    results, traces = run(minute[minute.index >= test_start], stem, margin)

    results.to_csv("reports/ventilation_results.csv", index=False, float_format="%.4f")
    write_markdown(results, sweep, margin, Path("reports/ventilation_results.md"), args.room)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plot_tradeoff(results, Path("reports/figures/ventilation_tradeoff.png"))
    plot_day(traces, Path("reports/figures/ventilation_day.png"))
    print(results.round(3).to_string())


if __name__ == "__main__":
    main()
