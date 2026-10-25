"""Compare ventilation strategies for one room over the 2026 test period.

The true occupancy is rebuilt at 1-min resolution from the raw events. Each
strategy sets the outdoor-air supply once per control interval (5 min, 10 min,
30 min or 1 h), and the room's CO2 is simulated minute by minute with the
exact solution of the mass balance in :mod:`smart_infra.ventilation`.

The design occupancy (what the fixed strategies are sized for) is the 99.9th
percentile of the headcount over occupied minutes in the training period
(:func:`smart_infra.ventilation.design_occupancy`).

Strategies (supply = rate per person x headcount used):

- ``always_on``: constant air volume for the design occupancy, all the time;
- ``fixed_schedule``: design occupancy, Mon-Fri 07:00-19:00, off otherwise;
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

Sensor failure: the live-sensor strategies supply no air when the sensor is
dead, because a dead sensor looks like an empty room. The fallback supplies
full design air whenever the room's average headcount for that time of the
week exceeds a threshold (:func:`smart_infra.ventilation.fallback_airflow`).
The threshold is chosen on 2025 (averages from 2024): the largest in
:data:`THRESHOLDS` whose person-hours above the limit, with the sensor dead
all year, are no more than the working live-sensor strategy's. It is tested
by simulating the sensor dead for the whole test period, where the true
headcount is known.

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
    DESIGN_QUANTILE,
    INFILTRATION_ACH,
    RATE_PER_PERSON,
    ROOM_AREA,
    ROOM_HEIGHT,
    assess,
    design_occupancy,
    fallback_airflow,
    infiltration_airflow,
    simulate_co2,
    steady_state_co2,
)  # noqa: E402

CONTROL_RESOLUTIONS = ["5min", "10min", "30min", "1h"]
TUNE_START, TEST_START = "2025-01-01", "2026-01-01"
DT = 60.0
MARGINS = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
MARGIN_TARGET = 0.9
THRESHOLDS = (0.1, 0.25, 0.5, 1.0, 2.0)

# Default chart palette (dataviz skill) and ink.
COLORS = {"always_on": "#0b0b0b", "fixed_schedule": "#52514e", "oracle": "#1baf7a", "live_sensor": "#eda100", "recent_max": "#9e9d98",
          "forecast": "#2a78d6", "hybrid": "#6250d6", "live_margin": "#eb6834"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
AIR_AXIS_CUT = 12.0  # % of fixed-schedule air where Figure 1's air axis is cut
ACCENT, MUTED_BAR = "#2a78d6", "#c3c2bd"  # highlighted final design; other strategies
LABELS = {"always_on": "Always on (design occupancy)", "fixed_schedule": "Fixed schedule (07-19 weekdays)", "oracle": "Oracle (actual headcount)",
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


def design_people(minute: pd.DataFrame) -> int:
    """Design occupancy from the training period's 1-min headcounts (outages excluded)."""
    train = minute[(minute.index < pd.Timestamp(TEST_START, tz=minute.index.tz)) & ~minute["outage"]]
    return design_occupancy(train["mean_headcount"].to_numpy())


def fixed_schedule(index: pd.DatetimeIndex, people: float) -> np.ndarray:
    """Design airflow for ``people`` on weekdays 07:00-19:00, off otherwise."""
    on = (index.dayofweek < 5) & (index.hour >= 7) & (index.hour < 19)
    return np.where(on, RATE_PER_PERSON * people, 0.0)


def week_slot(index: pd.DatetimeIndex) -> np.ndarray:
    """5-min slot of the week (0 = Monday 00:00), in local time."""
    return index.dayofweek * 288 + index.hour * 12 + index.minute // 5


def usual_headcount(index: pd.DatetimeIndex, stem: str, end: str) -> np.ndarray:
    """Average measured headcount for each time's slot of the week, from data before ``end``."""
    table = read_table(Path("data/processed") / f"{stem}_5min.csv")
    past = table[(table.index < pd.Timestamp(end, tz=table.index.tz)) & ~table["outage"]]
    mean = past["mean_headcount"].groupby(week_slot(past.index)).mean()
    return mean.reindex(week_slot(index)).fillna(0.0).to_numpy()


def choose_threshold(minute: pd.DataFrame, stem: str, people_design: int,
                     target: float) -> tuple[float, pd.DataFrame]:
    """Pick the sensor-failure fallback threshold on the tuning period.

    The sensor is simulated dead for the whole period, with averages from the
    data before it; the largest threshold whose person-hours above the limit
    do not exceed ``target`` is chosen.
    """
    people = minute["mean_headcount"].where(~minute["outage"], 0.0).to_numpy()
    usual = usual_headcount(minute.index, stem, end=TUNE_START)
    valid = ~minute["outage"].to_numpy()
    rows = []
    for threshold in THRESHOLDS:
        airflow = fallback_airflow(usual, people_design, threshold)
        co2 = simulate_co2(people, airflow + infiltration_airflow(), DT)
        rows.append({"threshold": threshold, **assess(people, airflow, co2, DT, mask=valid)})
    sweep = pd.DataFrame(rows)
    ok = sweep["person_hours_over_limit"] <= target + 1e-9
    return float(sweep.loc[ok, "threshold"].max() if ok.any() else THRESHOLDS[0]), sweep


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


def run(minute: pd.DataFrame, stem: str, margin: float, people_design: int) -> tuple[pd.DataFrame, dict]:
    """Simulate every strategy at every control resolution."""
    people = minute["mean_headcount"].where(~minute["outage"], 0.0).to_numpy()
    rows, traces = [], {"co2": {}, "air": {}}
    fixed = fixed_schedule(minute.index, people_design)
    always_on = np.full(len(minute), RATE_PER_PERSON * people_design)
    for res in CONTROL_RESOLUTIONS:
        heads = control_headcounts(stem, res, margin)
        # Only minutes whose control interval has a valid forecast are scored.
        slot = slots(minute.index, res)
        valid = slot.isin(heads.dropna().index) & ~minute["outage"].to_numpy()
        strategies = {name: (RATE_PER_PERSON * heads[name].clip(lower=0)).reindex(slot).fillna(0.0).to_numpy()
                      for name in heads.columns}
        if margin == 0:
            strategies.pop("live_margin")
        strategies = {"always_on": always_on, "fixed_schedule": fixed, **strategies}
        for name, airflow in strategies.items():
            co2 = simulate_co2(people, airflow + infiltration_airflow(), DT)
            rows.append({"control": res, "strategy": name, **assess(people, airflow, co2, DT, mask=valid)})
            if res == "5min":
                traces["co2"][name], traces["air"][name] = co2, airflow
    return pd.DataFrame(rows), {"people": people, "index": minute.index, **traces}


def sensor_failure(minute: pd.DataFrame, stem: str, margin: float, people_design: int,
                   threshold: float) -> pd.DataFrame:
    """Air and air quality over the test period if the headcount sensor were dead throughout.

    The true headcount is still known, so each response can be scored.
    """
    people = minute["mean_headcount"].where(~minute["outage"], 0.0).to_numpy()
    usual = usual_headcount(minute.index, stem, end=TEST_START)
    heads = control_headcounts(stem, "5min", margin)
    slot = slots(minute.index, "5min")
    valid = slot.isin(heads.dropna().index) & ~minute["outage"].to_numpy()
    cases = {
        "Working sensor" + (f" (live + {margin:.0%})" if margin else " (live)"): (RATE_PER_PERSON * heads["live_margin"]).reindex(slot).fillna(0.0).to_numpy(),
        "Dead sensor, no fallback": np.zeros(len(minute)),
        "Dead sensor, average headcount" + (f" + {margin:.0%}" if margin else ""): RATE_PER_PERSON * (1 + margin) * usual,
        f"Dead sensor, fallback (design air when average > {threshold:g})":
            fallback_airflow(usual, people_design, threshold),
    }
    rows = []
    for name, airflow in cases.items():
        co2 = simulate_co2(people, airflow + infiltration_airflow(), DT)
        rows.append({"case": name, **assess(people, airflow, co2, DT, mask=valid)})
    return pd.DataFrame(rows)


def plot_strategies(results: pd.DataFrame, final: str, path: Path) -> None:
    """Outdoor air supplied and person-hours above the CO2 limit per strategy, 5-min control.

    Two panels share the strategy rows, so each measure keeps its own scale;
    the final design is highlighted. Air is shown as a share of the fixed
    schedule's. The two baselines use 15-40 times more air than the
    demand-controlled strategies, which would be slivers on a full scale, so
    the axis is cut at :data:`AIR_AXIS_CUT` and the baseline bars run off it
    with a break mark and their value written at the end.
    """
    r = results[results["control"] == "5min"].set_index("strategy")
    order = [k for k in ("always_on", "fixed_schedule", "oracle", "live_sensor", "recent_max", "forecast",
                         "hybrid", "live_margin") if k in r.index and (k != "live_margin" or final == k)]
    r = r.loc[order]
    names = [("Live sensor" if k == "live_sensor" else LABELS[k]) + (" (final design)" if k == final else "")
             for k in order]
    colors = [ACCENT if k == final else MUTED_BAR for k in order]
    y = np.arange(len(order))[::-1]

    fig, (left, right) = plt.subplots(1, 2, figsize=(7.2, 3.6), dpi=200, sharey=True,
                                      gridspec_kw={"width_ratios": [1.4, 1]})
    share = 100 * r["air_m3"] / r.loc["fixed_schedule", "air_m3"]
    cut = AIR_AXIS_CUT
    left.barh(y, share.clip(upper=cut), height=0.62, color=colors, zorder=2)
    for yi, v, m3 in zip(y, share, r["air_m3"] / 1000):
        if v > cut:
            # Break mark: a white slash through the bar near the axis cut.
            left.plot([cut * 0.38, cut * 0.42], [yi - 0.38, yi + 0.38], color="white", lw=3, zorder=3)
            left.text(cut * 0.97, yi, f"{v:,.0f}%  ({m3:,.0f}k m³)", va="center", ha="right",
                      fontsize=7.5, color=INK, zorder=4)
        else:
            left.text(v + cut * 0.015, yi, f"{v:.1f}%  ({m3:,.0f}k m³)", va="center", fontsize=7.5, color=INK)
    left.set_xlim(0, cut)
    left.set_xticks([])
    left.set_title("Outdoor air supplied\n(% of fixed schedule; axis cut at "
                   f"{cut:.0f}%)", color=INK, fontsize=9, loc="left")

    hours = r["person_hours_over_limit"]
    right.barh(y, hours, height=0.62, color=colors, zorder=2)
    for yi, v in zip(y, hours):
        right.text(v + hours.max() * 0.02, yi, f"{v:,.1f}", va="center", fontsize=7.5, color=INK)
    right.set_xlim(0, hours.max() * 1.3 if hours.max() > 0 else 1)
    right.set_xticks([])
    right.set_title(f"Person-hours above {CO2_LIMIT:.0f} ppm\n(modelled CO2)", color=INK, fontsize=9, loc="left")
    for ax in (left, right):
        style(ax)
        ax.grid(False)
        ax.spines["bottom"].set_visible(False)
    left.set_yticks(y, names, fontsize=8, color=INK)
    left.tick_params(axis="y", length=0)
    right.tick_params(axis="y", length=0)
    fig.suptitle("Ventilation strategies over the test period, 5-min control", color=INK, fontsize=11,
                 x=0.01, ha="left")
    # Fixed margins: tight_layout clips the long strategy names next to a suptitle.
    fig.subplots_adjust(left=0.34, right=0.97, top=0.78, bottom=0.03, wspace=0.10)
    fig.savefig(path, facecolor="white")


def plot_day(traces: dict, final: str, path: Path) -> None:
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
    for name in ("fixed_schedule", final):
        mid.step(hours, traces["air"][name][m], where="post", color=COLORS[name], lw=1.4,
                 label=f"Supplied: {LABELS[name]}")
    mid.set_ylabel("Outdoor air (L/s)", color=MUTED)
    mid.legend(frameon=False, fontsize=7, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, labelcolor=INK)

    bottom.axvspan(19, 24, color=GRID, alpha=0.6, lw=0, label="Fixed schedule off (assumed, from 19:00)")
    for name in ("fixed_schedule", final):
        bottom.plot(hours, traces["co2"][name][m], color=COLORS[name], lw=1.5, ls=(0, (5, 2)), label=LABELS[name])
    bottom.axhline(CO2_LIMIT, color=MUTED, lw=1, ls=(0, (1, 2)))
    bottom.annotate(f"{CO2_LIMIT:.0f} ppm limit", (9.5, CO2_LIMIT), xytext=(0, 4), textcoords="offset points",
                    color=MUTED, fontsize=8)
    bottom.set_ylabel("CO2 (ppm)", color=MUTED)
    bottom.text(0.01, 0.97, "Modelled, not measured (mass-balance simulation)", transform=bottom.transAxes,
                color=MUTED, fontsize=8, va="top", style="italic")
    bottom.legend(frameon=False, fontsize=7.5, loc="upper left", bbox_to_anchor=(0, 0.9), labelcolor=INK)
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


def write_markdown(results: pd.DataFrame, sweep: pd.DataFrame, margin: float, path: Path, room: str,
                   people_design: int, failure: pd.DataFrame, threshold_sweep: pd.DataFrame,
                   threshold: float, outage_share: float) -> None:
    """Margin and fallback selection, results per control resolution, and the sensor-failure test."""
    lines = [
        f"# {room} ventilation strategies: 2026 test period",
        "",
        f"Outdoor air at {RATE_PER_PERSON:.0f} L/s per person; CO2 simulated minute by minute (well-mixed room, "
        f"{ROOM_AREA * ROOM_HEIGHT:.0f} m³, 0.0052 L/s CO2 per person, 420 ppm outdoors; steady state at the design rate "
        f"{steady_state_co2():.0f} ppm; background infiltration {INFILTRATION_ACH} air changes per hour). "
        "All CO2 values are modelled, not measured. Sensor outages excluded from scoring. *Air vs fixed* is "
        "outdoor air supplied relative to the fixed schedule (a proxy for fan and conditioning energy).",
        "",
        f"Generated by `python scripts/ventilation_study.py --room {room}`.",
        "",
        f"Design occupancy (fixed strategies and fallback): **{people_design} people** "
        f"({RATE_PER_PERSON * people_design:.0f} L/s), the {DESIGN_QUANTILE:.1%} percentile of the headcount "
        f"over occupied minutes in the training period (before {TEST_START}).",
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
    lines += [f"## Sensor-failure fallback threshold, chosen on {TUNE_START[:4]} (sensor dead all year)", "",
              f"Largest threshold whose person-hours above the limit do not exceed the working live sensor's: "
              f"**{threshold:g} people** (average headcount for that time of the week).", "",
              "| Threshold (people) | Air (m³) | Person-hours > limit |", "|---|---|---|"]
    lines += [f"| {r.threshold:g} | {r.air_m3:,.0f} | {r.person_hours_over_limit:,.1f} |"
              for r in threshold_sweep.itertuples()]
    lines += ["", "## Sensor failure: test period with the sensor dead throughout", "",
              "| Case | Air (m³) | Person-hours > limit | Peak (ppm) |", "|---|---|---|---|"]
    lines += [f"| {r.case} | {r.air_m3:,.0f} | {r.person_hours_over_limit:,.1f} | {r.peak_ppm:.0f} |"
              for r in failure.itertuples()]
    working, fallback = failure["air_m3"].iloc[0], failure["air_m3"].iloc[-1]
    expected = (1 - outage_share) * working + outage_share * fallback
    lines += ["", f"Sensor outages covered {outage_share:.1%} of the training period. At that rate the "
              f"expected air use with the fallback is about {expected:,.0f} m³ "
              f"({expected / working - 1:+.0%} on the working sensor alone)."]
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
    people_design = design_people(minute)
    print(f"design occupancy: {people_design} people")
    tune = minute[(minute.index >= tune_start) & (minute.index < test_start)]
    margin, sweep = choose_margin(tune, stem)
    print(f"safety margin chosen on {TUNE_START[:4]}: {margin:.0%}")
    target = float(sweep.loc[sweep["margin"] == margin, "person_hours_over_limit"].iloc[0])
    threshold, threshold_sweep = choose_threshold(tune, stem, people_design, target)
    print(f"fallback threshold chosen on {TUNE_START[:4]}: {threshold:g} people")
    test = minute[minute.index >= test_start]
    results, traces = run(test, stem, margin, people_design)
    failure = sensor_failure(test, stem, margin, people_design, threshold)
    failure.to_csv("reports/sensor_failure.csv", index=False, float_format="%.4f")
    print(failure.round(1).to_string())

    results.to_csv("reports/ventilation_results.csv", index=False, float_format="%.4f")
    outage_share = float(minute.loc[minute.index < test_start, "outage"].mean())
    write_markdown(results, sweep, margin, Path("reports/ventilation_results.md"), args.room, people_design,
                   failure, threshold_sweep, threshold, outage_share)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    final = "live_margin" if margin > 0 else "live_sensor"
    plot_strategies(results, final, Path("reports/figures/ventilation_strategies.png"))
    plot_day(traces, final, Path("reports/figures/ventilation_day.png"))
    print(results.round(3).to_string())


if __name__ == "__main__":
    main()
