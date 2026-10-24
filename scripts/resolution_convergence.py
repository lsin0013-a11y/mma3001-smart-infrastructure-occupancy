"""Resolution convergence study for one room's occupancy labels.

For a range of interval lengths (1 h down to 10 s) this compares the share of
usable intervals labelled occupied under two definitions:

- **held state** (the method used by :mod:`smart_infra.occupancy`): each
  event's status holds until the next event, for at most ``max_hold``;
- **event instants** (the first method tried): an interval is occupied only
  if an event with headcount >= 1 falls inside it.

The time-weighted occupied share (mean ``occupied_fraction``) is also reported
at every resolution; it should be identical, and the held-state share should
converge to it from above as the intervals shrink.

Writes ``reports/resolution_convergence.csv`` and
``reports/figures/resolution_convergence.png``.

Usage (from the repository root)::

    python scripts/resolution_convergence.py [--room G.20] [--datasets Datasets]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from build_occupancy_tables import OCC_CSV, load_space_events  # noqa: E402
from smart_infra.occupancy import SPACE_IDS, build_occupancy_table, clean_events  # noqa: E402

STUDY_RESOLUTIONS = ["1h", "30min", "10min", "5min", "2min", "1min", "30s", "15s", "10s"]

# Default chart palette (dataviz skill, categorical slots 1 and 2) and ink.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def study(events: pd.DataFrame) -> pd.DataFrame:
    """Occupied shares per resolution under both definitions, over usable intervals."""
    clean, _ = clean_events(events)
    rows = []
    for res in STUDY_RESOLUTIONS:
        table, _ = build_occupancy_table(events, resolution=res)
        usable = ~table["outage"]
        # Event-instant labels on the same intervals, for comparison.
        hit = (clean["headcount"] >= 1).groupby(clean["timestamp"].dt.tz_convert("UTC").dt.floor(res)).any()
        instants = hit.reindex(table.index.tz_convert("UTC"), fill_value=False).to_numpy()
        rows.append({
            "resolution": res,
            "interval_s": pd.Timedelta(res).total_seconds(),
            "n_intervals": int(usable.sum()),
            "held_state_share": table.loc[usable, "occupied"].mean(),
            "event_instant_share": instants[usable.to_numpy()].mean(),
            "time_weighted_share": table.loc[usable, "occupied_fraction"].mean(),
        })
        print(f"{res:>6}: held {rows[-1]['held_state_share']:.1%}  instants {rows[-1]['event_instant_share']:.1%}  "
              f"time-weighted {rows[-1]['time_weighted_share']:.2%}", flush=True)
    return pd.DataFrame(rows)


def plot(results: pd.DataFrame, room: str, path: Path) -> None:
    """Draw both shares against interval length, with the time-weighted limit."""
    limit = results["time_weighted_share"].mean() * 100
    x = results["interval_s"] / 60
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
    fig.patch.set_facecolor("white")

    ax.axhline(limit, color=MUTED, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.plot(x, results["held_state_share"] * 100, color=BLUE, lw=2, marker="o", ms=6,
            mec="white", mew=1.2, zorder=3, label="Held state (method used)")
    ax.plot(x, results["event_instant_share"] * 100, color=ORANGE, lw=2, marker="s", ms=6,
            mec="white", mew=1.2, zorder=3, label="Event instants (first method)")

    ax.set_xscale("log")
    ax.invert_xaxis()  # refine left to right
    ticks = [60, 30, 10, 5, 2, 1, 0.5, 0.25, 1 / 6]
    ax.set_xticks(ticks, ["1 h", "30 min", "10 min", "5 min", "2 min", "1 min", "30 s", "15 s", "10 s"])
    ax.minorticks_off()
    ax.set_ylim(0, 62)
    ax.set_xlabel("Interval length (finer →)", color=MUTED)
    ax.set_ylabel("Intervals labelled occupied (%)", color=MUTED)
    ax.set_title(f"Room {room}: occupied share converges only when occupancy is a held state",
                 color=INK, fontsize=11, loc="left")

    # Direct labels at the finest resolution, in ink rather than series colour.
    last = results.iloc[-1]
    ax.annotate(f"{last['held_state_share']:.1%}", (x.iloc[-1], last["held_state_share"] * 100),
                xytext=(6, 6), textcoords="offset points", color=INK, fontsize=9)
    ax.annotate(f"{last['event_instant_share']:.1%}", (x.iloc[-1], last["event_instant_share"] * 100),
                xytext=(6, -12), textcoords="offset points", color=INK, fontsize=9)
    ax.annotate(f"Time actually occupied: {limit:.1f}%", (60, limit), xytext=(0, 5),
                textcoords="offset points", color=MUTED, fontsize=9)

    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.legend(frameon=False, loc="lower left", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--room", default="G.20", choices=sorted(SPACE_IDS), help="room to process")
    parser.add_argument("--datasets", type=Path, default=Path("Datasets"), help="folder holding the raw exports")
    args = parser.parse_args()

    events = load_space_events(args.datasets / OCC_CSV, SPACE_IDS[args.room])
    results = study(events)
    results.to_csv("reports/resolution_convergence.csv", index=False, float_format="%.5f")
    plot(results, args.room, Path("reports/figures/resolution_convergence.png"))
    print("wrote reports/resolution_convergence.csv and reports/figures/resolution_convergence.png")


if __name__ == "__main__":
    main()
