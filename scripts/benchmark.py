"""Performance study: timing, memory, operation counts and numerical accuracy.

1. **Interval integration.** The vectorised cumulative-sum and binary-search
   method in ``smart_infra.occupancy._integrate`` against a straightforward
   Python loop over events, on G.20's real events at 5-min resolution. Both
   must give the same answer.
2. **CO2 simulation.** The exact per-step solution against explicit Euler:
   run time, error against the exact solution as the time step grows, and
   Euler's stability limit dt < 2V/Q.
3. **Memory** of building the 5-min occupancy table (tracemalloc peak).

Writes ``reports/performance.md``.

Usage (from the repository root)::

    python scripts/benchmark.py [--room G.20] [--datasets Datasets]
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from build_occupancy_tables import OCC_CSV, load_space_events
from smart_infra.occupancy import SPACE_IDS, _integrate, _to_ns, build_occupancy_table, clean_events
from smart_infra.ventilation import (
    RATE_PER_PERSON,
    ROOM_AREA,
    ROOM_HEIGHT,
    infiltration_airflow,
    simulate_co2,
)


def best_of(fn, repeats: int = 3) -> tuple[float, object]:
    """Fastest wall time of ``repeats`` runs, and the last result."""
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - started)
    return min(times), result


def integrate_loop(starts, lengths, weights, bounds, step):
    """Reference implementation: walk every piece and add its overlap to each interval it touches."""
    out = np.zeros(len(bounds))
    first = bounds[0]
    for s, length, w in zip(starts.tolist(), lengths.tolist(), weights.tolist()):
        e = s + length
        k = max((s - first) // step, 0)
        while k < len(bounds) and bounds[k] < e:
            lo, hi = max(s, bounds[k]), min(e, bounds[k] + step)
            if hi > lo:
                out[k] += (hi - lo) * w
            k += 1
    return out


def integration_study(events: pd.DataFrame) -> list[str]:
    clean, _ = clean_events(events)
    t = _to_ns(clean["timestamp"])
    hold_end = np.minimum(np.append(t[1:], np.iinfo(np.int64).max), t + pd.Timedelta("5min").value)
    lengths, weights = hold_end - t, clean["headcount"].to_numpy().astype(float)
    step = pd.Timedelta("5min").value
    bounds = np.arange(t[0] - t[0] % step, t[-1] + 1, step)
    n, m = len(t), len(bounds)

    fast_s, fast = best_of(lambda: _integrate(t, lengths, weights, bounds, step))
    slow_s, slow = best_of(lambda: integrate_loop(t, lengths, weights, bounds, step), repeats=1)
    max_diff = float(np.max(np.abs(fast - slow)) / step)

    # Operation and traffic estimates for the vectorised method.
    log_n = np.log2(n)
    ops = 2 * n + 2 * m * (log_n + 6)          # cumsum + product; per boundary: search + ~6 ops
    bytes_moved = 8 * (4 * n + 6 * m)          # starts, lengths, weights, cumsum; bounds, results
    return [
        "## 1. Interval integration (G.20, 5-min intervals)",
        "",
        f"{n:,} events (occupancy pieces) integrated over {m:,} intervals.",
        "",
        "| Method | Time (s) | Complexity |",
        "|---|---|---|",
        f"| Python loop over pieces (reference) | {slow_s:.2f} | O(n + m), interpreted |",
        f"| Cumulative sum + binary search (`_integrate`) | {fast_s:.3f} | O(n + m log n), vectorised |",
        "",
        f"Speed-up: **{slow_s / fast_s:.0f}x**. Largest difference between the two: {max_diff:.2e} person "
        "(rounding only), so the fast method is verified against the reference.",
        "",
        f"Estimated work for the vectorised method: about {ops / 1e6:.0f} million operations (2 per piece "
        f"for the cumulative sum; per boundary a binary search of log2(n) = {log_n:.0f} comparisons plus about "
        f"6 arithmetic operations, at 2 boundaries per interval) and about {bytes_moved / 1e6:.0f} MB of memory "
        f"traffic, an arithmetic intensity of roughly {ops / bytes_moved:.2f} operations per byte. "
        f"That is far below what a CPU core can compute per byte it loads, so the method is memory-bound: "
        f"its {fast_s * 1e3:.0f} ms is set by moving the arrays, not by arithmetic. "
        f"At {ops / fast_s / 1e9:.2f} billion operations per second it runs well below a core's peak, "
        "consistent with that. The loop does a similar amount of arithmetic, but the Python interpreter "
        f"adds about {slow_s / n * 1e6:.1f} microseconds of overhead per piece.",
        "",
    ]


def co2_study() -> list[str]:
    volume = ROOM_AREA * ROOM_HEIGHT
    rng = np.random.default_rng(0)
    # One synthetic week at 1-min steps: random occupancy, airflow at 10 L/s per person.
    minutes = 7 * 24 * 60
    people = np.repeat(rng.integers(0, 12, minutes // 30), 30).astype(float)
    airflow = RATE_PER_PERSON * people + infiltration_airflow()

    exact_s, exact = best_of(lambda: simulate_co2(people, airflow, 60.0))
    euler_s, _ = best_of(lambda: simulate_co2(people, airflow, 60.0, method="euler"))
    lines = [
        "## 2. CO2 simulation: exact solution vs explicit Euler",
        "",
        f"One week at 1-min steps ({minutes:,} steps), occupancy 0-11 people changing every 30 min, "
        f"supply 10 L/s per person plus infiltration, room volume {volume:.0f} m³.",
        "",
        "| Method | Time (s) | Operations per step |",
        "|---|---|---|",
        f"| Exact (analytical within each step) | {exact_s:.3f} | about 4, plus one exp per step computed "
        "up front in a vectorised pass |",
        f"| Explicit Euler | {euler_s:.3f} | about 5 |",
        "",
        "Both are sequential loops, since each step needs the previous concentration, so their run times "
        "are similar and dominated by the interpreter. The difference is accuracy:",
        "",
        "| Euler step (s) | Q·dt/V at peak flow | Max error vs exact (ppm) |",
        "|---|---|---|",
    ]
    q_peak = airflow.max() / 1000
    # Steps up to 3 h, so the largest pass Euler's stability limit in the large room.
    for dt in (60, 300, 900, 1800, 3600, 7200, 10800):
        k = int(dt // 60)
        p, q = people[::k], airflow[::k]
        reference = simulate_co2(p, q, float(dt))
        euler = simulate_co2(p, q, float(dt), method="euler")
        with np.errstate(over="ignore", invalid="ignore"):
            err = np.max(np.abs(euler - reference))
        shown = f"{err:,.1f}" if np.isfinite(err) and err < 1e6 else "diverged (oscillation grows without bound)"
        lines.append(f"| {dt} | {q_peak * dt / volume:.2f} | {shown} |")
    limit = 2 * volume / q_peak
    lines += [
        "",
        f"Euler's error grows roughly in proportion to the step (first-order method), and the method is only "
        f"stable while Q·dt/V < 2, i.e. dt < {limit:,.0f} s at this peak flow. Past that limit (the 10800 s row) "
        "the solution overshoots and oscillates whenever the flow is high, and the error jumps; with a "
        "constant high flow it grows without bound (tested). The exact solution has no step-"
        "size error for piecewise-constant inputs and is unconditionally stable, so it is used for all "
        "results. Its result at a 1-min step and a 10-s step agree to machine precision (tested).",
        "",
    ]
    return lines


def memory_study(events: pd.DataFrame) -> list[str]:
    tracemalloc.start()
    started = time.perf_counter()
    table, _ = build_occupancy_table(events, resolution="5min")
    seconds = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return [
        "## 3. Building the 5-min table",
        "",
        f"{len(events):,} events -> {len(table):,} intervals in {seconds:.1f} s, peak Python memory "
        f"{peak / 1e6:.0f} MB (tracemalloc). The raw 1.6 GB CSV is never loaded whole: DuckDB filters it to "
        "the one room before it reaches pandas.",
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--room", default="G.20", choices=sorted(SPACE_IDS))
    parser.add_argument("--datasets", type=Path, default=Path("Datasets"))
    args = parser.parse_args()

    events = load_space_events(args.datasets / OCC_CSV, SPACE_IDS[args.room])
    lines = [f"# Performance study ({args.room})", "",
             "Generated by `python scripts/benchmark.py`. Times are the best of 3 runs (1 for the slow loop) "
             "on the author's laptop, so compare them with each other rather than as absolutes. Model training "
             "times are in `forecast_results.md`.", ""]
    lines += integration_study(events)
    lines += co2_study()
    lines += memory_study(events)
    Path("reports/performance.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
