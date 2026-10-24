"""Profile the raw Dataset 2 exports and print the figures quoted in
``reports/data_audit_findings.md``.

This is an exploratory, read-only audit: it never modifies the raw files. It
loads the occupancy and environmental CSVs into a local DuckDB cache (the
occupancy file is ~1.6 GB, too large to profile comfortably in pandas),
unpacks the nested environmental ``jsondata`` column, and runs the coverage,
duplication, staleness, range and cross-dataset checks.

Usage (from the repository root)::

    python scripts/profile_raw_data.py [--datasets Datasets] [--rebuild]

Requires ``duckdb`` and ``openpyxl`` in addition to the core requirements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import pandas as pd

OCC_CSV = "5occupancySensor_MayToDec2024_9MRows/5occupancySensor_MayToDec2024_9MRows.csv"
ENV_CSV = "5EnvSensor_MayToDec2024_180kRows/5EnvSensor_MayToDec2024_180kRows.csv"
META_XLSX = "Sensor ID and Locations.xlsx"
CACHE = Path("data/profile_cache.duckdb")  # untracked, see .gitignore

# Windows in which each env sensor's measurement timestamp actually advances
# (identified by this script's staleness check; used for the CO2 correlation).
LIVE_WINDOWS = {
    "6012002000869": ("2024-03-01", "2024-06-13"),
    "6012002000326": ("2025-09-12", "2026-04-22"),
}


def show(con: duckdb.DuckDBPyConnection, title: str, sql: str) -> pd.DataFrame:
    """Run ``sql``, print the result under ``title`` and return it."""
    df = con.sql(sql).df()
    print(f"\n## {title}\n{df.to_string()}", flush=True)
    return df


def build_cache(con: duckdb.DuckDBPyConnection, root: Path) -> None:
    """Load the raw CSVs into typed DuckDB tables (raw values unchanged)."""
    con.execute(f"""
        CREATE OR REPLACE TABLE occ AS
        SELECT id, floorspaceid AS sp, occupancystatus AS st,
               previousoccupancystatus AS prev, headcount AS hc,
               collecteddate::TIMESTAMPTZ AS t,
               occupancystatuschangedate::TIMESTAMPTZ AS tc
        FROM read_csv('{(root / OCC_CSV).as_posix()}', header=true, nullstr='NULL',
                      types={{'collecteddate':'VARCHAR','occupancystatuschangedate':'VARCHAR'}})""")
    con.execute(f"""
        CREATE OR REPLACE TABLE envraw AS
        SELECT * FROM read_csv('{(root / ENV_CSV).as_posix()}', header=true, nullstr='NULL',
                               types={{'sensorid':'VARCHAR','createdate':'VARCHAR',
                                       'processdate':'VARCHAR','jsondata':'VARCHAR'}},
                               max_line_size=10000000)""")
    # jsondata is a list of {variable: {name, unit}, values: [{time, value}]};
    # explode to one row per (record, variable, value).
    con.execute("""
        CREATE OR REPLACE TABLE envlong AS
        SELECT id, sensorid, v->>'$.variable.name' AS var, v->>'$.variable.unit' AS unit,
               to_timestamp((x->>'$.time')::DOUBLE) AS t, (x->>'$.value')::DOUBLE AS val
        FROM (SELECT id, sensorid, unnest(from_json(jsondata::JSON, '["JSON"]')) AS v FROM envraw) a,
             LATERAL (SELECT unnest(from_json(v->'$.values', '["JSON"]')) AS x)""")
    # Wide table: t = sensor measurement time (JSON), cdate = platform ingest time.
    con.execute("""
        CREATE OR REPLACE TABLE env AS
        SELECT w.*, r.createdate::TIMESTAMPTZ AS cdate
        FROM (PIVOT (SELECT id, sensorid, t, var, val FROM envlong)
              ON var USING first(val) GROUP BY id, sensorid, t) w
        JOIN envraw r USING (id)""")


def profile_occupancy(con: duckdb.DuckDBPyConnection) -> None:
    """Coverage, state semantics, duplicates, gaps and plausibility of occupancy rows."""
    show(con, "Occupancy overview", """
        SELECT count(*) AS n_rows, count(DISTINCT id) AS n_ids, count(DISTINCT sp) AS n_spaces,
               min(t) AS first_t, max(t) AS last_t FROM occ""")
    show(con, "Occupancy per space (rooms from metadata sheet)", """
        SELECT o.sp, count(*) AS n, min(t) AS first_t, max(t) AS last_t,
               round(avg(hc), 2) AS mean_hc, max(hc) AS max_hc,
               (SELECT string_agg(DISTINCT m."Floor/Room", '; ') FROM meta m WHERE m.SpaceId = o.sp) AS room
        FROM occ o GROUP BY o.sp ORDER BY n DESC""")
    show(con, "Status transitions", "SELECT st, prev, count(*) AS n FROM occ GROUP BY ALL ORDER BY n DESC")
    show(con, "Status vs zero headcount", "SELECT st, hc = 0 AS zero_hc, count(*) AS n FROM occ GROUP BY ALL ORDER BY 1, 2")
    show(con, "Rows per month", """
        SELECT strftime(date_trunc('month', t), '%Y-%m') AS month, count(*) AS n,
               count(DISTINCT sp) AS n_spaces FROM occ GROUP BY 1 ORDER BY 1""")
    show(con, "Duplicates", """
        SELECT (SELECT count(*) FROM occ) - (SELECT count(*) FROM (SELECT DISTINCT sp, st, prev, hc, t FROM occ))
                   AS dup_rows_ignoring_id,
               (SELECT count(*) FROM (SELECT sp, t FROM occ GROUP BY 1, 2 HAVING count(*) > 1))
                   AS dup_space_timestamp_keys""")
    con.execute("""CREATE OR REPLACE TEMP TABLE occ_gaps AS
        SELECT sp, t, epoch(t - lag(t) OVER (PARTITION BY sp ORDER BY t, id)) AS dt FROM occ""")
    show(con, "Inter-row interval (s) per space", """
        SELECT left(sp, 8) AS sp, quantile_cont(dt, [0.05, 0.5, 0.95, 0.99]) AS dt_quantiles,
               count(*) FILTER (WHERE dt > 3600) AS gaps_gt_1h, count(*) FILTER (WHERE dt > 86400) AS gaps_gt_1d,
               round(max(dt) / 86400, 1) AS max_gap_days FROM occ_gaps GROUP BY 1""")
    show(con, "Gaps longer than 3 days", """
        SELECT left(sp, 8) AS sp, t - to_seconds(dt::BIGINT) AS gap_start, t AS gap_end,
               round(dt / 86400, 1) AS gap_days FROM occ_gaps WHERE dt > 3 * 86400 ORDER BY 1, 2""")
    show(con, "Night-time (01:00-04:59) detections and headcount extremes", """
        SELECT left(sp, 8) AS sp,
               count(DISTINCT t::DATE) FILTER (WHERE hour(t) BETWEEN 1 AND 4 AND hc > 0) AS nights_with_occ,
               count(DISTINCT t::DATE) AS n_days, max(hc) AS max_hc, quantile_cont(hc, 0.999) AS p999_hc
        FROM occ GROUP BY 1""")
    show(con, "Headcount >= 60 (context rows)", """
        SELECT left(sp, 8) AS sp, t, hc, lag(hc) OVER w AS prev_hc, lead(hc) OVER w AS next_hc
        FROM occ WINDOW w AS (PARTITION BY sp ORDER BY t, id) QUALIFY hc >= 60 ORDER BY t LIMIT 10""")


def profile_environment(con: duckdb.DuckDBPyConnection) -> None:
    """Coverage, staleness, duplication and value ranges of environmental rows."""
    show(con, "Env overview", """
        SELECT count(*) AS n_rows, count(DISTINCT sensorid) AS n_sensors,
               count(*) FILTER (WHERE processing_errors IS NOT NULL) AS n_processing_errors,
               min(createdate) AS first_ingest, max(createdate) AS last_ingest FROM envraw""")
    show(con, "Staleness per sensor (distinct measurement times / values)", """
        SELECT sensorid, count(*) AS n_rows, count(DISTINCT t) AS distinct_meas_times,
               count(DISTINCT ("Carbon dioxide" || '|' || Temperature || '|' || Humidity)) AS distinct_readings,
               (SELECT string_agg(DISTINCT m."Floor/Room" || ' ' || m.Area, '; ')
                  FROM meta_env m WHERE m.ID::VARCHAR = e.sensorid) AS room,
               min(cdate) AS first_ingest, max(cdate) AS last_ingest
        FROM env e GROUP BY sensorid ORDER BY n_rows DESC""")
    show(con, "Live (advancing-timestamp) rows per sensor per month", """
        SELECT strftime(date_trunc('month', cdate), '%Y-%m') AS month, sensorid,
               count(*) AS n_rows, count(DISTINCT t) AS distinct_meas_times
        FROM env GROUP BY ALL ORDER BY 2, 1""")
    show(con, "Ingest cadence and ingest lag", """
        SELECT sensorid, quantile_cont(dc, [0.05, 0.5, 0.95]) AS ingest_interval_s,
               round(median(epoch(cdate - t)) / 86400, 1) AS median_ingest_lag_days
        FROM (SELECT sensorid, cdate, t,
                     epoch(cdate - lag(cdate) OVER (PARTITION BY sensorid ORDER BY cdate, id)) AS dc FROM env)
        GROUP BY 1""")
    show(con, "Variable ranges (all rows, incl. frozen)", """
        SELECT var, count(*) AS n, count(val) AS non_null, min(val) AS min,
               quantile_cont(val, 0.01) AS p01, median(val) AS median,
               quantile_cont(val, 0.99) AS p99, max(val) AS max
        FROM envlong GROUP BY var ORDER BY n DESC""")


def co2_vs_occupancy(con: duckdb.DuckDBPyConnection) -> None:
    """Correlate live hourly CO2 with hourly max headcount of every space.

    A raw correlation is dominated by the shared building schedule, so a
    partial correlation (hour-of-week mean removed from both series) is also
    reported as weak evidence of co-location.
    """
    env = con.sql("""SELECT sensorid, t, any_value("Carbon dioxide") AS co2 FROM env
                     WHERE sensorid IN ('6012002000869', '6012002000326') GROUP BY 1, 2""").df()
    env["t"] = pd.to_datetime(env.t, utc=True).dt.tz_convert("Australia/Melbourne")
    occ = con.sql("SELECT left(sp, 8) AS sp, date_trunc('hour', t) AS h, max(hc) AS hc FROM occ GROUP BY 1, 2").df()
    occ["h"] = pd.to_datetime(occ.h, utc=True).dt.tz_convert("Australia/Melbourne")
    occ_h = occ.pivot(index="h", columns="sp", values="hc")

    for sensor, (lo, hi) in LIVE_WINDOWS.items():
        idx = pd.date_range(lo, hi, freq="1h", tz="Australia/Melbourne", inclusive="left")
        co2 = (env[env.sensorid == sensor].set_index("t").co2.resample("1h").mean().reindex(idx))
        o = occ_h.reindex(idx).fillna(0)  # no occupancy rows in an hour => nobody detected
        how = idx.dayofweek * 24 + idx.hour

        def resid(x: pd.Series) -> pd.Series:
            return x - x.groupby(how).transform("mean")

        profile = co2.groupby(co2.index.hour).mean().round(0)
        print(f"\n## CO2 vs occupancy: sensor {sensor} ({lo} to {hi})")
        print(f"hourly CO2 profile: min {profile.min():.0f} ppm at {profile.idxmin()}h, "
              f"max {profile.max():.0f} ppm at {profile.idxmax()}h")
        for sp in o.columns:
            if o[sp].sum() == 0:
                continue
            print(f"  space {sp}: r = {co2.corr(o[sp]):+.2f}, partial r = {resid(co2).corr(resid(o[sp])):+.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--datasets", type=Path, default=Path("Datasets"), help="folder holding the raw exports")
    parser.add_argument("--rebuild", action="store_true", help="reload the CSVs even if the cache exists")
    args = parser.parse_args()

    CACHE.parent.mkdir(exist_ok=True)
    rebuild = args.rebuild or not CACHE.exists()
    con = duckdb.connect(str(CACHE))
    con.execute("SET TimeZone = 'Australia/Melbourne'")
    if rebuild:
        build_cache(con, args.datasets)
    sheets = pd.read_excel(args.datasets / META_XLSX, sheet_name=None)
    con.register("meta_env", sheets["ENV"])
    con.register("meta", sheets["OCC (To be confirmed)"])

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    profile_occupancy(con)
    profile_environment(con)
    co2_vs_occupancy(con)


if __name__ == "__main__":
    main()
