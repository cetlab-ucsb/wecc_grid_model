"""Transform PUDL demand data into GridPath .tab input files.

First pass: baseline 2024 weather year, BA-level load zones (with CAISO
split into its 4 EIA-930 subregions), single `load_component = 'all'`.

Reads:
    local_data/input_data/demand/raw/eia930_hourly_wecc_2024.parquet
    local_data/input_data/demand/raw/eia930_subregion_2024.parquet      (pulled inline)

Writes (into local_data/input_data/demand/gridpath_inputs/2025_baseline/):
    load_zones.tab     — one row per load zone (zone ID + penalty params)
    timepoints.tab     — one row per hour of 2024 (1..8784 integer IDs)
    load_mw.tab        — one row per (zone, timepoint, load_component)
    timepoint_map.csv  — companion file: timepoint int → datetime_utc, for
                         traceability back to the source data. Not a GridPath
                         input but indispensable for debugging.

Reference format pulled from bluemarble example:
    gridpath_bluemarble/examples/ra_toolkit_sync_single_year/
        weather_iteration_2010/hydro_iteration_2010/
        availability_iteration_1/1/inputs/{load_zones,timepoints,load_mw}.tab

Run:
    python transform_demand_to_gridpath.py
    python transform_demand_to_gridpath.py --year 2024 --period 2025
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# =====================================================================
# CONFIG — the decisions from the scope review, captured as constants
# =====================================================================

# PUDL S3 base URL. Tagged release for reproducibility (v2026.1.0 as of
# 2026-04-19). Use 'nightly' for dev builds or 'stable' for latest-tagged alias.
PUDL_S3_BASE = "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop"
PUDL_RELEASE = "v2026.1.0"

# Load zones we want in the output.
#
# The existing local_data/input_data/load_zone/load_zones.csv.xlsx has 40
# entries, but not all of them have a public hourly data source we can use:
#   - 27 BAs come from EIA-930 main hourly table (CISO replaced with its
#     subregions — see CAISO_SUBREGIONS below)
#   - 4 CAISO subregions from the EIA-930 subregion table
#   - Entries we CANNOT build right now:
#       * CIPB, CIPV: CAISO Bay/Valley PG&E split — EIA-930 only gives us
#         combined PGAE
#       * PAID, PAUT, PAWY: PACE subdivisions — no public hourly source
#       * IPFE, IPMV, IPTV: IPCO subdivisions — no public hourly source
#       * AESO, BCHA, CFE: Canada/Mexico — not in PUDL
#       * SPPC: consolidated into NEVP years ago; not a live BA
#
# So this first pass produces 31 load zones. The skipped ones are listed
# in SKIPPED_ZONES below for transparency in the output log.
BA_LOAD_ZONES = [
    "AVA", "AZPS", "BANC", "BPAT", "CHPD", "DOPD", "EPE", "GCPD", "IID",
    "IPCO", "LDWP", "NEVP", "NWMT", "PACE", "PACW", "PGE", "PNM", "PSCO",
    "PSEI", "SCL", "SRP", "TEPC", "TIDC", "TPWR", "WACM", "WALC", "WAUW",
]

# CAISO (CISO) is intentionally excluded from BA_LOAD_ZONES above — we
# split it into these 4 subregions instead (sum ≈ CISO aggregate, verified
# within 0.5%). Each subregion is ~8784 hours with ~48 missing hours.
CAISO_SUBREGIONS = ["PGAE", "SCE", "SDGE", "VEA"]

# Zones from the existing load_zones.csv that this pipeline does NOT emit,
# with the reason. Used for logging/transparency only.
SKIPPED_ZONES = {
    "CIPB": "PG&E Bay — EIA-930 only has combined PGAE",
    "CIPV": "PG&E Valley — EIA-930 only has combined PGAE",
    "CISC": "Already covered by CAISO subregion SCE",   # name aliasing note
    "CISD": "Already covered by CAISO subregion SDGE",  # name aliasing note
    "PAID": "PACE-Idaho sub-area — no public hourly source",
    "PAUT": "PACE-Utah sub-area — no public hourly source",
    "PAWY": "PACE-Wyoming sub-area — no public hourly source",
    "IPFE": "IPCO Far East — no public hourly source",
    "IPMV": "IPCO Magic Valley — no public hourly source",
    "IPTV": "IPCO Treasure Valley — no public hourly source",
    "AESO": "Alberta — not in PUDL; needs separate source",
    "BCHA": "BC Hydro — not in PUDL; needs separate source",
    "CFE":  "CENACE Baja — not in PUDL; needs separate source",
    "SPPC": "Sierra Pacific — consolidated into NEVP; confirm with PI",
}

# GridPath load_zones.tab parameters. First-pass defaults borrowed from
# bluemarble's 2periods_new_build_2zones_loadcomponents example. The very
# large unserved-energy penalty lets GridPath fall back to unserved energy
# if it can't solve, rather than failing the run.
LOAD_ZONE_DEFAULTS = {
    "allow_overgeneration": 1,
    "overgeneration_penalty_per_mw": 99999999.0,
    "allow_unserved_energy": 1,
    "unserved_energy_penalty_per_mwh": 99999999.0,
    "unserved_energy_limit_mwh": ".",                 # '.' is GridPath's null
    "max_unserved_load_penalty_per_mw": 0.0,
    "max_unserved_load_limit_mw": ".",
    "export_penalty_cost_per_mwh": 0.0,
}

# First-pass scope decisions.
WEATHER_YEAR = 2024          # calendar year of the EIA-930 hourly shape
INVESTMENT_PERIOD = 2025     # the GridPath "period" these timepoints belong to
LOAD_COMPONENT = "all"       # single component for first pass

# =====================================================================
# PATHS
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]           # wecc_grid_model/
WECC_ROOT = REPO_ROOT.parent                               # wecc/
RAW_DIR = WECC_ROOT / "local_data" / "input_data" / "demand" / "raw"
DEFAULT_OUT = (
    WECC_ROOT / "local_data" / "input_data" / "demand"
    / "gridpath_inputs" / f"{INVESTMENT_PERIOD}_baseline"
)

# =====================================================================
# LOADERS — one function per source table
# =====================================================================

def load_eia930_ba_hourly(year: int) -> pd.DataFrame:
    """Load the already-downloaded EIA-930 BA hourly parquet and filter to
    the BA_LOAD_ZONES list + the requested year.

    Returns a long DataFrame with columns: zone, datetime_utc, load_mw.
    """
    src = RAW_DIR / f"eia930_hourly_wecc_{year}.parquet"
    print(f"[load] {src}")
    df = pd.read_parquet(src)

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)

    # Keep only the BAs we want and the target year. (The raw file already
    # has the year filter applied, but re-checking keeps this script safe
    # if the raw file is ever regenerated with a different scope.)
    df = df[
        df["balancing_authority_code_eia"].isin(BA_LOAD_ZONES)
        & (df["datetime_utc"].dt.year == year)
    ].copy()

    # Choose the best demand column per row, in this priority order:
    #   1. demand_imputed_pudl_mwh  (PUDL fills every hour — preferred)
    #   2. demand_adjusted_mwh      (EIA's cleaned version)
    #   3. demand_reported_mwh      (raw)
    df["load_mw"] = (
        df["demand_imputed_pudl_mwh"]
        .fillna(df["demand_adjusted_mwh"])
        .fillna(df["demand_reported_mwh"])
    )

    # Rename BA code → zone name so it matches the subregion data shape.
    df = df.rename(columns={"balancing_authority_code_eia": "zone"})

    out = df[["zone", "datetime_utc", "load_mw"]].copy()
    n_bas = out["zone"].nunique()
    print(f"[load] EIA-930 BA rows: {len(out):,}  distinct zones: {n_bas}")
    return out


def load_eia930_subregion_hourly(year: int) -> pd.DataFrame:
    """Pull CAISO subregion hourly demand directly from PUDL S3.

    This table is ~5.5M rows total across all subregions and all years, so
    pulling it from the web is fine for a one-off.

    Returns: zone, datetime_utc, load_mw.
    """
    # If we ever download this table into raw/ we can just read locally, but
    # for now the script reaches out to the PUDL S3 every time it runs.
    url = (
        f"{PUDL_S3_BASE}/{PUDL_RELEASE}/"
        "core_eia930__hourly_subregion_demand.parquet"
    )
    print(f"[fetch] CAISO subregion demand from {url}")

    # Only pull the columns we actually need.
    df = pd.read_parquet(
        url,
        columns=[
            "datetime_utc",
            "balancing_authority_code_eia",
            "balancing_authority_subregion_code_eia",
            "demand_reported_mwh",
        ],
    )
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)

    # Filter to CAISO, target year, and the 4 subregions we want.
    df = df[
        (df["balancing_authority_code_eia"] == "CISO")
        & (df["datetime_utc"].dt.year == year)
        & (df["balancing_authority_subregion_code_eia"].isin(CAISO_SUBREGIONS))
    ].copy()

    # The subregion table has no PUDL imputation layer — only
    # demand_reported_mwh. Forward-fill missing hours within each zone to
    # patch the ~48 gaps per subregion.
    df = df.rename(
        columns={
            "balancing_authority_subregion_code_eia": "zone",
            "demand_reported_mwh": "load_mw",
        }
    )
    df = df.sort_values(["zone", "datetime_utc"])

    missing_before = df["load_mw"].isna().sum()
    # Per-zone forward-fill then backward-fill, so the first-hour-missing
    # case is also handled.
    df["load_mw"] = df.groupby("zone")["load_mw"].transform(
        lambda s: s.ffill().bfill()
    )
    missing_after = df["load_mw"].isna().sum()

    print(
        f"[load] CAISO subregion rows: {len(df):,}  "
        f"gaps filled: {missing_before - missing_after} "
        f"(remaining NaN: {missing_after})"
    )

    out = df[["zone", "datetime_utc", "load_mw"]].copy()
    return out


# =====================================================================
# TRANSFORMERS — shape the long DataFrame into GridPath tables
# =====================================================================

def build_timepoint_index(year: int, period: int) -> pd.DataFrame:
    """Create the 8760/8784-row timepoint table for the year.

    GridPath's `timepoint` column is just an integer ID. Following the
    bluemarble ra_toolkit_sync_single_year convention, we use sequential
    integers (1, 2, 3, ...) rather than YYYYMMDDHH encoding — it's simpler,
    smaller, and matches the RA toolkit output the team already uses.

    We also keep a `datetime_utc` column so we can JOIN loads onto it later
    without having to decode the timepoint ID.

    Returns: timepoint (int), period (int), datetime_utc (UTC), month,
             day_of_month, hour_of_day, timepoint_weight, number_of_hours,
             previous_stage_timepoint_map.
    """
    # Build one UTC datetime per hour of the year. `inclusive="left"` so we
    # don't spill into the first hour of the next year.
    idx = pd.date_range(
        start=f"{year}-01-01 00:00:00+00:00",
        end=f"{year + 1}-01-01 00:00:00+00:00",
        freq="h",
        inclusive="left",
    )
    tp = pd.DataFrame({"datetime_utc": idx})

    # 1-based sequential integer, matches ra_toolkit convention.
    tp["timepoint"] = range(1, len(tp) + 1)
    tp["period"] = period

    # hour_of_day is 1..24 in the bluemarble example, not 0..23. So shift.
    tp["hour_of_day"] = (tp["datetime_utc"].dt.hour + 1).astype(float)
    tp["month"] = tp["datetime_utc"].dt.month
    tp["day_of_month"] = tp["datetime_utc"].dt.day

    # For a full-resolution single-year pass, each timepoint represents
    # one real hour, so weight = 1.0 and number_of_hours = 1.
    tp["timepoint_weight"] = 1.0
    tp["number_of_hours_in_timepoint"] = 1
    tp["previous_stage_timepoint_map"] = "."   # '.' = GridPath null

    # Reorder for GridPath, put the helper column at the end.
    cols = [
        "timepoint", "period", "timepoint_weight",
        "number_of_hours_in_timepoint", "previous_stage_timepoint_map",
        "month", "day_of_month", "hour_of_day", "datetime_utc",
    ]
    return tp[cols]


def build_load_zones_tab(zones: list[str]) -> pd.DataFrame:
    """Build load_zones.tab from the zone list and the shared defaults."""
    rows = []
    for z in sorted(zones):
        row = {"load_zone": z, **LOAD_ZONE_DEFAULTS}
        rows.append(row)
    return pd.DataFrame(rows)


def build_load_mw_tab(demand_long: pd.DataFrame,
                      timepoints: pd.DataFrame) -> pd.DataFrame:
    """Join long demand table onto timepoints to produce load_mw.tab.

    Input columns:
        demand_long: zone, datetime_utc, load_mw
        timepoints:  timepoint, datetime_utc  (other timepoint cols ignored)

    Output columns: load_zone, timepoint, load_component, load_mw.
    """
    tp_map = timepoints[["timepoint", "datetime_utc"]]
    merged = demand_long.merge(tp_map, on="datetime_utc", how="inner")

    out = pd.DataFrame({
        "load_zone": merged["zone"],
        "timepoint": merged["timepoint"],
        "load_component": LOAD_COMPONENT,
        "load_mw": merged["load_mw"].round(3),
    })
    # Sort so the output is deterministic and easy to eyeball.
    out = out.sort_values(["load_zone", "timepoint"], kind="stable")
    return out


# =====================================================================
# WRITERS
# =====================================================================

def write_tab(df: pd.DataFrame, path: Path) -> None:
    """Write a TAB-delimited .tab file (GridPath's required format)."""
    df.to_csv(path, sep="\t", index=False)
    print(f"[save] {path}  rows={len(df):,}  cols={len(df.columns)}")


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=WEATHER_YEAR,
                    help=f"Weather year (default {WEATHER_YEAR})")
    ap.add_argument("--period", type=int, default=INVESTMENT_PERIOD,
                    help=f"GridPath investment period label "
                         f"(default {INVESTMENT_PERIOD})")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help="Output directory for GridPath .tab files")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[config] year={args.year}  period={args.period}")
    print(f"[config] zones: {len(BA_LOAD_ZONES)} BAs + "
          f"{len(CAISO_SUBREGIONS)} CAISO subregions = "
          f"{len(BA_LOAD_ZONES) + len(CAISO_SUBREGIONS)} load zones")
    print(f"[config] out: {args.out_dir}")
    print(f"[config] load zones SKIPPED (documented in SKIPPED_ZONES):")
    for z, reason in SKIPPED_ZONES.items():
        print(f"           {z:6s}  {reason}")

    # ---------- 1. Load both demand sources ----------
    ba_df = load_eia930_ba_hourly(args.year)
    sub_df = load_eia930_subregion_hourly(args.year)
    demand = pd.concat([ba_df, sub_df], ignore_index=True)
    print(f"\n[combine] total demand rows: {len(demand):,}  "
          f"zones: {demand['zone'].nunique()}")

    # ---------- 2. Build the timepoint index ----------
    timepoints = build_timepoint_index(args.year, args.period)
    expected_hours = 8784 if args.year % 4 == 0 else 8760
    assert len(timepoints) == expected_hours, (
        f"Expected {expected_hours} hours for {args.year}, got {len(timepoints)}"
    )
    print(f"\n[timepoints] {len(timepoints):,} hourly timepoints for {args.year}")

    # ---------- 3. Build the three .tab outputs ----------
    all_zones = sorted(set(demand["zone"].unique()))
    load_zones_tab = build_load_zones_tab(all_zones)
    load_mw_tab = build_load_mw_tab(demand, timepoints)

    # Drop helper datetime_utc column before writing timepoints.tab —
    # GridPath won't recognize it. Save it separately for traceability.
    timepoint_map = timepoints[["timepoint", "datetime_utc"]].copy()
    timepoints_tab = timepoints.drop(columns=["datetime_utc"])

    # ---------- 4. Sanity checks before writing ----------
    expected_rows = len(all_zones) * len(timepoints)
    if len(load_mw_tab) != expected_rows:
        print(
            f"[WARN] load_mw.tab has {len(load_mw_tab):,} rows but "
            f"expected {expected_rows:,} "
            f"({len(all_zones)} zones × {len(timepoints)} timepoints). "
            "Some zone/timepoint combinations are missing."
        )
    nan_count = load_mw_tab["load_mw"].isna().sum()
    if nan_count:
        print(f"[WARN] load_mw.tab has {nan_count:,} NaN load values "
              f"— GridPath will reject these")

    # ---------- 5. Write the output files ----------
    write_tab(load_zones_tab, args.out_dir / "load_zones.tab")
    write_tab(timepoints_tab, args.out_dir / "timepoints.tab")
    write_tab(load_mw_tab, args.out_dir / "load_mw.tab")

    # Traceability companion (not a GridPath input — just for us).
    timepoint_map.to_csv(args.out_dir / "timepoint_map.csv", index=False)
    print(f"[save] {args.out_dir / 'timepoint_map.csv'}  "
          f"(companion, not a GridPath input)")

    # ---------- 6. Per-zone coverage summary ----------
    coverage = (
        load_mw_tab.groupby("load_zone")
        .agg(hours=("load_mw", "count"),
             nan=("load_mw", lambda s: s.isna().sum()),
             mean_mw=("load_mw", "mean"),
             peak_mw=("load_mw", "max"))
        .round(1)
    )
    print("\n[summary] coverage per zone:")
    print(coverage.to_string())

    print("\n[done] GridPath inputs written to:", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
