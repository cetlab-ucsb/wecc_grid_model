"""Build GridPath periods.tab and horizons files from our timepoint set.

Reads:
    local_data/input_data/demand/gridpath_inputs/2025_baseline/timepoint_map.csv
        (our companion file mapping timepoint int -> UTC datetime)

Writes (into the same gridpath_inputs/2025_baseline/ folder):
    periods.tab
    horizons_user_defined.tab
    horizon_user_defined_timepoints.tab

Reference files in bluemarble:
    gridpath_bluemarble/examples/2periods_new_build_2zones_loadcomponents/
        inputs/periods.tab, horizons_user_defined.tab,
        horizon_user_defined_timepoints.tab
    gridpath_bluemarble/examples/ra_toolkit_sync_single_year/
        .../inputs/periods.tab, horizons_user_defined.tab

First-pass scope (2026-04-24):
    * One investment period: 2025 (using 2024 weather year)
    * Two balancing types: `day` (366 daily horizons) + `year` (1 annual horizon)
    * All boundaries circular — per meeting notes
    * Horizon IDs: YYYYMMDD for days, YYYY for year
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# =====================================================================
# CONFIG
# =====================================================================

WEATHER_YEAR = 2024
INVESTMENT_PERIOD = 2025
# When 2030 is added, this becomes the NEXT period's start.
# For 2025 alone, we treat it as a 5-year block ending at 2030.
PERIOD_END_YEAR = 2030

# Balancing types to include at first pass. Order matters only for output
# readability; GridPath doesn't care.
BALANCING_TYPES = ["day", "year"]

BOUNDARY = "circular"   # matches meeting-note requirement

DISCOUNT_FACTOR = 1.0   # no discounting yet — placeholder for PI review

# =====================================================================
# PATHS
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
WECC_ROOT = REPO_ROOT.parent
IN_OUT_DIR = (
    WECC_ROOT / "local_data" / "input_data" / "demand"
    / "gridpath_inputs" / f"{INVESTMENT_PERIOD}_baseline"
)

# =====================================================================
# BUILDERS
# =====================================================================

def build_periods_tab() -> pd.DataFrame:
    """One row per investment period.

    Columns match bluemarble `2periods_new_build_2zones_loadcomponents`:
        period, discount_factor, period_start_year, period_end_year,
        hours_in_period_timepoints, prev_period

    `hours_in_period_timepoints` is the total weighted hours represented by
    the period's timepoints — for our full-hourly 2024 pass that's 8784
    (since every timepoint_weight = 1.0).
    """
    hours_in_period = 8784 if WEATHER_YEAR % 4 == 0 else 8760

    # Match bluemarble formatting exactly: floats for start/end years and
    # hours, integer for period, '.' for missing prev_period.
    row = {
        "period": INVESTMENT_PERIOD,
        "discount_factor": float(DISCOUNT_FACTOR),
        "period_start_year": float(INVESTMENT_PERIOD),
        "period_end_year": float(PERIOD_END_YEAR),
        "hours_in_period_timepoints": float(hours_in_period),
        "prev_period": ".",
    }
    return pd.DataFrame([row])


def build_horizons_and_mapping(
    timepoint_map: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build horizons_user_defined.tab + horizon_user_defined_timepoints.tab.

    Input: timepoint_map with columns [timepoint, datetime_utc].

    Strategy:
        * DAY horizons — one per calendar day. Horizon ID = YYYYMMDD as int.
          Each timepoint belongs to the day that contains its datetime_utc.
        * YEAR horizon — one for the whole weather year. Horizon ID = YYYY.
          Every timepoint belongs to it.

    Returns (horizons_df, mapping_df).
    """
    tm = timepoint_map.copy()
    tm["datetime_utc"] = pd.to_datetime(tm["datetime_utc"], utc=True)

    # ---- DAY horizons ----
    # YYYYMMDD as an integer — self-documenting and unique across years.
    tm["day_horizon"] = (
        tm["datetime_utc"].dt.strftime("%Y%m%d").astype(int)
    )

    # ---- YEAR horizons ----
    tm["year_horizon"] = tm["datetime_utc"].dt.year

    # Build the horizons table — one row per (horizon, balancing_type).
    day_horizons = (
        pd.DataFrame({
            "horizon": sorted(tm["day_horizon"].unique()),
            "balancing_type_horizon": "day",
            "boundary": BOUNDARY,
        })
    )
    year_horizons = (
        pd.DataFrame({
            "horizon": sorted(tm["year_horizon"].unique()),
            "balancing_type_horizon": "year",
            "boundary": BOUNDARY,
        })
    )
    horizons = pd.concat([day_horizons, year_horizons], ignore_index=True)

    # Build the mapping table — one row per (timepoint, balancing_type).
    day_map = tm.rename(columns={"day_horizon": "horizon"})[
        ["horizon", "timepoint"]
    ].copy()
    day_map["balancing_type_horizon"] = "day"

    year_map = tm.rename(columns={"year_horizon": "horizon"})[
        ["horizon", "timepoint"]
    ].copy()
    year_map["balancing_type_horizon"] = "year"

    mapping = pd.concat([day_map, year_map], ignore_index=True)
    # Reorder to match bluemarble convention: horizon, balancing_type, tp
    mapping = mapping[["horizon", "balancing_type_horizon", "timepoint"]]
    mapping = mapping.sort_values(
        ["balancing_type_horizon", "horizon", "timepoint"], kind="stable"
    )

    return horizons, mapping


# =====================================================================
# WRITER
# =====================================================================

def write_tab(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, sep="\t", index=False)
    print(f"[save] {path.name}  rows={len(df):,}  cols={len(df.columns)}")


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-out-dir", type=Path, default=IN_OUT_DIR,
                    help="Folder containing timepoint_map.csv; outputs go here too")
    args = ap.parse_args()

    tp_map_path = args.in_out_dir / "timepoint_map.csv"
    if not tp_map_path.exists():
        print(f"[error] Missing {tp_map_path}. Run transform_demand_to_gridpath.py first.")
        return 1

    timepoint_map = pd.read_csv(tp_map_path)
    print(f"[load] timepoint_map: {len(timepoint_map):,} timepoints")

    # ---- 1. periods.tab ----
    periods = build_periods_tab()
    write_tab(periods, args.in_out_dir / "periods.tab")

    # ---- 2. horizons + mapping ----
    horizons, mapping = build_horizons_and_mapping(timepoint_map)
    write_tab(horizons, args.in_out_dir / "horizons_user_defined.tab")
    write_tab(mapping, args.in_out_dir / "horizon_user_defined_timepoints.tab")

    # ---- Sanity ----
    day_count = (horizons["balancing_type_horizon"] == "day").sum()
    year_count = (horizons["balancing_type_horizon"] == "year").sum()
    expected_day = 366 if WEATHER_YEAR % 4 == 0 else 365
    assert day_count == expected_day, f"Expected {expected_day} day horizons, got {day_count}"
    assert year_count == 1, f"Expected 1 year horizon, got {year_count}"

    expected_rows = len(timepoint_map) * len(BALANCING_TYPES)
    assert len(mapping) == expected_rows, (
        f"Mapping has {len(mapping):,} rows, expected {expected_rows:,} "
        f"({len(timepoint_map)} timepoints × {len(BALANCING_TYPES)} balancing types)"
    )

    print(f"\n[summary] {day_count} day horizons, {year_count} year horizon, "
          f"{len(mapping):,} (timepoint, balancing_type) mappings")
    print(f"[done] Files written to: {args.in_out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
