"""Build the WECC temporal-structure CSVs in Guille's Blue Marble template format.

Produces 6 CSV files matching the schema of Guille's reference folder:
    local_data/temporal/4_5periods_2020-2040-24_1subproblem/

Output goes to:
    local_data/input_data/temporal/5_5periods_2025-2045-24_1subproblem/
    (input_data/ syncs to the team's Google Drive)
        period_params.csv
        horizon_params.csv
        horizon_timepoints.csv
        structure.csv
        superperiods.csv         (empty — header only)
        iterations.csv           (empty — header only)

These CSVs are the *intermediate* Blue Marble format. A later converter step
(or the team's pipeline) is expected to translate them into the GridPath .tab
files (timepoints.tab, periods.tab, horizons_user_defined.tab, etc.).

DESIGN CHOICES (set 2026-04-29 with El, mostly placeholder):
    * 5 investment periods: 2025, 2030, 2035, 2040, 2045
        - Each represents a 5-year block (period_end_year = period + 5)
        - Discount factor = (1.07)^(-5*N) where N is the period index from 2025
        - i.e. 7%/yr discount rate compounded over 5-year periods
    * Calendar year (Jan–Dec), not Apr–Mar fiscal year
    * 2 representative days per month × 12 months × 24 hours = 576 timepoints/period
        - 5 periods × 576 = 2,880 total timepoints
    * Timepoint encoding: YYYYMMDDHH (10-digit int)
        - YYYY: investment-period year (e.g. 2025)
        - MM:   month (01-12)
        - DD:   rep-day index within the month (01 or 02)  -- NOT actual day-of-month
        - HH:   hour-of-day (01-24, hour-ending)
    * Timepoint weights match Guille's convention:
        - rep-day-01 weight = days_in_month - 1   (the "average" day; placeholder)
        - rep-day-02 weight = 1                   (the "peak" day; placeholder)
        - Sum × 24h per month ≈ real hours in month → period total = 8760
    * 3 balancing types: day, month, year — all boundary=circular
        - day:   YYYYMMDD (8-digit) — 5×12×2 = 120 horizons
        - month: YYYYMM   (6-digit) — 5×12   = 60 horizons
        - year:  YYYY     (4-digit) — 5      = 5 horizons
    * Single subproblem, single stage (subproblem_id=1, stage_id=1)
    * No spinup/lookahead, no stage linking
    * superperiods.csv and iterations.csv left empty (just headers) — match reference

REFERENCE: local_data/temporal/4_5periods_2020-2040-24_1subproblem/
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

import pandas as pd

# =====================================================================
# CONFIG
# =====================================================================

PERIODS = [2025, 2030, 2035, 2040, 2045]
PERIOD_LENGTH_YEARS = 5
DISCOUNT_RATE = 0.07          # 7% per year, compounded over the period

REP_DAYS_PER_MONTH = 2        # rep-day indices 01 and 02
HOURS_PER_DAY = 24

MONTHS = list(range(1, 13))   # 1..12, calendar year

# Use a fixed (non-leap) days-per-month vector so February = 28 every period.
# These rep days don't track real dates — we just need consistent weights.
DAYS_IN_MONTH = {m: calendar.monthrange(2025, m)[1] for m in MONTHS}
# {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

BOUNDARY = "circular"
SUBPROBLEM_ID = 1
STAGE_ID = 1

# Output folder name follows Guille's convention:
#   [index]_[N]periods_[start]-[end]-[total_rep_days]_[N]subproblem
OUTPUT_FOLDER_NAME = "5_5periods_2025-2045-24_1subproblem"

# =====================================================================
# PATHS
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
WECC_ROOT = REPO_ROOT.parent
OUTPUT_DIR = (
    WECC_ROOT / "local_data" / "input_data" / "temporal" / OUTPUT_FOLDER_NAME
)

# =====================================================================
# BUILDERS
# =====================================================================

def build_period_params() -> pd.DataFrame:
    """One row per investment period.

    Columns: period, discount_factor, period_start_year, period_end_year

    discount_factor for period N (counting from PERIODS[0]) is:
        (1 + DISCOUNT_RATE) ^ (-PERIOD_LENGTH_YEARS * N)
    so the first period is 1.0, each subsequent period is discounted further.
    period_end_year = period + PERIOD_LENGTH_YEARS (every block 5 years; the
    last period 2045 ends at 2050).
    """
    rows = []
    for idx, period in enumerate(PERIODS):
        discount_factor = (1 + DISCOUNT_RATE) ** (-PERIOD_LENGTH_YEARS * idx)
        rows.append({
            "period": period,
            "discount_factor": discount_factor,
            "period_start_year": period,
            "period_end_year": period + PERIOD_LENGTH_YEARS,
        })
    return pd.DataFrame(rows)


def build_structure() -> pd.DataFrame:
    """Build the timepoints table — one row per (period, month, rep_day, hour).

    Schema matches Guille's reference structure.csv exactly. Some metadata
    columns (year, day_of_month, timestamp, ignore_horizon_day) are left blank
    to match the reference; only `month` and `hour_of_day` of the optional
    auxiliaries get filled.

    timepoint_weight: rep-day 01 = days_in_month - 1, rep-day 02 = 1.
    """
    rows = []
    for period in PERIODS:
        for month in MONTHS:
            n_days = DAYS_IN_MONTH[month]
            for rep_day in range(1, REP_DAYS_PER_MONTH + 1):
                # Rep-day weights: day-01 carries (n_days - 1), day-02 carries 1.
                weight = (n_days - 1) if rep_day == 1 else 1
                for hour in range(1, HOURS_PER_DAY + 1):
                    timepoint = int(
                        f"{period:04d}{month:02d}{rep_day:02d}{hour:02d}"
                    )
                    rows.append({
                        "subproblem_id": SUBPROBLEM_ID,
                        "stage_id": STAGE_ID,
                        "timepoint": timepoint,
                        "period": period,
                        "number_of_hours_in_timepoint": 1,
                        "timepoint_weight": float(weight),
                        "previous_stage_timepoint_map": "",
                        "spinup_or_lookahead": 0,
                        "linked_timepoint": "",
                        "year": "",          # left blank to match reference
                        "month": month,
                        "day_of_month": "",  # left blank to match reference
                        "hour_of_day": hour,
                        "timestamp": "",     # left blank to match reference
                        "ignore_horizon_day": "",  # left blank to match reference
                    })
    return pd.DataFrame(rows)


def build_horizon_params() -> pd.DataFrame:
    """One row per (balancing_type_horizon, horizon).

    Three balancing types:
      * day   — YYYYMMDD where DD is rep-day index (01 or 02), 120 rows
      * month — YYYYMM, 60 rows
      * year  — YYYY, 5 rows

    All boundaries are 'circular' per meeting notes.
    """
    rows = []

    # Day horizons — 5 periods × 12 months × 2 rep days
    for period in PERIODS:
        for month in MONTHS:
            for rep_day in range(1, REP_DAYS_PER_MONTH + 1):
                horizon = int(f"{period:04d}{month:02d}{rep_day:02d}")
                rows.append({
                    "balancing_type_horizon": "day",
                    "horizon": horizon,
                    "boundary": BOUNDARY,
                })

    # Month horizons — 5 periods × 12 months
    for period in PERIODS:
        for month in MONTHS:
            horizon = int(f"{period:04d}{month:02d}")
            rows.append({
                "balancing_type_horizon": "month",
                "horizon": horizon,
                "boundary": BOUNDARY,
            })

    # Year horizons — one per period
    for period in PERIODS:
        rows.append({
            "balancing_type_horizon": "year",
            "horizon": period,
            "boundary": BOUNDARY,
        })

    return pd.DataFrame(rows)


def build_horizon_timepoints() -> pd.DataFrame:
    """One row per (stage, balancing_type, horizon) giving the start/end timepoint.

    Schema: stage_id, balancing_type_horizon, horizon,
            tmp_start, tmp_start_spinup_or_lookahead,
            tmp_end,   tmp_end_spinup_or_lookahead

    Spinup/lookahead always 0 (we don't model spinup yet).

    Day horizon spans hours 01-24 of one rep-day:
        tmp_start = YYYYMMDD01,  tmp_end = YYYYMMDD24

    Month horizon spans both rep-days:
        tmp_start = YYYYMM0101,  tmp_end = YYYYMM0224

    Year horizon spans Jan rep-day-01 hour-01 → Dec rep-day-02 hour-24:
        tmp_start = YYYY010101,  tmp_end = YYYY120224
    """
    rows = []

    # Day horizons
    for period in PERIODS:
        for month in MONTHS:
            for rep_day in range(1, REP_DAYS_PER_MONTH + 1):
                horizon = int(f"{period:04d}{month:02d}{rep_day:02d}")
                tmp_start = int(f"{period:04d}{month:02d}{rep_day:02d}01")
                tmp_end = int(f"{period:04d}{month:02d}{rep_day:02d}24")
                rows.append({
                    "stage_id": STAGE_ID,
                    "balancing_type_horizon": "day",
                    "horizon": horizon,
                    "tmp_start": tmp_start,
                    "tmp_start_spinup_or_lookahead": 0,
                    "tmp_end": tmp_end,
                    "tmp_end_spinup_or_lookahead": 0,
                })

    # Month horizons
    for period in PERIODS:
        for month in MONTHS:
            horizon = int(f"{period:04d}{month:02d}")
            tmp_start = int(f"{period:04d}{month:02d}0101")
            tmp_end = int(f"{period:04d}{month:02d}0224")
            rows.append({
                "stage_id": STAGE_ID,
                "balancing_type_horizon": "month",
                "horizon": horizon,
                "tmp_start": tmp_start,
                "tmp_start_spinup_or_lookahead": 0,
                "tmp_end": tmp_end,
                "tmp_end_spinup_or_lookahead": 0,
            })

    # Year horizons
    for period in PERIODS:
        horizon = period
        tmp_start = int(f"{period:04d}010101")
        tmp_end = int(f"{period:04d}120224")
        rows.append({
            "stage_id": STAGE_ID,
            "balancing_type_horizon": "year",
            "horizon": horizon,
            "tmp_start": tmp_start,
            "tmp_start_spinup_or_lookahead": 0,
            "tmp_end": tmp_end,
            "tmp_end_spinup_or_lookahead": 0,
        })

    return pd.DataFrame(rows)


def build_empty_superperiods() -> pd.DataFrame:
    """Header-only file. Used to group periods into super-periods. Empty for now."""
    return pd.DataFrame(columns=["superperiod", "period"])


def build_empty_iterations() -> pd.DataFrame:
    """Header-only file. Used for monte carlo iterations. Empty for now."""
    return pd.DataFrame(columns=[
        "weather_iteration", "hydro_iteration", "availability_iteration",
    ])


# =====================================================================
# WRITER
# =====================================================================

def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"[save] {path.name:30s}  rows={len(df):,}  cols={len(df.columns)}")


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                    help="Folder to write the 6 CSVs into (will be created).")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[init] Writing to {args.output_dir}")

    # ---- Build all six DataFrames ----
    period_params      = build_period_params()
    structure          = build_structure()
    horizon_params     = build_horizon_params()
    horizon_timepoints = build_horizon_timepoints()
    superperiods       = build_empty_superperiods()
    iterations         = build_empty_iterations()

    # ---- Sanity checks before writing ----
    expected_timepoints = (
        len(PERIODS) * len(MONTHS) * REP_DAYS_PER_MONTH * HOURS_PER_DAY
    )
    assert len(structure) == expected_timepoints, (
        f"structure has {len(structure)} rows, expected {expected_timepoints}"
    )

    # Each period's weighted hours should sum to 8760 (non-leap calendar year)
    weighted_hours_per_period = (
        structure
        .assign(weighted=structure["timepoint_weight"]
                          * structure["number_of_hours_in_timepoint"])
        .groupby("period")["weighted"].sum()
    )
    for period, hours in weighted_hours_per_period.items():
        assert hours == 8760, (
            f"Period {period}: weighted hours = {hours}, expected 8760"
        )

    expected_horizons = (
        len(PERIODS) * len(MONTHS) * REP_DAYS_PER_MONTH  # day
        + len(PERIODS) * len(MONTHS)                     # month
        + len(PERIODS)                                   # year
    )
    assert len(horizon_params) == expected_horizons, (
        f"horizon_params has {len(horizon_params)} rows, expected {expected_horizons}"
    )
    assert len(horizon_timepoints) == expected_horizons

    # Discount factor for first period must be exactly 1.0
    assert period_params.iloc[0]["discount_factor"] == 1.0

    # ---- Write all six files ----
    write_csv(period_params,      args.output_dir / "period_params.csv")
    write_csv(structure,          args.output_dir / "structure.csv")
    write_csv(horizon_params,     args.output_dir / "horizon_params.csv")
    write_csv(horizon_timepoints, args.output_dir / "horizon_timepoints.csv")
    write_csv(superperiods,       args.output_dir / "superperiods.csv")
    write_csv(iterations,         args.output_dir / "iterations.csv")

    print(f"\n[summary]")
    print(f"  Periods:          {len(PERIODS)} ({PERIODS[0]}–{PERIODS[-1]})")
    print(f"  Timepoints:       {len(structure):,}")
    print(f"  Day horizons:     {len(PERIODS) * len(MONTHS) * REP_DAYS_PER_MONTH}")
    print(f"  Month horizons:   {len(PERIODS) * len(MONTHS)}")
    print(f"  Year horizons:    {len(PERIODS)}")
    print(f"  Hours per period: 8760 (verified)")
    print(f"[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
