"""Build the WECC demand CSVs in Blue Marble's `system_load` schema.

Replaces the now-deprecated transform_demand_to_gridpath.py. That script wrote
GridPath .tab files in the old 8,784-timepoint scheme. This script writes the
4-file system_load CSV bundle for the new 5-period × 24-rep-day temporal
structure (2,880 timepoints) — matching the team's bluemarble India model
template at local_data/system_load/.

Reads:
    local_data/input_data/demand/raw/eia930_hourly_wecc_2024.parquet
    PUDL S3:  core_eia930__hourly_subregion_demand.parquet  (CAISO subregions)
    local_data/input_data/temporal/5_5periods_2025-2045-24_1subproblem/
        structure.csv  — the 2,880 timepoints we need to populate

Writes (into local_data/input_data/system_load/):
    load_zones/1_WECCZones_NoOverGenPenalty.csv          (31 rows)
    system_load/1_EIA930_mid.csv                          (index, 1 row)
    system_load/load_components/1_EIA930_mid.csv          (31 rows)
    system_load/load_levels/1_EIA930_mid.csv              (89,280 rows)

DESIGN CHOICES (set 2026-04-29 with El):
    * REP-DAY SELECTION (system-wide, deterministic from 2024 data):
        - rep-day-01: the calendar day in 2024 whose 24-hour system-wide load
          profile minimizes sum-of-squared-difference from the month's mean
          profile. Weight = days_in_month - 1.
        - rep-day-02: the calendar day in 2024 with the highest single-hour
          system-wide load.  Weight = 1.
        - Selection is system-wide so all 31 zones share the same source
          dates (preserves cross-zone correlation in peak hours).
    * GROWTH SCALING (added 2026-04-30): per-(zone, period) multipliers from
      `local_data/input_data/demand/growth_factors.csv`, built by
      build_growth_factors.py.  Sources:
        - 2025, 2030: FERC-714 respondent forecast for most zones; WECC-wide
          flat rate substituted for TEPC (filing under-reports BA load).
        - 2035: only 6 zones (Avista, NorthWestern, 4 CAISO subregions) have
          coverage; everyone else is placeholder=1.0.
        - 2040, 2045: only CAISO has coverage; everyone else placeholder=1.0.
      PI input pending for 2035-2045 fallback (EIA AEO / WECC ADS).
    * SCENARIOS: only `mid` produced this session. `low`/`high` are a
      follow-up — schema/folder layout already supports them.
    * LOAD COMPONENTS: single component "all" per zone for first pass.
    * LOAD ZONE PENALTY: NoOverGenPenalty variant (overgeneration_penalty=0).
      Other variants (Soft/Mid/Hard) added later if the model needs them.
    * 31 ZONES: 27 EIA-930 BAs + 4 CAISO subregions, same set as session 1.
      See the BA_LOAD_ZONES / CAISO_SUBREGIONS / SKIPPED_ZONES blocks below.

OPEN QUESTIONS FOR PI (also tracked in PROGRESS_LOG.md):
    * How to define low/high scenarios — ±X% bands, or pull EIA AEO?
    * Source for 2040/2045 forecasts — extrapolate FERC-714, or use AEO/WECC ADS?
    * FERC-714 respondent → EIA-930 zone crosswalk for forecast scaling.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# =====================================================================
# CONFIG — ZONES (carried over from session 1; same 31 zones as before)
# =====================================================================

PUDL_S3_BASE = "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop"
PUDL_RELEASE = "v2026.1.0"

BA_LOAD_ZONES = [
    "AVA", "AZPS", "BANC", "BPAT", "CHPD", "DOPD", "EPE", "GCPD", "IID",
    "IPCO", "LDWP", "NEVP", "NWMT", "PACE", "PACW", "PGE", "PNM", "PSCO",
    "PSEI", "SCL", "SRP", "TEPC", "TIDC", "TPWR", "WACM", "WALC", "WAUW",
]
CAISO_SUBREGIONS = ["PGAE", "SCE", "SDGE", "VEA"]
ALL_ZONES = sorted(BA_LOAD_ZONES + CAISO_SUBREGIONS)  # 31 zones, sorted

# Penalty defaults for the "NoOverGenPenalty" variant — matches the bluemarble
# India `1_statesAsZones_NoOverGenPenalty.csv` template byte-for-byte except
# for the zone names themselves.
LOAD_ZONE_DEFAULTS = {
    "allow_overgeneration": 1,
    "overgeneration_penalty_per_mw": 0.0,
    "allow_unserved_energy": 1,
    "unserved_energy_penalty_per_mwh": 99999999,
    "unserved_energy_limit_mwh": "",
    "max_unserved_load_penalty_per_mw": 0,
    "max_unserved_load_limit_mw": "",
    "export_penalty_cost_per_mwh": 0,
    "unserved_energy_stats_threshold_mw": "",
}

# =====================================================================
# CONFIG — TEMPORAL + SCENARIO
# =====================================================================

WEATHER_YEAR = 2024            # year of the EIA-930 hourly shape we sample from
PERIODS = [2025, 2030, 2035, 2040, 2045]
# GROWTH_FACTOR replaced by a (zone, period) lookup from growth_factors.csv
# (built by build_growth_factors.py). See load_growth_factors() below.

# Names — follow bluemarble convention `N_<source>_<variant>.csv`.
SCENARIO_NAME = "EIA930_mid"
ZONES_SCENARIO_NAME = "WECCZones_NoOverGenPenalty"
LOAD_COMPONENT = "all"
WEATHER_ITERATION = 0          # bluemarble convention for non-MC runs
STAGE_ID = 1

# =====================================================================
# PATHS
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]           # wecc_grid_model/
WECC_ROOT = REPO_ROOT.parent                              # wecc/
RAW_DIR = WECC_ROOT / "local_data" / "input_data" / "demand" / "raw"
TEMPORAL_DIR = (
    WECC_ROOT / "local_data" / "input_data" / "temporal"
    / "5_5periods_2025-2045-24_1subproblem"
)
SYSTEM_LOAD_ROOT = (
    WECC_ROOT / "local_data" / "input_data" / "system_load"
)
GROWTH_FACTORS_PATH = (
    WECC_ROOT / "local_data" / "input_data" / "demand" / "growth_factors.csv"
)

# =====================================================================
# LOADERS
# =====================================================================

def load_eia930_ba_hourly(year: int) -> pd.DataFrame:
    """Load EIA-930 BA hourly data, filter to WECC BAs and the given year.

    Reuses the column-coalescing logic from session 1: prefer
    demand_imputed_pudl_mwh, fall back to adjusted, then reported.

    Returns: zone, datetime_utc, load_mw  (long form).
    """
    src = RAW_DIR / f"eia930_hourly_wecc_{year}.parquet"
    print(f"[load] BA hourly: {src.name}")
    df = pd.read_parquet(src)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)

    df = df[
        df["balancing_authority_code_eia"].isin(BA_LOAD_ZONES)
        & (df["datetime_utc"].dt.year == year)
    ].copy()

    df["load_mw"] = (
        df["demand_imputed_pudl_mwh"]
        .fillna(df["demand_adjusted_mwh"])
        .fillna(df["demand_reported_mwh"])
    )
    df = df.rename(columns={"balancing_authority_code_eia": "zone"})
    out = df[["zone", "datetime_utc", "load_mw"]].copy()
    print(f"[load]   rows={len(out):,}  zones={out['zone'].nunique()}")
    return out


def load_eia930_subregion_hourly(year: int) -> pd.DataFrame:
    """Pull CAISO subregion hourly demand from PUDL S3 (no local copy yet).

    The subregion table has no PUDL imputation layer — only demand_reported.
    We forward-/back-fill missing hours within each zone (~48 gaps per
    subregion in 2024, < 0.5% of the year).

    Returns: zone, datetime_utc, load_mw  (long form).
    """
    url = (
        f"{PUDL_S3_BASE}/{PUDL_RELEASE}/"
        "core_eia930__hourly_subregion_demand.parquet"
    )
    print(f"[load] CAISO subregions from S3 (one-shot)")
    df = pd.read_parquet(url, columns=[
        "datetime_utc", "balancing_authority_code_eia",
        "balancing_authority_subregion_code_eia", "demand_reported_mwh",
    ])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df[
        (df["balancing_authority_code_eia"] == "CISO")
        & (df["datetime_utc"].dt.year == year)
        & (df["balancing_authority_subregion_code_eia"].isin(CAISO_SUBREGIONS))
    ].copy()

    df = df.rename(columns={
        "balancing_authority_subregion_code_eia": "zone",
        "demand_reported_mwh": "load_mw",
    })
    df = df.sort_values(["zone", "datetime_utc"])

    missing_before = df["load_mw"].isna().sum()
    df["load_mw"] = df.groupby("zone")["load_mw"].transform(
        lambda s: s.ffill().bfill()
    )
    print(f"[load]   rows={len(df):,}  gaps filled={missing_before}")
    return df[["zone", "datetime_utc", "load_mw"]]


def load_growth_factors() -> pd.DataFrame:
    """Read the (zone, period) growth-factor table built by
    build_growth_factors.py."""
    print(f"[load] growth_factors.csv")
    gf = pd.read_csv(GROWTH_FACTORS_PATH)
    print(f"[load]   rows={len(gf):,}  zones={gf['zone'].nunique()}  periods={gf['period'].nunique()}")
    print(f"[load]   sources: {gf['source'].value_counts().to_dict()}")
    return gf[["zone", "period", "growth_factor"]]


def load_temporal_structure() -> pd.DataFrame:
    """Read the 2,880-row structure.csv produced by build_temporal.py.

    Used as the master timepoint list — every timepoint here gets a load row
    for every zone.

    Returns: the full structure DataFrame, with timepoint as int.
    """
    src = TEMPORAL_DIR / "structure.csv"
    print(f"[load] temporal structure: {src.name}")
    df = pd.read_csv(src)
    print(f"[load]   timepoints={len(df):,}  periods={df['period'].nunique()}")
    return df


# =====================================================================
# REP-DAY SELECTION
# =====================================================================

def select_rep_days(demand: pd.DataFrame) -> pd.DataFrame:
    """Pick the (avg, peak) source date in WEATHER_YEAR for each month.

    Aggregates load to system-wide hourly (sum across all 31 zones), then for
    each month picks:
        - avg day: minimizes SSD vs the month's mean 24-hour profile
        - peak day: has the highest single-hour load anywhere in the month

    Returns a DataFrame with columns:
        month, rep_day_idx, source_date, peak_load_mw, mean_diff_mw

    `rep_day_idx` is 1 for avg, 2 for peak — matching the timepoint encoding.
    """
    # System-wide load per hour (sum of all 31 zones)
    sys = (
        demand.groupby("datetime_utc")["load_mw"].sum()
        .reset_index().rename(columns={"load_mw": "sys_mw"})
    )
    sys["date"] = sys["datetime_utc"].dt.date
    sys["month"] = sys["datetime_utc"].dt.month
    sys["hour"] = sys["datetime_utc"].dt.hour

    rows = []
    for m in range(1, 13):
        month_data = sys[sys["month"] == m]

        # 24-column-by-N-row pivot, one row per real date in the month.
        # If a date is missing any of its 24 hours, drop it from candidates.
        pivot = month_data.pivot(index="date", columns="hour", values="sys_mw")
        pivot = pivot.dropna()

        # Average day = closest to the mean profile in SSD distance.
        mean_profile = pivot.mean(axis=0)
        ssd = ((pivot - mean_profile) ** 2).sum(axis=1)
        avg_date = ssd.idxmin()

        # Peak day = highest single-hour load.
        peak_date = pivot.max(axis=1).idxmax()

        # If the avg and peak end up being the same date (rare), nudge avg
        # to the second-best — we want two distinct source dates per month.
        if avg_date == peak_date:
            ssd_sorted = ssd.sort_values()
            avg_date = ssd_sorted.index[1]

        rows.append({
            "month": m, "rep_day_idx": 1, "source_date": avg_date,
            "ssd_from_mean": float(ssd.loc[avg_date]),
            "day_peak_mw": float(pivot.loc[avg_date].max()),
        })
        rows.append({
            "month": m, "rep_day_idx": 2, "source_date": peak_date,
            "ssd_from_mean": float(ssd.loc[peak_date]),
            "day_peak_mw": float(pivot.loc[peak_date].max()),
        })

    return pd.DataFrame(rows)


# =====================================================================
# OUTPUT BUILDERS
# =====================================================================

def build_load_zones() -> pd.DataFrame:
    """One row per zone with the penalty defaults."""
    rows = [{"load_zone": z, **LOAD_ZONE_DEFAULTS} for z in ALL_ZONES]
    return pd.DataFrame(rows)


def build_load_components() -> pd.DataFrame:
    """One row per zone with load_component='all', load_level_default blank."""
    rows = [
        {"load_zone": z, "load_component": LOAD_COMPONENT, "load_level_default": ""}
        for z in ALL_ZONES
    ]
    return pd.DataFrame(rows)


def build_index_file() -> pd.DataFrame:
    """The tiny 2-column index pointing at the load_components and load_levels
    sub-scenarios that combine to make this named scenario."""
    return pd.DataFrame([{
        "load_components_scenario_id": 1,
        "load_levels_scenario_id": 1,
    }])


def build_load_levels(demand: pd.DataFrame,
                      structure: pd.DataFrame,
                      rep_days: pd.DataFrame,
                      growth: pd.DataFrame) -> pd.DataFrame:
    """Build the 89,280-row load_levels.csv.

    Schema: load_zone, weather_iteration, stage_id, timepoint, load_component, load_mw

    Approach:
        1. For each (month, rep_day_idx), get the source date in 2024.
        2. For that source date, every (zone, hour-of-day) has a load value.
        3. For each period in PERIODS, the same source-date hours get reused
           under different timepoint IDs.
        4. Apply per-(zone, period) growth factor from growth_factors.csv.

    Decode the timepoint: YYYYMMDDHH where DD is the rep_day_idx (01 or 02).
    """
    # Decode timepoints in structure to (month, rep_day_idx, hour_of_day).
    s = structure.copy()
    s["timepoint_str"] = s["timepoint"].astype(str)
    s["rep_day_idx"] = s["timepoint_str"].str[6:8].astype(int)
    # `month`, `hour_of_day`, `period` are already columns in structure.csv

    # Join to rep_days to get the source_date for each timepoint.
    s = s.merge(
        rep_days[["month", "rep_day_idx", "source_date"]],
        on=["month", "rep_day_idx"], how="left",
    )

    # Build a (zone, source_date, hour_of_day) → load lookup from 2024 data.
    d = demand.copy()
    d["source_date"] = d["datetime_utc"].dt.date
    d["hour_of_day"] = d["datetime_utc"].dt.hour + 1   # 1..24 to match
    lookup = d[["zone", "source_date", "hour_of_day", "load_mw"]]

    # For each timepoint, we need 31 rows — one per zone. So expand by zone.
    expanded = (
        s[["timepoint", "period", "month", "rep_day_idx",
           "hour_of_day", "source_date"]]
        .merge(pd.DataFrame({"zone": ALL_ZONES}), how="cross")
    )
    merged = expanded.merge(
        lookup, on=["zone", "source_date", "hour_of_day"], how="left",
    )

    # Apply (zone, period) growth factor.
    merged = merged.merge(
        growth, left_on=["zone", "period"], right_on=["zone", "period"],
        how="left",
    )
    assert merged["growth_factor"].isna().sum() == 0, "missing growth factors"
    merged["load_mw"] = (merged["load_mw"] * merged["growth_factor"]).round(3)

    # Final schema in bluemarble order.
    out = pd.DataFrame({
        "load_zone": merged["zone"],
        "weather_iteration": WEATHER_ITERATION,
        "stage_id": STAGE_ID,
        "timepoint": merged["timepoint"],
        "load_component": LOAD_COMPONENT,
        "load_mw": merged["load_mw"],
    })
    out = out.sort_values(["load_zone", "timepoint"], kind="stable")
    return out


# =====================================================================
# WRITER
# =====================================================================

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[save] {path.relative_to(WECC_ROOT)}  rows={len(df):,}")


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=WEATHER_YEAR)
    ap.add_argument("--scenario", type=str, default=SCENARIO_NAME)
    args = ap.parse_args()

    print(f"[config] weather_year={args.year}  scenario={args.scenario}")
    print(f"[config] zones={len(ALL_ZONES)}  periods={PERIODS}  growth=(zone,period) lookup")

    # ---------- 1. Load demand ----------
    ba = load_eia930_ba_hourly(args.year)
    sub = load_eia930_subregion_hourly(args.year)
    demand = pd.concat([ba, sub], ignore_index=True)
    assert demand["zone"].nunique() == len(ALL_ZONES), (
        f"Got {demand['zone'].nunique()} zones, expected {len(ALL_ZONES)}"
    )
    print(f"[combine] {len(demand):,} (zone, hour) rows  zones={demand['zone'].nunique()}")

    # ---------- 2. Load temporal structure + growth factors ----------
    structure = load_temporal_structure()
    growth = load_growth_factors()

    # ---------- 3. Pick rep days ----------
    rep_days = select_rep_days(demand)
    print("\n[rep-days] selected from", args.year, "(system-wide):")
    print(rep_days.to_string(index=False))

    # ---------- 4. Build the four output DataFrames ----------
    load_zones = build_load_zones()
    load_components = build_load_components()
    load_levels = build_load_levels(demand, structure, rep_days, growth)
    index_file = build_index_file()

    # ---------- 5. Sanity checks ----------
    expected_levels = len(ALL_ZONES) * len(structure)
    assert len(load_levels) == expected_levels, (
        f"load_levels has {len(load_levels):,} rows, expected {expected_levels:,}"
    )
    nan_count = load_levels["load_mw"].isna().sum()
    assert nan_count == 0, f"{nan_count} NaN load values — fix before writing"

    # ---------- 6. Write all four files ----------
    out_zones = SYSTEM_LOAD_ROOT / "load_zones" / f"1_{ZONES_SCENARIO_NAME}.csv"
    out_index = SYSTEM_LOAD_ROOT / "system_load" / f"1_{args.scenario}.csv"
    out_components = (
        SYSTEM_LOAD_ROOT / "system_load" / "load_components"
        / f"1_{args.scenario}.csv"
    )
    out_levels = (
        SYSTEM_LOAD_ROOT / "system_load" / "load_levels"
        / f"1_{args.scenario}.csv"
    )

    write_csv(load_zones, out_zones)
    write_csv(index_file, out_index)
    write_csv(load_components, out_components)
    write_csv(load_levels, out_levels)

    # ---------- 7. Summary ----------
    print(f"\n[summary]")
    print(f"  Zones:        {len(ALL_ZONES)}")
    print(f"  Timepoints:   {len(structure):,}")
    print(f"  Load rows:    {len(load_levels):,}  ({len(ALL_ZONES)} × {len(structure)})")
    print(f"  Periods:      {PERIODS}")
    # Per-period total energy (system-wide, weighted by timepoint_weight)
    lvl = load_levels.merge(
        structure[["timepoint", "period", "timepoint_weight"]], on="timepoint",
    )
    lvl["mwh"] = lvl["load_mw"] * lvl["timepoint_weight"]
    period_twh = lvl.groupby("period")["mwh"].sum() / 1e6
    print(f"  Period TWh:")
    for p, t in period_twh.items():
        print(f"     {p}: {t:7.1f} TWh")
    print(f"  Mean MW/zone: {load_levels['load_mw'].mean():,.0f}")
    print(f"  Peak MW:      {load_levels['load_mw'].max():,.0f}")
    print(f"[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
