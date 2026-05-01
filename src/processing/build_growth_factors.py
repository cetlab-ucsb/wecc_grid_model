"""Build per-(zone, period) demand growth multipliers from FERC-714 forecasts.

Output: `local_data/input_data/demand/growth_factors.csv`
Schema: zone, period, growth_factor, source, notes

WHAT IT DOES:

For each of the 31 zones and 5 periods (2025, 2030, 2035, 2040, 2045), produces
a multiplicative growth factor to apply to that zone's 2024 actual load.  The
factor is the FERC-714 respondent's forecast growth rate from 2025 to the
target period.

    growth_factor[zone, period] = forecast[respondent, period]
                                  / forecast[respondent, 2025]

For BA_SUBSET zones (PACE/PACW) and SUBREGION_OF_AGGREGATE zones (PGAE/SCE/
SDGE/VEA), the same parent-respondent rate is applied to each split zone --
i.e., we assume the parent's growth rate applies uniformly across its sub-
zones.  See build_ferc714_crosswalk.py for the share weights.

WHY 2025 IS THE DENOMINATOR (NOT 2024 ACTUAL):
    - FERC respondents file forecasts starting from the report year forward.
      The earliest forecast year for a 2024 report is 2025.
    - We could divide forecast / 2024 EIA-930 actual to get an absolute
      multiplier, but that conflates two error sources:
        (1) FERC respondent's planning area mismatch with our zone definition
        (2) FERC's modeling vs reality
      Using forecast/forecast as a *rate* eliminates (1).  Then we apply the
      rate to our trusted 2024 actual.
    - Net effect: 2025 multiplier = 1.0 (zone's 2024 weather-shape used
      as-is for the 2025 period); 2030 multiplier = the respondent's
      5-year growth rate; etc.

UNITS DETECTION (PUDL net_demand_forecast_mwh column):
    The PUDL column is *labeled* MWh but contains a mix of MWh and GWh
    depending on respondent. Detected via load-factor sanity check: real-world
    annual load factors are 0.3–0.7. If `mwh / (peak_mw × 8760)` < 0.05,
    the column is in GWh and we multiply by 1000.

TEPC SPECIAL CASE:
    Tucson Electric Power's FERC-714 filing reports ~2 TWh annual energy in
    2025 (post-units-correction). EIA-930 actual for the TEPC BA was ~14.5 TWh
    in 2024. TEPC's filing appears to cover only their internal customer load,
    not the full BA's load.  Their own forecast growth rate may still be
    valid for those customers, but using it would only scale 14% of TEPC's
    real load — leaving the rest at 2024 levels.  Following El's call:
    use the system-wide WECC FERC-714 growth rate as a flat substitute for
    TEPC.  Documented in the output `notes` column.

2035 / 2040 / 2045 COVERAGE:
    Most FERC-714 respondents file 5-year forecasts (through ~2030 from the
    2024 report).  Only Avista, NorthWestern, and CAISO go to ~2034.  Nothing
    publicly available in PUDL covers 2040–2045.  For the outer periods we
    write growth_factor=1.0 (i.e., 2024 weather + 2024 levels) with
    source='placeholder' so they're easy to find later.  PI to advise on
    EIA AEO / WECC ADS source for those periods.

OPEN PI QUESTIONS:
    * Is per-respondent growth rate the right level, or should we compute
      a single WECC-wide rate and apply uniformly?
    * How to handle 2035–2045 for the 24+ respondents that don't go that
      far out?
    * TEPC: confirm flat rate is the right call, or instruct otherwise?
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# =====================================================================
# PATHS / CONSTANTS
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
WECC_ROOT = REPO_ROOT.parent
DEMAND_DIR = WECC_ROOT / "local_data" / "input_data" / "demand"

PUDL_S3_BASE = "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop"
PUDL_RELEASE = "v2026.1.0"

PERIODS = [2025, 2030, 2035, 2040, 2045]
DENOM_YEAR = 2025  # forecast year used as denominator (rate baseline)

# Load-factor heuristic for unit detection. Real annual LF is ~0.4–0.6.
# If MWh value implies LF < 0.05 against peak, it's actually in GWh.
LF_THRESHOLD = 0.05
HOURS_PER_YEAR = 8760

# Zones that get the WECC-wide flat rate instead of their respondent's own.
FLAT_RATE_ZONES = ["TEPC"]


# =====================================================================
# LOADERS
# =====================================================================

def load_crosswalk() -> pd.DataFrame:
    """Read the (zone → respondent) crosswalk built by
    build_ferc714_crosswalk.py."""
    src = DEMAND_DIR / "ferc714_to_zone_crosswalk.csv"
    print(f"[load] {src.name}")
    return pd.read_csv(src)


def load_forecasts() -> pd.DataFrame:
    """Pull FERC-714 forecasts from PUDL S3, latest report year per
    (respondent, forecast_year). Restrict to our periods of interest."""
    url = (
        f"{PUDL_S3_BASE}/{PUDL_RELEASE}/"
        "core_ferc714__yearly_planning_area_demand_forecast.parquet"
    )
    print(f"[load] FERC-714 forecasts from PUDL S3")
    fc = pd.read_parquet(url)
    fc = fc[fc["forecast_year"].isin(PERIODS + [DENOM_YEAR])]
    fc = (
        fc.sort_values("report_year")
        .drop_duplicates(["respondent_id_ferc714", "forecast_year"], keep="last")
    )
    print(f"[load]   rows={len(fc):,}  respondents={fc['respondent_id_ferc714'].nunique()}")
    return fc


# =====================================================================
# UNIT NORMALIZATION
# =====================================================================

def normalize_units(fc: pd.DataFrame) -> pd.DataFrame:
    """Per respondent, infer whether net_demand_forecast_mwh is in MWh or
    GWh by computing implied load factor against the peak-MW column.
    If LF << 0.3, multiply by 1000 to convert to MWh.

    Decision is per respondent (not per row), using their 2025 forecast row
    as the reference, since LF should be stable across forecast years.
    """
    fc = fc.copy()
    fc["energy_mwh_normalized"] = fc["net_demand_forecast_mwh"]

    ref = fc[fc["forecast_year"] == DENOM_YEAR].copy()
    ref["lf_assume_mwh"] = (
        ref["net_demand_forecast_mwh"]
        / (ref["summer_peak_demand_forecast_mw"] * HOURS_PER_YEAR)
    )
    flag = ref[ref["lf_assume_mwh"] < LF_THRESHOLD][
        "respondent_id_ferc714"
    ].tolist()
    print(f"[units] respondents filing in GWh (auto-detected): {sorted(flag)}")

    mask = fc["respondent_id_ferc714"].isin(flag)
    fc.loc[mask, "energy_mwh_normalized"] = (
        fc.loc[mask, "net_demand_forecast_mwh"] * 1000.0
    )
    return fc


# =====================================================================
# RATE COMPUTATION
# =====================================================================

def compute_per_respondent_rates(fc: pd.DataFrame) -> pd.DataFrame:
    """For each respondent, compute growth_factor[period] = forecast[period]
    / forecast[2025]. Returns long-form rows.

    Missing forecast_year for a respondent → row not emitted (that
    (respondent, period) gets a placeholder in the output).
    """
    p = fc.pivot_table(
        index="respondent_id_ferc714", columns="forecast_year",
        values="energy_mwh_normalized", aggfunc="first",
    )
    rows = []
    for rid, r in p.iterrows():
        denom = r.get(DENOM_YEAR)
        if pd.isna(denom) or denom == 0:
            continue
        for period in PERIODS:
            num = r.get(period)
            if pd.isna(num):
                continue
            rows.append({
                "respondent_id_ferc714": int(rid),
                "period": period,
                "growth_factor": round(float(num / denom), 6),
            })
    return pd.DataFrame(rows)


def compute_wecc_wide_rate(crosswalk: pd.DataFrame,
                           fc: pd.DataFrame) -> pd.Series:
    """A single WECC-wide growth rate per period.

    Subtle but important: for each target period, only include respondents
    that filed BOTH a 2025 forecast AND a forecast for the target year.
    Otherwise the numerator and denominator come from different sets and the
    rate is meaningless (e.g., for 2040 only CAISO files; summing CAISO over
    27-respondent total gives a fake "0.02" rate).

    If fewer than half the WECC respondents have coverage for a target period,
    we return None (caller falls back to placeholder).
    """
    wecc_rids = crosswalk["respondent_id_ferc714"].unique()
    wfc = fc[fc["respondent_id_ferc714"].isin(wecc_rids)]

    p = wfc.pivot_table(
        index="respondent_id_ferc714", columns="forecast_year",
        values="energy_mwh_normalized", aggfunc="first",
    )

    out = {}
    for period in PERIODS:
        if period == DENOM_YEAR:
            out[period] = 1.0
            continue
        if DENOM_YEAR not in p.columns or period not in p.columns:
            out[period] = None
            continue
        both = p[[DENOM_YEAR, period]].dropna()
        # Require coverage of >= half the WECC respondents for the rate to
        # be meaningful.
        if len(both) < len(wecc_rids) / 2:
            print(f"[wecc-rate] period {period}: only {len(both)} respondents "
                  f"have both 2025 and {period} forecasts (of {len(wecc_rids)}) "
                  f"— skipping, will fall back to placeholder")
            out[period] = None
            continue
        denom_sum = both[DENOM_YEAR].sum()
        num_sum = both[period].sum()
        out[period] = round(float(num_sum / denom_sum), 6) if denom_sum > 0 else None
    return pd.Series(out, name="wecc_wide")


# =====================================================================
# OUTPUT BUILDER
# =====================================================================

def build_growth_factors(crosswalk: pd.DataFrame,
                         per_resp: pd.DataFrame,
                         wecc_wide: pd.Series) -> pd.DataFrame:
    """Build (zone × period) growth factors with sources documented per row."""
    rows = []
    for _, xw in crosswalk.iterrows():
        zone = xw["zone"]
        rid = xw["respondent_id_ferc714"]
        for period in PERIODS:
            # ---- Decide source ----
            if zone in FLAT_RATE_ZONES:
                gf = wecc_wide.get(period)
                source = "wecc_wide_flat"
                notes = (
                    "TEPC: own FERC filing under-reports BA load (2 TWh "
                    "vs 14.5 TWh actual). Using WECC-wide rate."
                )
            else:
                match = per_resp[
                    (per_resp["respondent_id_ferc714"] == rid)
                    & (per_resp["period"] == period)
                ]
                if not match.empty:
                    gf = float(match["growth_factor"].iloc[0])
                    source = "ferc714_respondent"
                    notes = ""
                else:
                    gf = 1.0
                    source = "placeholder"
                    notes = "no FERC-714 forecast for this period; PI input needed"

            if gf is None or pd.isna(gf):
                gf = 1.0
                source = "placeholder"
                notes = (
                    (notes + "; " if notes else "")
                    + "wecc_wide also unavailable"
                )

            rows.append({
                "zone": zone,
                "period": period,
                "growth_factor": round(float(gf), 6),
                "source": source,
                "notes": notes,
            })
    return pd.DataFrame(rows)


# =====================================================================
# VALIDATION
# =====================================================================

def validate(out: pd.DataFrame) -> None:
    print("\n[validate]")
    # 31 × 5 = 155 rows
    assert len(out) == 31 * 5, f"expected 155 rows, got {len(out)}"
    print(f"  rows: {len(out)} (31 zones × 5 periods) ✓")

    # 2025 should be 1.0 for everyone (since denom == numer for FERC source)
    p25 = out[out["period"] == 2025]
    bad = p25[(p25["growth_factor"] - 1.0).abs() > 1e-5]
    assert bad.empty, f"non-1.0 growth in 2025:\n{bad}"
    print(f"  2025 growth = 1.0 for all zones ✓")

    # No NaN anywhere
    nans = out["growth_factor"].isna().sum()
    assert nans == 0, f"{nans} NaN growth factors — fix before saving"
    print(f"  no NaN growth factors ✓")

    # Source breakdown
    print(f"\n[source breakdown]")
    print(out.groupby(["period", "source"]).size().unstack(fill_value=0).to_string())


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    crosswalk = load_crosswalk()
    fc = load_forecasts()
    fc = normalize_units(fc)

    per_resp = compute_per_respondent_rates(fc)
    wecc_wide = compute_wecc_wide_rate(crosswalk, fc)

    print(f"\n[wecc-wide rate per period] {wecc_wide.to_dict()}")

    out = build_growth_factors(crosswalk, per_resp, wecc_wide)
    validate(out)

    print(f"\n[growth factors per zone, period 2030]")
    p30 = out[out["period"] == 2030][["zone", "growth_factor", "source"]]
    print(p30.sort_values("growth_factor", ascending=False).to_string(index=False))

    out_path = DEMAND_DIR / "growth_factors.csv"
    out.to_csv(out_path, index=False)
    print(f"\n[save] {out_path.relative_to(WECC_ROOT)}  rows={len(out):,}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
