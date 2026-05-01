"""Build the FERC-714 respondent → 31-zone crosswalk.

This is the unblocking step for FERC-714 demand-forecast scaling. Once the
crosswalk exists, we can compute growth multipliers per (zone, period) by
dividing each zone's forecast for the period year by its 2024 actual energy.

Inputs:
    local_data/input_data/demand/raw/ferc714_respondents_with_fips.parquet
    PUDL S3:  core_ferc714__yearly_planning_area_demand_forecast.parquet
    local_data/input_data/demand/raw/eia930_hourly_wecc_2024.parquet
    PUDL S3:  core_eia930__hourly_subregion_demand.parquet  (CAISO subregions)

Output:
    local_data/input_data/demand/ferc714_to_zone_crosswalk.csv
    Columns: zone, respondent_id_ferc714, respondent_name, share,
             mapping_method, notes

CROSSWALK STRATEGY (decided 2026-04-30):

The PUDL `respondents_with_fips` table already includes balancing_authority_code_eia
on each respondent row, which gets us 90% of the way there.  Three classes of
mapping:

    1. DIRECT (24 zones): respondent ↔ zone is 1:1.  share = 1.0.

    2. BA_SUBSET (PACE, PACW): PacifiCorp files a single FERC-714 respondent
       (id 194) covering both BAs combined.  We split it into PACE and PACW
       weighted by their 2024 EIA-930 actual energy share.  Both rows get
       respondent 194; share columns sum to 1.0 across them.

    3. SUBREGION_OF_AGGREGATE (PGAE, SCE, SDGE, VEA):  CAISO files a single
       respondent (id 24) covering all four subregions.  We split it weighted
       by their 2024 actual share of CAISO total energy.  All four rows point
       at respondent 24; share columns sum to 1.0.

Plus 2 manual corrections to PUDL metadata (the `balancing_authority_code_eia`
column on respondents 162 and 217 is flipped in PUDL v2026.1.0 — see below).

WALC / WAUW BA-CODE FLIP IN PUDL — what we found and why we override:

    Both EIA-930 codes refer to Western Area Power Administration regions:

      WALC = Western Area Power Administration - Desert Southwest Region
             (Lower Colorado).  Service territory: AZ, southern CA, NV, UT,
             NM.  Around Hoover Dam.  ~8.5 TWh annual demand in 2024.

      WAUW = Western Area Power Administration - Upper Great Plains West
             (Western Interconnection slice).  Service territory: WY, MT.
             ~0.8 TWh annual demand in 2024.

    PUDL's `respondents_with_fips` table for respondents 162 and 217:

      respondent 162
          respondent_name_ferc714       = "Western Area Power Administration-
                                          Lower Colorado (Desert Southwest
                                          Region)"
          balancing_authority_code_eia  = "WAUW"   ← WRONG
          balancing_authority_name_eia  = "USDOE-WAPA-Lower Colorado"
          (FIPS counties listed as MT, WY — also wrong)

      respondent 217
          respondent_name_ferc714       = "Western Area Power Administration -
                                          Upper Great Plains Region"
          balancing_authority_code_eia  = "WALC"   ← WRONG
          balancing_authority_name_eia  = "USDOE-WAPA-Upper Missouri-West"
          (FIPS counties listed as AZ, CA, NM, NV, UT — also wrong)

    Three independent signals confirm the BA code field is flipped:

      (1) The respondent NAME for 162 ("Lower Colorado / Desert Southwest")
          describes WALC's territory, not WAUW's.  The respondent NAME for
          217 ("Upper Great Plains") describes WAUW's territory, not WALC's.

      (2) The `balancing_authority_name_eia` field (a separate column in the
          same PUDL row) agrees with the respondent name: 162's BA name is
          "Lower Colorado" → WALC; 217's BA name is "Upper Missouri-West" →
          WAUW.  So PUDL's own metadata is internally inconsistent — the BA
          *code* field disagrees with the BA *name* field.

      (3) Forecast magnitudes are the decisive check.  Respondent 162 files
          a 2025 forecast of 7,971,000 MWh (~8 TWh, summer peak 1,634 MW),
          which matches WALC's 2024 EIA-930 actual of 8.5 TWh.  Respondent
          217 files 950,000 MWh (~0.95 TWh, summer peak 190 MW), matching
          WAUW's 2024 actual of 0.8 TWh.  The two would be ~10× misscaled
          if we trusted PUDL's BA code as-is.

    Final override → respondent 162: WALC.  Respondent 217: WAUW.

    Respondent 163 ("WAPA - Upper Missouri West") last reported 2020 and is
    superseded by 217.  Dropped from crosswalk entirely.

NOTE on growth-multiplier semantics:
    For the BA_SUBSET and SUBREGION_OF_AGGREGATE classes, every split zone
    gets the *same* growth multiplier as the parent respondent.  share is
    informational — it tells you the absolute-MWh allocation if you ever
    need it, but the multiplier math doesn't actually need it.

OPEN QUESTIONS for PI:
    * Is it OK to apply CAISO's system-wide growth rate to all 4 subregions,
      or do we need subregion-specific forecasts from another source?
    * Same question for PACE/PACW.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# =====================================================================
# PATHS / CONSTANTS
# =====================================================================

REPO_ROOT = Path(__file__).resolve().parents[2]
WECC_ROOT = REPO_ROOT.parent
RAW_DIR = WECC_ROOT / "local_data" / "input_data" / "demand" / "raw"
OUT_DIR = WECC_ROOT / "local_data" / "input_data" / "demand"

PUDL_S3_BASE = "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop"
PUDL_RELEASE = "v2026.1.0"

BA_LOAD_ZONES = [
    "AVA", "AZPS", "BANC", "BPAT", "CHPD", "DOPD", "EPE", "GCPD", "IID",
    "IPCO", "LDWP", "NEVP", "NWMT", "PACE", "PACW", "PGE", "PNM", "PSCO",
    "PSEI", "SCL", "SRP", "TEPC", "TIDC", "TPWR", "WACM", "WALC", "WAUW",
]
CAISO_SUBREGIONS = ["PGAE", "SCE", "SDGE", "VEA"]
ALL_ZONES = sorted(BA_LOAD_ZONES + CAISO_SUBREGIONS)

# Manual corrections to PUDL respondents_with_fips metadata.
# Key: respondent_id_ferc714 → corrected BA code.
BA_CODE_OVERRIDES = {
    217: "WAUW",   # PUDL says WALC, but Upper Great Plains is WAUW
    162: "WALC",   # PUDL says WAUW, but Lower Colorado is WALC
}

# Respondents to drop (stale, superseded, etc.)
DROP_RESPONDENTS = {
    163,  # WAPA Upper Missouri West — last reported 2020, superseded by 217
}

WEATHER_YEAR = 2024


# =====================================================================
# LOADERS
# =====================================================================

def load_respondents() -> pd.DataFrame:
    """Latest report-year row per respondent, with PUDL BA code overrides
    applied. Filters to balancing-authority respondent type and drops
    stale respondents."""
    src = RAW_DIR / "ferc714_respondents_with_fips.parquet"
    print(f"[load] {src.name}")
    r = pd.read_parquet(src)
    r = r.sort_values("report_date").drop_duplicates(
        "respondent_id_ferc714", keep="last"
    )
    r = r[r["respondent_type"] == "balancing_authority"]
    r = r[~r["respondent_id_ferc714"].isin(DROP_RESPONDENTS)]
    # Apply manual BA code overrides
    for rid, ba in BA_CODE_OVERRIDES.items():
        mask = r["respondent_id_ferc714"] == rid
        if mask.any():
            old = r.loc[mask, "balancing_authority_code_eia"].iloc[0]
            print(f"[fix]  respondent {rid}: BA code {old} → {ba}")
            r.loc[mask, "balancing_authority_code_eia"] = ba
    return r[
        ["respondent_id_ferc714", "respondent_name_ferc714",
         "balancing_authority_code_eia"]
    ].rename(columns={
        "respondent_name_ferc714": "respondent_name",
        "balancing_authority_code_eia": "ba_code",
    })


def load_2024_zone_energy() -> pd.Series:
    """2024 actual energy (TWh) per zone, used as the denominator for share
    weights and (later) growth multipliers."""
    print(f"[load] 2024 zonal energy from EIA-930 + CAISO subregions")

    # BAs from local parquet
    ba = pd.read_parquet(RAW_DIR / f"eia930_hourly_wecc_{WEATHER_YEAR}.parquet")
    ba["datetime_utc"] = pd.to_datetime(ba["datetime_utc"], utc=True)
    ba = ba[
        ba["balancing_authority_code_eia"].isin(BA_LOAD_ZONES)
        & (ba["datetime_utc"].dt.year == WEATHER_YEAR)
    ].copy()
    ba["load_mw"] = (
        ba["demand_imputed_pudl_mwh"]
        .fillna(ba["demand_adjusted_mwh"])
        .fillna(ba["demand_reported_mwh"])
    )
    ba_energy = ba.groupby("balancing_authority_code_eia")["load_mw"].sum()

    # CAISO subregions from PUDL S3
    url = (
        f"{PUDL_S3_BASE}/{PUDL_RELEASE}/"
        "core_eia930__hourly_subregion_demand.parquet"
    )
    sub = pd.read_parquet(url, columns=[
        "datetime_utc", "balancing_authority_code_eia",
        "balancing_authority_subregion_code_eia", "demand_reported_mwh",
    ])
    sub["datetime_utc"] = pd.to_datetime(sub["datetime_utc"], utc=True)
    sub = sub[
        (sub["balancing_authority_code_eia"] == "CISO")
        & (sub["datetime_utc"].dt.year == WEATHER_YEAR)
        & (sub["balancing_authority_subregion_code_eia"].isin(CAISO_SUBREGIONS))
    ].copy()
    sub["load_mw"] = sub["demand_reported_mwh"].ffill().bfill()
    sub_energy = sub.groupby(
        "balancing_authority_subregion_code_eia"
    )["load_mw"].sum()

    out = pd.concat([ba_energy, sub_energy])
    out.name = "energy_mwh_2024"
    out.index.name = "zone"
    return out


# =====================================================================
# CROSSWALK BUILDER
# =====================================================================

def build_crosswalk(respondents: pd.DataFrame,
                    zone_energy: pd.Series) -> pd.DataFrame:
    """Construct the (zone → respondent) crosswalk row-by-row.

    Row schema:
        zone, respondent_id_ferc714, respondent_name, share,
        mapping_method, notes
    """
    rows = []

    # ---- 1. DIRECT mappings (1:1 BA ↔ respondent) ----
    # Look up each BA in respondents; one row per BA except the splits.
    SPLIT_BAS = {"PACE", "PACW"}      # PacifiCorp covers both
    direct_bas = [b for b in BA_LOAD_ZONES if b not in SPLIT_BAS]

    for ba in direct_bas:
        match = respondents[respondents["ba_code"] == ba]
        if match.empty:
            print(f"[warn] no respondent for {ba} — skipping")
            continue
        if len(match) > 1:
            print(f"[warn] multiple respondents for {ba}: "
                  f"{match['respondent_id_ferc714'].tolist()}")
        row = match.iloc[0]
        rows.append({
            "zone": ba,
            "respondent_id_ferc714": int(row["respondent_id_ferc714"]),
            "respondent_name": row["respondent_name"],
            "share": 1.0,
            "mapping_method": "direct",
            "notes": "",
        })

    # ---- 2. BA_SUBSET split: PACE + PACW share PacifiCorp respondent 194 ----
    pacificorp = respondents[respondents["respondent_id_ferc714"] == 194].iloc[0]
    pace_e = zone_energy["PACE"]
    pacw_e = zone_energy["PACW"]
    total_e = pace_e + pacw_e
    pace_share = pace_e / total_e
    pacw_share = pacw_e / total_e
    print(f"[split] PACE/PACW share of PacifiCorp: "
          f"PACE={pace_share:.3f}  PACW={pacw_share:.3f}")
    for zone, share in [("PACE", pace_share), ("PACW", pacw_share)]:
        rows.append({
            "zone": zone,
            "respondent_id_ferc714": 194,
            "respondent_name": pacificorp["respondent_name"],
            "share": round(float(share), 6),
            "mapping_method": "ba_subset",
            "notes": "PacifiCorp respondent covers PACE+PACW combined; "
                    f"share by 2024 EIA-930 energy",
        })

    # ---- 3. SUBREGION_OF_AGGREGATE: 4 CAISO subregions share respondent 24 ----
    caiso = respondents[respondents["respondent_id_ferc714"] == 24].iloc[0]
    caiso_total = sum(zone_energy[z] for z in CAISO_SUBREGIONS)
    print(f"[split] CAISO subregion shares (2024):")
    for z in CAISO_SUBREGIONS:
        share = zone_energy[z] / caiso_total
        print(f"          {z}: {share:.3f}")
        rows.append({
            "zone": z,
            "respondent_id_ferc714": 24,
            "respondent_name": caiso["respondent_name"],
            "share": round(float(share), 6),
            "mapping_method": "subregion_of_aggregate",
            "notes": "CAISO respondent covers all 4 subregions; "
                    "share by 2024 EIA-930 subregion energy",
        })

    return pd.DataFrame(rows).sort_values("zone").reset_index(drop=True)


# =====================================================================
# VALIDATION
# =====================================================================

def validate(crosswalk: pd.DataFrame) -> None:
    """Sanity checks on the crosswalk."""
    print("\n[validate]")

    # Every zone covered exactly once
    zones_in_xw = set(crosswalk["zone"])
    expected = set(ALL_ZONES)
    missing = expected - zones_in_xw
    extra = zones_in_xw - expected
    assert not missing, f"missing zones: {missing}"
    assert not extra, f"extra zones in crosswalk: {extra}"
    print(f"  zone coverage: {len(zones_in_xw)}/{len(expected)} ✓")

    # Shares per respondent sum to 1.0 (within float epsilon)
    by_resp = crosswalk.groupby("respondent_id_ferc714")["share"].sum()
    bad = by_resp[(by_resp - 1.0).abs() > 1e-5]
    assert bad.empty, f"respondents whose shares != 1.0:\n{bad}"
    print(f"  share sums per respondent: all 1.0 ✓")

    # No respondent should appear with mixed mapping_methods (sanity)
    mixed = (
        crosswalk.groupby("respondent_id_ferc714")["mapping_method"]
        .nunique()
    )
    bad = mixed[mixed > 1]
    assert bad.empty, f"respondents with mixed methods:\n{bad}"
    print(f"  no respondent with mixed mapping methods ✓")


# =====================================================================
# MAIN
# =====================================================================

def main() -> int:
    respondents = load_respondents()
    zone_energy = load_2024_zone_energy()

    print(f"\n[zone energy 2024 (TWh)]")
    print((zone_energy / 1e6).round(1).sort_values(ascending=False).to_string())

    crosswalk = build_crosswalk(respondents, zone_energy)

    print(f"\n[crosswalk] {len(crosswalk)} rows")
    print(crosswalk.to_string(index=False))

    validate(crosswalk)

    out = OUT_DIR / "ferc714_to_zone_crosswalk.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(out, index=False)
    print(f"\n[save] {out.relative_to(WECC_ROOT)}  rows={len(crosswalk):,}")

    print("\n[summary]")
    print(f"  zones: {crosswalk['zone'].nunique()}")
    print(f"  respondents: {crosswalk['respondent_id_ferc714'].nunique()}")
    print(f"  mapping methods: "
          f"{crosswalk['mapping_method'].value_counts().to_dict()}")
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
