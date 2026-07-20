"""Assemble the bus-level annual demand layer for PyPSA-USA.

Components, per model year (default 2030):
1. efs_base    — EFS Reference Res+Com+Trans state totals -> BA via the
                 930-anchored allocation (ba_allocation_v2.csv) -> substations
                 by BE Pd weights within (BA, state). EPE's El Paso (TX)
                 load is an exogenous add-on grown at the NM rate.
2. industrial  — the carve-out layer built previously:
                 substation_industrial_remainder.csv (EFS remainder, grows)
                 + phase1_facility_to_substation.csv (fixed baseline).
3. data_centers— EPRI 2026 state projections (Medium central; Low/High in
                 sensitivity columns) -> BA via v2 total-load shares
                 (PLACEHOLDER — data centers are lumpy; site-list allocation
                 is the flagged upgrade) -> substations by Pd. EPRI ends at
                 2030; later years hold the 2030 value (flagged).

Hourly shapes (applied downstream, documented in README):
  efs_base & industrial -> state EFS hourly shape by sector;
  data_centers -> flat at EPRI 88-94% load-factor guidance.

Scenarios (--scenario, default "reference"):
  reference: EFS Reference + EPRI Medium -> demand/pypsa_demand/
  high:      EFS High + EPRI High       -> demand/pypsa_demand_high/
             (industrial remainder scaled by state EFS High/Reference
             industrial ratio per year; phase1 facility baseline fixed)

Output: demand/pypsa_demand[_high]/bus_demand_<year>.csv
        (sub_id, lat, lon, ba, state, component, twh)
"""

import argparse
from pathlib import Path

import pandas as pd

WECC_ROOT = Path(__file__).resolve().parents[1]
RAW = WECC_ROOT / "local_data" / "input_data" / "demand" / "raw"
DEM = WECC_ROOT / "demand"
P1 = DEM / "phase1_baseline"

SCENARIOS = {
    "reference": {"efs": "efs_wecc_reference_moderate.parquet",
                  "epri": "Medium", "outdir": "pypsa_demand"},
    "high": {"efs": "efs_wecc_high_moderate.parquet",
             "epri": "High", "outdir": "pypsa_demand_high"},
}

# shapefile pools for BA codes without their own polygon (see substation
# script): AZ BAs share the merged "Arizona" polygon, WA munis sit in PSEI
BA_POOL = {"AZPS": "Arizona", "SRP": "Arizona", "TEPC": "Arizona",
           "SCL": "PSEI", "TPWR": "PSEI"}
NONIND = ["Residential", "Commercial", "Transportation"]


def spread(cells, subs, value_col):
    """Spread (ba, state, value) rows across substations by Pd weights."""
    frames = []
    for _, r in cells.iterrows():
        pool_ba = BA_POOL.get(r.ba, r.ba)
        pool = subs[(subs.ba == pool_ba) & (subs.Pd > 0)]
        if r.state in set(subs.state) and r.state != "TX_exogenous":
            p2 = pool[pool.state == r.state]
            pool = p2 if len(p2) else subs[(subs.state == r.state)
                                           & (subs.Pd > 0)]
        if pool.empty:
            pool = subs[subs.Pd > 0]
        w = pool.Pd / pool.Pd.sum()
        frames.append(pd.DataFrame({"sub_id": pool.sub_id,
                                    "twh": r[value_col] * w.values}))
    return (pd.concat(frames).groupby("sub_id").twh.sum())


def main(year=2030, scenario="reference"):
    scen = SCENARIOS[scenario]
    out = DEM / scen["outdir"]
    out.mkdir(exist_ok=True)
    subs = pd.read_csv(P1 / "wecc_substation_weights.csv")
    alloc = pd.read_csv(DEM / "ba_allocation_v2.csv")

    # ---- 1. efs_base (non-industrial) ----
    efs = pd.read_parquet(
        RAW / scen["efs"],
        columns=["Year", "State", "Sector", "LoadMW"],
        filters=[("Year", "in", [2024, year])])
    st = efs[efs.Sector.isin(NONIND)].groupby(
        ["Year", "State"]).LoadMW.sum().unstack(0) / 1e6
    # Eastern Interconnection carve-out (same treatment as industrial):
    # EIA-861 all-class sales share of SWPP/ERCO inside each WECC state
    s861 = pd.read_parquet(RAW / "pudl_eia861_yearly_sales.parquet",
                           columns=["report_date", "state", "sales_mwh",
                                    "balancing_authority_code_eia",
                                    "customer_class", "service_type"])
    s861 = s861[(pd.to_datetime(s861.report_date).dt.year == 2024)
                & (s861.customer_class != "total")
                & (s861.service_type.isin(["bundled", "energy"]))]
    s861["east"] = s861.balancing_authority_code_eia.isin(
        {"SWPP", "ERCO", "MISO", "SPA", "AECI"})
    east = (s861[s861.state.isin(st.index)].groupby("state")
            .apply(lambda d: d[d.east].sales_mwh.sum() / d.sales_mwh.sum(),
                   include_groups=False))
    budget = st[year] * (1 - east.reindex(st.index).fillna(0))
    print(f"[carve] Eastern share removed from non-industrial: "
          f"{(st[year] - budget).sum():.1f} TWh "
          f"({', '.join(f'{s} {v*100:.0f}%' for s, v in east[east > 0.01].items())})")
    cells = alloc[alloc.flag != "exogenous_tx"].copy()
    cells["val"] = cells.share_of_state * cells.state.map(budget)
    # EPE Texas add-on grown at the NM total-load rate
    nm_growth = (efs[efs.State == "NM"].groupby("Year").LoadMW.sum()[year]
                 / efs[efs.State == "NM"].groupby("Year").LoadMW.sum()[2024])
    tx = alloc[alloc.flag == "exogenous_tx"].copy()
    tx["val"] = tx.ba_state_twh_930anchored * nm_growth
    base = spread(pd.concat([cells, tx]), subs, "val").rename("efs_base")
    print(f"[efs_base] {year}: {base.sum():.1f} TWh "
          f"(incl. EPE-TX {tx.val.sum():.1f}, NM growth {nm_growth:.3f})")

    # ---- 2. industrial (existing layer) ----
    rem = pd.read_csv(P1 / "substation_industrial_remainder.csv")
    rem = rem[rem.year == year].copy()
    if scenario != "reference":
        # remainder was precomputed from EFS Reference; scale each state's
        # remainder by this scenario's EFS industrial ratio for the year
        ref = pd.read_parquet(
            RAW / SCENARIOS["reference"]["efs"],
            columns=["Year", "State", "Sector", "LoadMW"],
            filters=[("Year", "==", year), ("Sector", "==", "Industrial")])
        scn = pd.read_parquet(
            RAW / scen["efs"],
            columns=["Year", "State", "Sector", "LoadMW"],
            filters=[("Year", "==", year), ("Sector", "==", "Industrial")])
        ratio = (scn.groupby("State").LoadMW.sum()
                 / ref.groupby("State").LoadMW.sum())
        sub_state = subs.set_index("sub_id").state
        rem["remainder_twh"] *= (rem.sub_id.map(sub_state).map(ratio)
                                 .fillna(1.0).values)
        print(f"[industrial] {scenario} scaling: state ratios "
              f"{ratio.min():.3f}-{ratio.max():.3f}")
    ind = (rem.groupby("sub_id").remainder_twh.sum()
           .rename("industrial_remainder"))
    fac = pd.read_csv(P1 / "phase1_facility_to_substation.csv")
    p1 = (fac[fac.in_wecc].groupby("sub_id").baseline_gwh.sum() / 1e3
          ).rename("industrial_phase1")
    print(f"[industrial] remainder {ind.sum():.1f} + phase1 {p1.sum():.1f} TWh")

    # ---- 3. data centers (EPRI 2026) ----
    epri = pd.read_csv(RAW / "EPRI Powering Intelligence - All States and "
                             "Total (2026-05-18).csv")
    dc_year = min(year, 2030)  # EPRI horizon ends 2030 — hold after
    # INCREMENT above 2024: existing 2024 DC load is already embedded in the
    # EFS/EIA-anchored baseline (2024 validation vs EIA-930 ≈ 1.0 includes
    # actual DC load). Adding EPRI totals outright would double count.
    med = epri[epri.Scenario == scen["epri"]]
    dc_state = (med[med.Year == dc_year].set_index("State")["Annual Energy (TWh)"]
                - med[med.Year == 2024].set_index("State")["Annual Energy (TWh)"]
                ).clip(lower=0)
    # same Eastern carve-out as efs_base (EPRI state totals cover whole
    # states too; the SWPP-side share of NM/MT projects leaves the model)
    dc_state = dc_state * (1 - east.reindex(dc_state.index).fillna(0))
    dcc = cells.copy()
    dcc["val"] = dcc.share_of_state * dcc.state.map(dc_state).fillna(0)
    dcc = dcc[dcc.val > 0]
    dcb = (spread(dcc, subs, "val") if len(dcc)
           else pd.Series(dtype=float)).rename("data_centers")
    held = " (held at 2030)" if year > 2030 else ""
    print(f"[data_centers] EPRI {scen['epri']} {dc_year}{held}: {dcb.sum():.1f} TWh")

    # ---- combine ----
    df = pd.concat([base, ind, p1, dcb], axis=1).fillna(0)
    df.index.name = "sub_id"
    df = df.join(subs.set_index("sub_id")[["name", "lat", "lon", "ba",
                                           "state"]])
    long = (df.reset_index().melt(
        id_vars=["sub_id", "name", "lat", "lon", "ba", "state"],
        value_vars=["efs_base", "industrial_remainder", "industrial_phase1",
                    "data_centers"], var_name="component", value_name="twh"))
    long = long[long.twh > 0]
    long.round(6).to_csv(out / f"bus_demand_{year}.csv", index=False)
    tot = long.twh.sum()
    print(f"\n[total] {year}: {tot:.1f} TWh at "
          f"{long.sub_id.nunique()} substations -> "
          f"{out / f'bus_demand_{year}.csv'}")
    print(long.groupby("component").twh.sum().round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("year", nargs="?", type=int, default=2030)
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="reference")
    a = ap.parse_args()
    main(a.year, a.scenario)
