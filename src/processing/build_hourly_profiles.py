"""Expand the bus-level annual demand layer to hourly (8760) profiles.

For each model year:
- efs_base:   state hourly shape = EFS Reference Res+Com+Trans hourly for
              that year, normalized to sum 1.
- industrial (remainder + phase1): state EFS Industrial hourly shape.
- data_centers: flat (constant every hour).
UTC alignment: EFS hours are state-local STANDARD time; profiles are rolled
to UTC (PST states +8 h: CA NV OR WA; MST +7 h: others; EPE-TX El Paso +7).

Outputs (demand/pypsa_demand/):
- hourly_shapes_<year>.parquet  (state, component, hour_utc, frac) — compact
- bus_hourly_<year>.parquet     (8760 rows x sub_id columns, MW, float32)

Conservation: sum(bus_hourly) MWh == sum(bus_demand annual) checked.

Usage: python3 build_hourly_profiles.py [year ...] [--scenario reference|high]
       (default: all 4 years, reference)
Scenario picks the matching EFS parquet for shapes and the in/out folder
(demand/pypsa_demand or demand/pypsa_demand_high) — see build_pypsa_demand.py.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

WECC_ROOT = Path(__file__).resolve().parents[1]
RAW = WECC_ROOT / "local_data" / "input_data" / "demand" / "raw"

SCENARIOS = {
    "reference": {"efs": "efs_wecc_reference_moderate.parquet",
                  "outdir": "pypsa_demand"},
    "high": {"efs": "efs_wecc_high_moderate.parquet",
             "outdir": "pypsa_demand_high"},
}

NONIND = ["Residential", "Commercial", "Transportation"]
UTC_OFFSET = {s: 8 for s in ["CA", "NV", "OR", "WA"]}  # PST
UTC_OFFSET.update({s: 7 for s in ["AZ", "CO", "ID", "MT", "NM", "UT", "WY",
                                  "TX"]})               # MST (+ El Paso)


def state_shapes(year: int, efs_file: str) -> pd.DataFrame:
    """Normalized UTC-aligned hourly shapes per (state, component)."""
    efs = pd.read_parquet(
        RAW / efs_file,
        columns=["Year", "LocalHourID", "State", "Sector", "LoadMW"],
        filters=[("Year", "==", year)])
    efs["grp"] = np.where(efs.Sector == "Industrial", "industrial",
                          "efs_base")
    hr = (efs.groupby(["State", "grp", "LocalHourID"]).LoadMW.sum()
          .reset_index())
    frames = []
    for (st, grp), d in hr.groupby(["State", "grp"]):
        prof = d.sort_values("LocalHourID").LoadMW.values.astype("float64")
        prof = prof / prof.sum()                       # normalize
        prof = np.roll(prof, UTC_OFFSET[st])           # local std -> UTC
        frames.append(pd.DataFrame({"state": st, "component": grp,
                                    "hour_utc": np.arange(8760),
                                    "frac": prof}))
    # flat DC shape + TX uses NM shapes (El Paso, same desert climate zone)
    flat = pd.DataFrame({"state": "ALL", "component": "data_centers",
                         "hour_utc": np.arange(8760), "frac": 1 / 8760})
    frames.append(flat)
    sh = pd.concat(frames, ignore_index=True)
    tx = sh[sh.state == "NM"].assign(state="TX")
    return pd.concat([sh, tx], ignore_index=True)


def build_year(year: int, scenario: str = "reference"):
    scen = SCENARIOS[scenario]
    OUT = WECC_ROOT / "demand" / scen["outdir"]
    bus = pd.read_csv(OUT / f"bus_demand_{year}.csv")
    bus["state"] = bus.state.replace({"TX_exogenous": "TX"}).fillna("NM")
    bus.loc[bus.component.isin(["industrial_remainder",
                                "industrial_phase1"]),
            "component_shape"] = "industrial"
    bus["component_shape"] = bus.component_shape.fillna(
        bus.component.map({"efs_base": "efs_base",
                           "data_centers": "data_centers"}))
    shapes = state_shapes(year, scen["efs"])
    shapes.to_parquet(OUT / f"hourly_shapes_{year}.parquet", index=False)

    sub_ids = np.sort(bus.sub_id.unique())
    col = {s: i for i, s in enumerate(sub_ids)}
    arr = np.zeros((8760, len(sub_ids)), dtype="float64")

    shp = {(r_state, comp): g.sort_values("hour_utc").frac.values
           for (r_state, comp), g in shapes.groupby(["state", "component"])}
    for (state, comp_shape), grp in bus.groupby(["state", "component_shape"]):
        key = ("ALL", "data_centers") if comp_shape == "data_centers" \
            else (state, comp_shape)
        prof = shp[key]                                  # 8760, sums to 1
        annual_mwh = grp.groupby("sub_id").twh.sum() * 1e6
        idx = [col[s] for s in annual_mwh.index]
        arr[:, idx] += np.outer(prof, annual_mwh.values)

    df = pd.DataFrame(arr.astype("float32"),
                      columns=[str(s) for s in sub_ids])
    df.insert(0, "hour_utc", np.arange(8760))
    df.to_parquet(OUT / f"bus_hourly_{year}.parquet", index=False)

    total_twh = arr.sum() / 1e6
    expect = bus.twh.sum()
    peak_gw = arr.sum(axis=1).max() / 1e3
    print(f"[{year}] {len(sub_ids)} buses; energy {total_twh:.1f} TWh "
          f"(expect {expect:.1f}, diff {abs(total_twh-expect):.4f}); "
          f"WECC coincident peak {peak_gw:.1f} GW")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("years", nargs="*", type=int, default=[2024, 2030, 2040, 2050])
    ap.add_argument("--scenario", choices=list(SCENARIOS), default="reference")
    a = ap.parse_args()
    for y in a.years or [2024, 2030, 2040, 2050]:
        build_year(y, a.scenario)
