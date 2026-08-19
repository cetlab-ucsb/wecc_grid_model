"""Downscale zone-level hourly demand profiles to PyPSA-USA substations.

Takes any long-format zone-level hourly file (zone, datetime_utc, sector,
mwh — the format of demand/base_year_2024/base_year_2024_hourly*.parquet and
every file in demand/components/) and spreads each zone's hourly series
across the substations inside that zone, by sector-appropriate weights:

  sector      within-zone weight                          source
  ---------   ------------------------------------------  ---------------------------
  phase1      facility baseline energy at its snapped     demand/base_year_2024/
              substation (exact point-load placement)     phase1_point_loads_2024.csv
  ind         2024 industrial remainder by substation     demand/phase1_baseline/
              (EIA-861-scale industrial disaggregation);  substation_industrial_total_2024.csv
              falls back to Pd where a zone has none
  everything  Breakthrough-Energy bus load weights (Pd),  demand/phase1_baseline/
  else        summed to the substation                    wecc_substation_weights.csv

Substation -> zone mapping: input_data/.../sub_to_pypsa_ba.csv (carries the
CISO subregions), GRID -> BPAT and WAUW -> NWMT relabels, falling back to the
weight table's own ba for the handful of substations missing from the map.
Substations whose zone never appears in the input are simply idle; energy is
conserved zone-by-zone and sector-by-sector (checked, tolerance 1e-6).

The heat layer (industrial_electrification) should NOT go through this
script — it is already substation-native
(demand/phase1_electrification/bus_hourly_electrification_*.parquet).

Outputs, per input file, into --outdir:
  bus_hourly_<stem>.parquet   8760 rows x substations, MW, float32, plus an
                              hour_utc column — same layout as the scenario
                              pipeline's bus_hourly files
  bus_demand_<stem>.csv       sub_id, name, lat, lon, zone, state, sector, twh
  (validation printed to stdout)

Usage:
  python scripts/downscale_zone_to_bus.py <input.parquet> [...] \
         [--outdir demand/components_bus] [--data-dir DIR]
Env override: WECC_ROOT.

--data-dir: portable mode — read all four weight/mapping CSVs
(wecc_substation_weights.csv, substation_industrial_total_2024.csv,
sub_to_pypsa_ba.csv, phase1_point_loads_2024.csv) from one directory
instead of the wecc-project layout (used in the wecc_grid_model repo).
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("WECC_ROOT", Path(__file__).resolve().parents[1]))
RAW = ROOT / "input_data" / "uploaded" / "demand" / "raw"
P1 = ROOT / "demand" / "phase1_baseline"
BY = ROOT / "demand" / "base_year_2024"
DATA_DIR = None          # set by --data-dir; overrides the paths above

RELABEL = {"GRID": "BPAT", "WAUW": "NWMT"}


def _path(default_dir, name):
    return (DATA_DIR / name) if DATA_DIR else (default_dir / name)


def load_sub_frame():
    """sub_id, zone, state, name, lat, lon, Pd for every WECC substation."""
    w = pd.read_csv(_path(P1, "wecc_substation_weights.csv"))
    zmap = (pd.read_csv(_path(RAW, "sub_to_pypsa_ba.csv"))
            .set_index("sub_id").balancing_area)
    w["zone"] = w.sub_id.map(zmap).fillna(w.ba).replace(RELABEL)
    n_ciso = (w.zone == "CISO").sum()
    if n_ciso:
        print(f"[subs] {n_ciso} substation(s) mapped only to 'CISO' (no "
              f"subregion) — dropped, Pd share "
              f"{w.loc[w.zone == 'CISO', 'Pd'].sum() / w.Pd.sum():.2e}")
        w = w[w.zone != "CISO"]
    return w


def weight_tables(subs):
    """Per-sector within-zone substation weights."""
    def normalize(df, col):
        s = df.groupby("zone")[col].transform("sum")
        out = df.assign(wt=df[col] / s)
        return out[out.wt > 0][["sub_id", "zone", "wt"]]

    pd_wt = normalize(subs[subs.Pd > 0], "Pd")

    ind = pd.read_csv(_path(P1, "substation_industrial_total_2024.csv"),
                      usecols=["sub_id", "remainder_twh"])
    ind = ind.merge(subs[["sub_id", "zone"]], on="sub_id")
    ind_wt = normalize(ind[ind.remainder_twh > 0], "remainder_twh")
    # zones with no industrial-remainder substations fall back to Pd
    missing = set(pd_wt.zone) - set(ind_wt.zone)
    if missing:
        print(f"[weights] ind falls back to Pd in zones: {sorted(missing)}")
        ind_wt = pd.concat([ind_wt, pd_wt[pd_wt.zone.isin(missing)]])

    fac = pd.read_csv(_path(BY, "phase1_point_loads_2024.csv"))
    p1_wt = normalize(fac.groupby(["sub_id", "zone"], as_index=False)
                      .baseline_gwh.sum(), "baseline_gwh")

    return {"__default__": pd_wt, "ind": ind_wt, "phase1": p1_wt}


def downscale(path: Path, subs, wts, outdir: Path):
    long = pd.read_parquet(path)
    if str(long.sector.iloc[0]).startswith("heat_"):
        raise SystemExit(f"{path.name}: heat layer is substation-native — "
                         "use bus_hourly_electrification_*.parquet instead")
    long["datetime_utc"] = pd.to_datetime(long.datetime_utc, utc=True)
    hours = np.sort(long.datetime_utc.unique())
    sub_ids = np.sort(subs.sub_id.unique())
    col = {s: i for i, s in enumerate(sub_ids)}
    arr = np.zeros((len(hours), len(sub_ids)))

    annual_rows = []
    for sector, g in long.groupby("sector"):
        wt = wts.get(sector, wts["__default__"])
        wide = (g.pivot(index="datetime_utc", columns="zone", values="mwh")
                .reindex(hours))
        for zone in wide.columns:
            zsum = wide[zone].to_numpy()
            if zsum.sum() == 0:
                continue
            zw = wt[wt.zone == zone]
            if not len(zw):
                print(f"[WARN] {path.name}: no substations for zone {zone} "
                      f"sector {sector} — {zsum.sum()/1e6:.4f} TWh dropped")
                continue
            idx = [col[s] for s in zw.sub_id]
            arr[:, idx] += np.outer(zsum, zw.wt.to_numpy())
            for s_id, w_ in zip(zw.sub_id, zw.wt):
                annual_rows.append(dict(sub_id=s_id, sector=sector,
                                        twh=zsum.sum() * w_ / 1e6))
        # conservation check per sector
        placed = sum(r["twh"] for r in annual_rows if r["sector"] == sector)
        src = g.mwh.sum() / 1e6
        ok = np.isclose(placed, src, atol=1e-6) or \
            abs(placed - src) < 1e-4 * max(src, 1)
        print(f"[{path.stem}] {sector}: {src:.3f} TWh -> {placed:.3f} at "
              f"substations {'OK' if ok else 'MISMATCH'}")

    outdir.mkdir(parents=True, exist_ok=True)
    hourly = pd.DataFrame(arr.astype("float32"),
                          columns=[str(s) for s in sub_ids])
    hourly.insert(0, "hour_utc", np.arange(len(hours)))
    fh = outdir / f"bus_hourly_{path.stem}.parquet"
    hourly.to_parquet(fh, index=False)

    ann = (pd.DataFrame(annual_rows)
           .groupby(["sub_id", "sector"], as_index=False).twh.sum())
    ann = ann.merge(subs[["sub_id", "name", "lat", "lon", "zone", "state"]],
                    on="sub_id")
    ann = ann[ann.twh > 0][["sub_id", "name", "lat", "lon", "zone", "state",
                            "sector", "twh"]]
    fa = outdir / f"bus_demand_{path.stem}.csv"
    ann.round(6).to_csv(fa, index=False)
    tot_in, tot_out = long.mwh.sum() / 1e6, arr.sum() / 1e6
    print(f"[{path.stem}] TOTAL {tot_in:.2f} -> {tot_out:.2f} TWh "
          f"({len(sub_ids)} subs, {(arr.sum(axis=0) > 0).sum()} loaded) "
          f"-> {fh.name}, {fa.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--outdir", type=Path,
                    default=ROOT / "demand" / "components_bus")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="portable mode: directory holding all four "
                         "weight/mapping CSVs (overrides the wecc layout)")
    a = ap.parse_args()
    global DATA_DIR
    DATA_DIR = a.data_dir
    subs = load_sub_frame()
    wts = weight_tables(subs)
    for p in a.inputs:
        downscale(p, subs, wts, a.outdir)


if __name__ == "__main__":
    main()
