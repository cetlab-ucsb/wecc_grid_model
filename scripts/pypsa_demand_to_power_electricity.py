"""Convert our bus_hourly demand outputs to PyPSA-USA's native demand-input
format.

PyPSA-USA's `build_electrical_demand` rule writes
`resources/{interconnect}/demand/power_electricity.csv`, which `add_demand.py`
reads with `pd.read_csv(f, index_col=0)` and attaches via `n.madd("Load", ...)`.

Native format (verified against build_demand.py / add_demand.py, July 2026):
  - index: single-level datetime strings, one 8760-h block per planning
    horizon, concatenated in horizon order (e.g. 2024 -> 2030 -> 2040 -> 2050).
    No Feb 29 ever: snapshots are a non-leap template with the year swapped in
    (get_multiindex_snapshots -> x.replace(year=year)).
  - columns: PyPSA bus names (Breakthrough Energy bus_id, NOT sub_id).
  - values: MW, rounded to 4 decimals.
  - row count must equal len(n.snapshots) exactly (asserted in add_demand.py).

Updated 2026-08-19: works on ANY bus_hourly parquet (8760 rows x sub_id
columns + hour_utc — the layout of both the scenario pipeline and
downscale_zone_to_bus.py), and maps sub_id -> bus_id properly: each
substation's profile is split across its buses by the buses' Pd weights from
the network's bus_gis.csv (equal split where a sub's buses all have Pd = 0).
The old first-bus-per-sub --bus2sub shortcut is retired.

Usage:
  python pypsa_demand_to_power_electricity.py FILE [FILE ...]
      [--bus-gis PATH]   default: pypsa-usa-wecc/workflow/resources/Default/
                                  western/bus_gis.csv
      [--out PATH]       default: power_electricity.csv next to first input
      [--sample]         first 48 h per horizon (quick inspection)
      [--parquet]        write parquet instead of csv (inspection only —
                         add_demand.py reads csv)
Each FILE supplies one planning horizon; the horizon year is parsed from the
first 20xx in the filename. Pass files in horizon order.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUS_GIS = (ROOT / "pypsa-usa-wecc" / "workflow" / "resources" /
                   "Default" / "western" / "bus_gis.csv")


def snapshot_template() -> pd.DatetimeIndex:
    """8760 hourly stamps from a non-leap year (matches PyPSA-USA's template)."""
    return pd.date_range("2019-01-01", "2020-01-01", freq="h", inclusive="left")


def bus_weights(bus_gis: Path) -> pd.DataFrame:
    """bus, sub_id, wt — each sub's load split across its buses by Pd."""
    g = pd.read_csv(bus_gis, usecols=["Bus", "sub_id", "Pd"])
    g = g.dropna(subset=["sub_id"])
    g["sub_id"] = g.sub_id.astype(int).astype(str)
    tot = g.groupby("sub_id").Pd.transform("sum")
    n = g.groupby("sub_id").Bus.transform("size")
    g["wt"] = (g.Pd / tot).where(tot > 0, 1.0 / n)   # equal split if Pd all 0
    return g[["Bus", "sub_id", "wt"]]


def load_horizon(path: Path, year: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "hour_utc" in df.columns:
        df = df.set_index("hour_utc").sort_index()
    assert len(df) == 8760, f"{path.name}: expected 8760 rows, got {len(df)}"
    # hour_utc h -> template stamp h, then swap in the horizon year.
    # Identical to PyPSA-USA: no Feb 29, so .replace(year=) is always safe.
    df.index = snapshot_template().map(lambda x: x.replace(year=year))
    df.index.name = "snapshot"
    return df


def to_buses(df: pd.DataFrame, w: pd.DataFrame) -> pd.DataFrame:
    missing = set(df.columns) - set(w.sub_id)
    if missing:
        lost = df[list(missing)].sum().sum() / 1e6
        print(f"WARNING: {len(missing)} sub_id(s) not in bus_gis — "
              f"{lost:.4f} TWh dropped: {sorted(missing)[:8]}...")
    w = w[w.sub_id.isin(df.columns)]
    # out[:, bus] = sum over that bus's sub rows of wt * df[:, sub]
    out = pd.DataFrame(0.0, index=df.index,
                       columns=pd.unique(w.Bus))
    for sub, grp in w.groupby("sub_id"):
        col = df[sub].to_numpy()
        for bus, wt in zip(grp.Bus, grp.wt):
            out[bus] += wt * col
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", type=Path,
                   help="bus_hourly parquet(s), one per horizon, in order")
    p.add_argument("--bus-gis", type=Path, default=DEFAULT_BUS_GIS)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--sample", action="store_true", help="first 48 h/horizon")
    p.add_argument("--parquet", action="store_true",
                   help="write parquet (inspection only; add_demand reads csv)")
    args = p.parse_args()

    w = bus_weights(args.bus_gis)
    blocks = []
    for f in args.files:
        m = re.search(r"(20[2-6]\d)", f.name)
        if not m:
            raise SystemExit(f"cannot parse horizon year from {f.name}")
        year = int(m.group(1))
        df = to_buses(load_horizon(f, year), w)
        print(f"[{year}] {f.name}: {df.shape[1]} buses, "
              f"{df.to_numpy().sum()/1e6:.2f} TWh, "
              f"peak {df.sum(axis=1).max()/1e3:.1f} GW")
        blocks.append(df.iloc[:48] if args.sample else df)
    out = pd.concat(blocks)

    tag = "_sample" if args.sample else ""
    dest = args.out or args.files[0].parent / f"power_electricity{tag}.csv"
    if args.parquet:
        dest = dest.with_suffix(".parquet")
        out.round(4).to_parquet(dest)
    else:
        out.round(4).to_csv(dest, index=True)
    print(f"wrote {dest}  shape={out.shape}  ({len(args.files)} horizon(s) x "
          f"{'48' if args.sample else '8760'} h)")


if __name__ == "__main__":
    main()
