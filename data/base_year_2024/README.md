# Base-year 2024 demand — PyPSA-USA regeneration package (2026-08-19)

Everything needed to produce the bus-level 2024 base-year demand for a
PyPSA-USA run. The final hourly file is ~208 MB and ~700 MB as CSV, so it is
**not** committed — regenerate it locally with the two commands below
(needs python3 with pandas + pyarrow; runs in ~2 minutes).

## What this is

Hourly 2024 demand anchored to observed EIA-930 (759.07 TWh across the WECC
zones, 8760 h — 2024 UTC calendar, Feb 29 dropped), with a constructed
sectoral split (bldg / ind / phase1 / road / rail) and the Phase-1 industrial
baseline included as explicit point loads. Zone-level file:
`base_year_2024_hourly_with_phase1.parquet` (long: zone, datetime_utc,
sector, mwh). Methods: `methodology/demand-methodology.md` and
`methodology/demand-components-2026-08-19.md` in the main project folder.

## Regenerate the bus-level demand

From the repo root:

```bash
# 1. zone -> substation (sector-aware weights; energy conserved exactly)
python3 scripts/downscale_zone_to_bus.py \
    data/base_year_2024/base_year_2024_hourly_with_phase1.parquet \
    --data-dir data/base_year_2024 --outdir out

# 2. substation -> PyPSA-USA power_electricity format (bus_id columns,
#    each sub split across its buses by Pd; snapshot index for horizon 2024)
python3 scripts/pypsa_demand_to_power_electricity.py \
    out/bus_hourly_base_year_2024_hourly_with_phase1.parquet \
    --bus-gis data/base_year_2024/bus_gis.csv \
    --out out/power_electricity.csv
```

Drop `out/power_electricity.csv` at
`resources/western/demand/power_electricity.csv` in the PyPSA-USA workflow
(the file `build_electrical_demand` would have written; `add_demand.py` reads
it directly). Config must be planning horizon **2024 only**, 8760 snapshots —
`add_demand.py` asserts the row count matches `n.snapshots` exactly.

Expected check numbers: 759.07 TWh total, 140.8 GW coincident peak,
10,157 bus columns. `bus_demand_base_year_2024_hourly_with_phase1.csv` is the
committed substation × sector annual table for sanity checks without
regenerating.

## Files

| file | role |
|---|---|
| `base_year_2024_hourly_with_phase1.parquet` | zone-level hourly demand, 5 sectors, sums to EIA-930 |
| `wecc_substation_weights.csv` | substation Pd weights (default within-zone spread) |
| `substation_industrial_total_2024.csv` | industrial remainder by substation (ind-sector weights) |
| `sub_to_pypsa_ba.csv` | substation → model zone (carries CISO subregions) |
| `phase1_point_loads_2024.csv` | Phase-1 facility baseline → exact substation placement |
| `bus_gis.csv` | PyPSA bus ↔ substation with bus Pd (from pypsa-usa-wecc resources) |
| `bus_demand_base_year_2024_hourly_with_phase1.csv` | committed bus-level annual table (sanity check) |
