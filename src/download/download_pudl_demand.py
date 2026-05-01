"""Download PUDL hourly demand data for WECC balancing authorities.

First-pass demand pipeline for the WECC GridPath model.

Scope (decided 2026-04-19 with PI):
  * BA-level load zones (utility-level disaggregation is a later pass).
  * Weather year 2024 (most recent full year of EIA-930).
  * Single load_component = 'all' (sector split is a later pass).

What this script does:
  1. Downloads EIA-930 hourly BA operations parquet from PUDL's public S3.
  2. Filters to 2024 and the WECC BA list.
  3. (Optionally) downloads FERC-714 hourly planning-area demand for sub-BA
     zones that EIA-930 does not break out (PacifiCorp East, Idaho Power).
  4. Saves filtered parquet + a compact CSV preview into
     local_data/input_data/demand/raw/ for inspection.

This is a DOWNLOAD + INSPECT script. It does NOT yet write GridPath .tab
files — that's the next step (see PUDL_demand_data_reference.md section 5).

Data sources:
  * out_eia930__hourly_operations  — PUDL-imputed hourly BA demand
  * out_ferc714__hourly_planning_area_demand  — planning-area demand
  * core_eia861__yearly_sales  — annual utility sales (for later utility pass)

Access:
  * S3 (no auth):
      https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop/nightly/<table>.parquet
  * See: https://catalystcoop-pudl.readthedocs.io/en/stable/data_access.html

Run:
    python download_pudl_demand.py
    python download_pudl_demand.py --year 2024 --skip-ferc714
    python download_pudl_demand.py --release v2024.11.0   # pinned instead of nightly
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

# PUDL public S3 bucket. "nightly" is the rolling build; swap for a tagged
# release like "v2024.11.0" for reproducibility.
PUDL_S3_BASE = "https://s3.us-west-2.amazonaws.com/pudl.catalyst.coop"

# Tables we care about for demand.
TABLES = {
    "eia930_hourly":   "out_eia930__hourly_operations",
    "eia930_subregion": "core_eia930__hourly_subregion_demand",
    "ferc714_hourly":  "out_ferc714__hourly_planning_area_demand",
    "ferc714_respondents": "out_ferc714__respondents_with_fips",
    "eia861_sales":    "core_eia861__yearly_sales",
    "eia861_ba_assn":  "core_eia861__assn_balancing_authority",
}

# WECC balancing authorities that are directly reported in EIA-930.
# Source: existing local_data/input_data/load_zone/load_zones.csv.xlsx,
# filtered to US BAs that EIA-930 reports natively (CAISO subregions and
# FERC-714 sub-planning-areas are handled separately).
WECC_EIA930_BAS = [
    "AVA",   # Avista
    "AZPS",  # Arizona Public Service
    "BANC",  # Balancing Authority of Northern California
    "BPAT",  # Bonneville Power Administration
    "CHPD",  # Chelan County PUD
    "CISO",  # California ISO (aggregate; split into subregions below if needed)
    "DOPD",  # Douglas County PUD
    "EPE",   # El Paso Electric
    "GCPD",  # Grant County PUD
    "IID",   # Imperial Irrigation District
    "IPCO",  # Idaho Power
    "LDWP",  # LA Dept of Water & Power
    "NEVP",  # Nevada Power / NV Energy South
    "NWMT",  # NorthWestern Energy (Montana)
    "PACE",  # PacifiCorp East
    "PACW",  # PacifiCorp West
    "PGE",   # Portland General Electric
    "PNM",   # Public Service of New Mexico
    "PSCO",  # Public Service of Colorado
    "PSEI",  # Puget Sound Energy
    "SCL",   # Seattle City Light
    "SRP",   # Salt River Project
    "TEPC",  # Tucson Electric Power
    "TIDC",  # Turlock Irrigation District
    "TPWR",  # Tacoma Power
    "WACM",  # Western Area Power - Rocky Mountain
    "WALC",  # Western Area Power - Desert Southwest
    "WAUW",  # Western Area Power - Upper Great Plains West
]

# WECC load zones that are NOT in EIA-930 at BA level — these need different
# sources. Recorded here for tracking; not pulled by this script yet.
NON_PUDL_LOAD_ZONES = {
    "AESO": "Alberta ESO (ets.aeso.ca)",
    "BCHA": "BC Hydro (bchydro.com)",
    "CFE":  "CENACE (cenace.gob.mx) — Mexican side of WECC",
}

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]       # .../wecc_grid_model/
WECC_ROOT = REPO_ROOT.parent                           # .../wecc/
DEFAULT_OUT = WECC_ROOT / "local_data" / "input_data" / "demand" / "raw"


# --------------------------------------------------------------------------
# Core helpers
# --------------------------------------------------------------------------

def pudl_table_url(table_name: str, release: str = "nightly") -> str:
    return f"{PUDL_S3_BASE}/{release}/{table_name}.parquet"


def load_parquet(table_key: str, columns: list[str] | None = None,
                 release: str = "nightly") -> pd.DataFrame:
    """Download a PUDL parquet table into memory."""
    table = TABLES[table_key]
    url = pudl_table_url(table, release=release)
    print(f"[fetch] {table}  <- {url}")
    df = pd.read_parquet(url, columns=columns)
    print(f"[fetch] {table}  rows={len(df):,} cols={len(df.columns)}")
    return df


def pull_eia930_hourly(year: int, release: str, out_dir: Path) -> pd.DataFrame:
    """EIA-930 hourly BA operations, filtered to WECC + year."""
    # Only pull the columns we actually need (parquet column pruning is free).
    cols = [
        "balancing_authority_code_eia",
        "datetime_utc",
        "demand_reported_mwh",
        "demand_adjusted_mwh",
        "demand_imputed_pudl_mwh",
        "demand_imputed_pudl_mwh_imputation_code",
        "demand_imputed_eia_mwh",
    ]
    try:
        df = load_parquet("eia930_hourly", columns=cols, release=release)
    except Exception as e:
        # Column list may drift between releases — fall back to pulling everything.
        print(f"[warn] column-filtered read failed ({e}); retrying with all columns")
        df = load_parquet("eia930_hourly", release=release)

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    mask = (
        df["balancing_authority_code_eia"].isin(WECC_EIA930_BAS)
        & (df["datetime_utc"].dt.year == year)
    )
    out = df.loc[mask].copy().sort_values(
        ["balancing_authority_code_eia", "datetime_utc"]
    )

    dst = out_dir / f"eia930_hourly_wecc_{year}.parquet"
    out.to_parquet(dst, index=False)
    print(f"[save]  {dst}  rows={len(out):,}")
    return out


def pull_ferc714_hourly(year: int, release: str, out_dir: Path) -> pd.DataFrame:
    """FERC-714 hourly planning-area demand, filtered to year.

    We don't filter to WECC respondents here because the crosswalk
    (respondents_with_fips) needs to be joined first. That join is
    left to a downstream processing step; this script just lands the
    raw year-of-data for inspection.
    """
    cols = [
        "respondent_id_ferc714",
        "report_date",
        "datetime_utc",
        "timezone",
        "demand_reported_mwh",
        "demand_imputed_pudl_mwh",
        "demand_imputed_pudl_mwh_imputation_code",
    ]
    try:
        df = load_parquet("ferc714_hourly", columns=cols, release=release)
    except Exception as e:
        print(f"[warn] column-filtered read failed ({e}); retrying with all columns")
        df = load_parquet("ferc714_hourly", release=release)

    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    out = df.loc[df["datetime_utc"].dt.year == year].copy()

    dst = out_dir / f"ferc714_hourly_{year}.parquet"
    out.to_parquet(dst, index=False)
    print(f"[save]  {dst}  rows={len(out):,}")
    return out


def pull_ferc714_respondents(release: str, out_dir: Path) -> pd.DataFrame:
    """Crosswalk of FERC-714 respondents → BA/utility."""
    df = load_parquet("ferc714_respondents", release=release)
    dst = out_dir / "ferc714_respondents_with_fips.parquet"
    df.to_parquet(dst, index=False)
    print(f"[save]  {dst}  rows={len(df):,}")
    return df


def pull_eia861_sales(release: str, out_dir: Path) -> pd.DataFrame:
    """Annual utility sales by customer class (residential/commercial/industrial)."""
    df = load_parquet("eia861_sales", release=release)
    dst = out_dir / "eia861_yearly_sales.parquet"
    df.to_parquet(dst, index=False)
    print(f"[save]  {dst}  rows={len(df):,}")
    return df


# --------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------

def inspect_eia930(df: pd.DataFrame, out_dir: Path) -> None:
    """Print coverage summary and save a small CSV preview."""
    print("\n[inspect] EIA-930 coverage by BA (2024)")
    grp = (
        df.groupby("balancing_authority_code_eia")
          .agg(
              hours=("datetime_utc", "count"),
              missing_reported=("demand_reported_mwh", lambda s: s.isna().sum()),
              missing_imputed=("demand_imputed_pudl_mwh", lambda s: s.isna().sum()),
              mean_demand_mw=("demand_imputed_pudl_mwh", "mean"),
              peak_demand_mw=("demand_imputed_pudl_mwh", "max"),
          )
          .round(1)
    )
    print(grp.to_string())

    missing_bas = set(WECC_EIA930_BAS) - set(df["balancing_authority_code_eia"].unique())
    if missing_bas:
        print(f"\n[inspect] BAs requested but NOT in returned data: {sorted(missing_bas)}")

    # Save a small preview so the user can eyeball it without loading parquet.
    preview = df.head(200)
    preview.to_csv(out_dir / "eia930_hourly_wecc_preview.csv", index=False)
    print(f"[save]  {out_dir / 'eia930_hourly_wecc_preview.csv'}  (first 200 rows)")


# --------------------------------------------------------------------------
# Entrypoint
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, default=2024,
                    help="Weather year to pull (default 2024)")
    ap.add_argument("--release", default="v2026.1.0",
                    help="PUDL release tag. Default 'v2026.1.0' (latest tagged as "
                         "of 2026-04-19). Use 'nightly' for dev, 'stable' for "
                         "most-recent-tagged alias.")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                    help=f"Output directory (default {DEFAULT_OUT})")
    ap.add_argument("--skip-ferc714", action="store_true",
                    help="Skip FERC-714 hourly download")
    ap.add_argument("--skip-eia861", action="store_true",
                    help="Skip EIA-861 annual sales download")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[config] year={args.year}  release={args.release}  out={args.out_dir}")
    print(f"[config] WECC BAs in EIA-930 scope: {len(WECC_EIA930_BAS)}")
    print(f"[config] Non-PUDL load zones (skipped): {list(NON_PUDL_LOAD_ZONES)}")

    eia930 = pull_eia930_hourly(args.year, args.release, args.out_dir)
    inspect_eia930(eia930, args.out_dir)

    if not args.skip_ferc714:
        pull_ferc714_hourly(args.year, args.release, args.out_dir)
        pull_ferc714_respondents(args.release, args.out_dir)

    if not args.skip_eia861:
        pull_eia861_sales(args.release, args.out_dir)

    print("\n[done] Raw demand tables landed in:", args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
