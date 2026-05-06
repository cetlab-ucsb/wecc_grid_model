import pandas as pd

# Load the PUDL yearly generators table from the nightly Parquet release
yearly_generators = pd.read_parquet(
    "s3://pudl.catalyst.coop/nightly/out_eia__yearly_generators.parquet",
    dtype_backend="pyarrow",
)

# Create a unique plant-generator ID by combining the EIA plant ID and generator ID
yearly_generators["plant_gen_combined_id"] = (
    yearly_generators["plant_id_eia"].astype("string")
    + "_"
    + yearly_generators["generator_id"].astype("string")
)

# Convert report_date to datetime format
yearly_generators["report_date"] = pd.to_datetime(
    yearly_generators["report_date"]
)

# Create a 2024 generator snapshot.
# If a plant-generator appears more than once, keep the last row.
yearly_generators_2024 = (
    yearly_generators
    .loc[yearly_generators["report_date"] == "2024-01-01"]
    .drop_duplicates(subset="plant_gen_combined_id", keep="last")
    .reset_index(drop=True)
)

# Separate proposed generators
yearly_generators_2024_proposed = yearly_generators_2024.loc[
    yearly_generators_2024["operational_status"] == "proposed"
]

# Keep only existing generators
yearly_generators_2024 = yearly_generators_2024.loc[
    yearly_generators_2024["operational_status"] == "existing"
]

# List of states that are fully or partially within the WECC region
states = ["AZ", "CA", "CO", "ID", "MT", "NE", "NV", "NM", "OR", "SD", "TX", "UT", "WA", "WY"]

# Filter the 2024 existing generators to only those located in WECC states
yg_wecc_states = yearly_generators_2024.loc[
    yearly_generators_2024["state"].isin(states)
]

# Summarize total generator capacity by state for the selected WECC states
yg_statewise = (
    yg_wecc_states
    .groupby("state", as_index=False)["capacity_mw"]
    .sum()
    .rename(columns={"capacity_mw": "total_capacity_mw"})
)

# Save the state-level capacity summary to an Excel file
# yg_statewise.to_excel(
#     "eia_yearly_generators_statewise_summary__2024_4_29.xlsx",
#     index=False
# )

# Load the "load_zones.csv" file to obtain required balancing authority codes
# The CSV is expected to have a column called "geography".
load_zones = pd.read_csv("../../../input_data/load_zone/load_zones.csv")["geography"].unique()

# From the WECC-state generators, keep only generators whose balancing
# authority code matches one of the load zone codes.
yg_wecc = yg_wecc_states.loc[
    yg_wecc_states["balancing_authority_code_eia"].isin(load_zones)
]

# Technology Description Related Columns:
    # "technology_description"
    # "prime_mover_code"
    # "energy_source_code_1"
    # "fuel_type_code_pudl"
    # "fuel_type_count"

# load energy source code for mapping
energy_source_code = (
    pd.read_csv("../../../input_data/core_eia__codes_energy_sources.csv")
    [["code", "label", "fuel_derived_from"]]
    .rename(
        columns={
            "code": "energy_source_code_1",
            "label": "energy_source_code_label",
            "fuel_derived_from": "energy_source_code_fuel_derived_from",
        }
    )
)

# load prime mover source code for mapping
prime_mover_code = (
    pd.read_csv("../../../input_data/core_eia__codes_prime_movers.csv")
    .rename(
        columns={
            "code": "prime_mover_code",
            "label": "prime_mover_label",
            "description": "prime_mover_description",
        }
    )
)

# merge energy source code and prime mover code mapping dfs to yg_wecc
yg_wecc = yg_wecc.merge(
    energy_source_code,
    on = 'energy_source_code_1',
    how = 'left').merge(
        prime_mover_code,
        on = 'prime_mover_code',
        how = 'left'
    )


generator_type_capacity = yg_wecc[
    [
        "capacity_mw",
        "technology_description",
        "prime_mover_code",
        "prime_mover_label",
        "prime_mover_description",
        "energy_source_code_1",
        "energy_source_code_label",
        "energy_source_code_fuel_derived_from"

    ]
].copy()

generator_type_capacity_summary = (
    generator_type_capacity
    .groupby(
        [
            "technology_description",
            "prime_mover_code",
            "prime_mover_label",
            "prime_mover_description",
            "energy_source_code_1",
            "energy_source_code_label",
            "energy_source_code_fuel_derived_from"
        ],
        as_index=False,
        dropna=False,
    )
    .agg(
        n_rows=("capacity_mw", "size"),
        total_capacity_mw=("capacity_mw", "sum"),
    )
    .reset_index(drop=True)
)

# generator_type_capacity_summary.to_excel('generator_types_summary.xlsx', index=False)

cc_steam = yg_wecc.loc[yg_wecc['prime_mover_code'] == 'CA']

cc_gas = yg_wecc.loc[yg_wecc['prime_mover_code'] == 'CT']

cc = yg_wecc.loc[yg_wecc['technology_description'] == 'Natural Gas Fired Combined Cycle'][[
        "plant_id_eia",
        "generator_id",
        "technology_description",
        "prime_mover_code",
        "prime_mover_label",
        "prime_mover_description",
        "energy_source_code_1",
        "energy_source_code_label",
        "energy_source_code_fuel_derived_from",
        "capacity_mw"]]

