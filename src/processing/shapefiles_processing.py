
import datetime, calendar, glob, os, requests, pickle

import pandas as pd
import geopandas as gpd


path_to_maps = "/Users/Guille/Desktop/wecc_model/input_data/geography/maps/"

# Process PACE shapefile out of CAISO and US map ----------------------------- 

US_ = gpd.read_file(path_to_maps + "/US/tl_2022_us_state.shp")

# Remove non-continental states/territories
exclude = ['02', '15', '60', '66', '69', '72', '78']

CONUS_ = US_[~US_['STATEFP'].isin(exclude)]
CONUS_.plot()

WESTERNUS_ = CONUS_[CONUS_['REGION'] == '4']

#CAISO_ = gpd.read_file(path_to_maps + "/CAISO/Balancing_Authority_Areas_in_CA.shp")
CAISO_ = gpd.read_file(path_to_maps + "/CAISO/California_Electric_Utility_Service_Territory.shp")
PACE_ = CAISO_.loc[[11]]

# States to split by
states_keep = ["Idaho", "Utah", "Wyoming"]

# Select states
states_ = WESTERNUS_[
    WESTERNUS_["NAME"].isin(states_keep)
].copy()

# Ensure same CRS
PACE_ = PACE_.to_crs(WESTERNUS_.crs)

# Split PACE by selected states
PACE_states = gpd.overlay(
    PACE_,
    states_,
    how="intersection"
)

# Add region label
PACE_states["NAME_2"] = PACE_states["NAME_2"].str.upper()

# Remaining western portion
PACE_west = gpd.overlay(
    PACE_,
    states_,
    how="difference"
)

PACE_west["region"] = "WEST"

PACE_ = pd.concat([PACE_states, PACE_west], axis = 0).reset_index(drop = True)

PACE_ = PACE_[["region", "geometry", "NAME_2"]]

PACE_.loc[PACE_["NAME_2"] == "IDAHO", "name"] = 'PacifiCorp (East Idaho)'
PACE_.loc[PACE_["NAME_2"] == "UTAH", "name"] = 'PacifiCorp (East Utah)'
PACE_.loc[PACE_["NAME_2"] == "WYOMING", "name"] = 'PacifiCorp (East Wyoming)'
PACE_.loc[PACE_["region"] == "WEST", "name"] = 'PacifiCorp (West)'

PACE_.loc[PACE_["NAME_2"] == "IDAHO", "load_zone"] = 'PAID'
PACE_.loc[PACE_["NAME_2"] == "UTAH", "load_zone"] = 'PAUT'
PACE_.loc[PACE_["NAME_2"] == "WYOMING", "load_zone"] = 'PAWY'
PACE_.loc[PACE_["region"] == "WEST", "load_zone"] = 'PACW'

PACE_.loc[PACE_["NAME_2"] == "IDAHO", "ba"] = 'PACE'
PACE_.loc[PACE_["NAME_2"] == "UTAH", "ba"] = 'PACE'
PACE_.loc[PACE_["NAME_2"] == "WYOMING", "ba"] = 'PACE'
PACE_.loc[PACE_["region"] == "WEST", "ba"] = 'PACE'

PACE_ = PACE_.drop(columns=["region", "NAME_2"])
print(PACE_)


# Make directory if does not exits
os.makedirs(path_to_maps + 'PACE/', exist_ok=True)

# Save
PACE_.to_file(path_to_maps + 'PACE/PACE.shp', 
              driver="ESRI Shapefile")
