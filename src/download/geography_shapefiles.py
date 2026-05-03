
import datetime, calendar, glob, os, requests, pickle

import pandas as pd
import geopandas as gpd

from shapely.geometry import box

path_to_maps = "/Users/Guille/Desktop/wecc_model/input_data/geography/maps/"

# Download WECC shapefile ----------------------------- 
url = "https://services2.arcgis.com/z5VWEXupJpBJnZRq/ArcGIS/rest/services/WECC_Balancing_Authorities/FeatureServer/0/query"

params = {"where": "1=1",
          "outFields": "*",
          "outSR": "4326",
          "f": "geojson"}

data = requests.get(url, params=params).json()

WECC_ = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")

WECC_ = WECC_.rename(columns={
    "Shape__Area": "ShapeArea",
    "Shape__Length": "ShapeLen"
})

# Make directory if does not exits
os.makedirs(path_to_maps + 'WECC/', exist_ok=True)

# Save
WECC_.to_file(path_to_maps + 'WECC/' + "WECC.shp", 
              driver="ESRI Shapefile")

# Download IPCO shapefile ----------------------------- 
url = "https://services1.arcgis.com/A5HVOoaVqcd8etEM/arcgis/rest/services/Regions/FeatureServer/0/query"

params = {"where": "1=1",
          "outFields": "*",
          "outSR": "4326",
          "f": "geojson"}

data = requests.get(url, params=params).json()

IPCO_ = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")

IPCO_ = IPCO_[["REGIONNAME", "geometry"]]

# Reassign regions
IPCO_["REGIONNAME"] = IPCO_["REGIONNAME"].replace({
    "SOUTHERN": "EASTERN",
    "CANYON": "WESTERN"
})

# Merge geometries by region
IPCO_ = IPCO_.dissolve(by="REGIONNAME").reset_index()
IPCO_ = IPCO_.rename(columns={"REGIONNAME": "name"})

IPCO_.loc[0, "name"] = 'Idaho Power (Magic Valley)'
IPCO_.loc[1, "name"] = 'Idaho Power (Far East)'
IPCO_.loc[2, "name"] = 'Idaho Power (Treasour Valley)'

IPCO_.loc[0, "load_zone"] = 'IPMV'
IPCO_.loc[1, "load_zone"] = 'IPFE'
IPCO_.loc[2, "load_zone"] = 'IPTV'

IPCO_.loc[0, "ba"] = 'IPCO'
IPCO_.loc[1, "ba"] = 'IPCO'
IPCO_.loc[2, "ba"] = 'IPCO'

# Make directory if does not exits
os.makedirs(path_to_maps + 'IPCO/', exist_ok=True)

# Save
IPCO_.to_file(path_to_maps + 'IPCO/' + "IPCO.shp", 
              driver="ESRI Shapefile")

# Download NVP shapefile ----------------------------- 
url = "https://services6.arcgis.com/RtGDJSaL1ODxRiz9/arcgis/rest/services/Nevada_Energy_Service_Territory/FeatureServer/0/query"

params = {"where": "1=1",
          "outFields": "*",
          "outSR": "4326",
          "f": "geojson"}

data = requests.get(url, params=params).json()

NEVP_ = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")

# Select rows
subset = NEVP_.loc[[0, 1, 2]]

# Merge geometries
merged_geom = subset.union_all()

# Merge attributes (first non-null value)
merged_attrs = subset.bfill().iloc[0].drop("geometry")

# Create merged GeoDataFrame
merged = gpd.GeoDataFrame(
    [merged_attrs],
    geometry=[merged_geom],
    crs=NEVP_.crs
)

# Drop original rows
NEVP_ = NEVP_.drop(index=[0, 1, 2])

# Append merged feature
NEVP_ = gpd.GeoDataFrame(
    pd.concat([NEVP_, merged], ignore_index=True),
    crs=NEVP_.crs
)

NEVP_ = NEVP_[['geometry']]

NEVP_.loc[0, 'name'] = 'Nevada Power'
NEVP_.loc[1, 'name'] = 'Sierra Pacific Power'

NEVP_.loc[0, "load_zone"] = 'NEVP'
NEVP_.loc[1, "load_zone"] = 'SPPC'

NEVP_.loc[0, "ba"] = 'NEVP'
NEVP_.loc[1, "ba"] = 'NEVP'

# Make directory if does not exits
os.makedirs(path_to_maps + 'NEVP/', exist_ok=True)

# Save
NEVP_.to_file(path_to_maps + 'NEVP/NEVP.shp', 
              driver="ESRI Shapefile")

# Download US GRID shapefile ----------------------------- 
url = "https://services2.arcgis.com/z5VWEXupJpBJnZRq/arcgis/rest/services/WECC_Transmission_Lines/FeatureServer/0/query"

params = {"where": "1=1",
          "outFields": "*",
          "outSR": "4326",
          "f": "geojson"}

data = requests.get(url, params=params).json()

TX_ = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")

TX_ = TX_.rename(columns={
    "Shape__Area": "ShapeArea",
    "Shape__Length": "ShapeLen",
})

# Make directory if does not exits
os.makedirs(path_to_maps + 'us_tx_lines/', exist_ok=True)

# Save
TX_.to_file(path_to_maps + 'us_tx_lines/' + "us_tx_lines.shp", 
              driver="ESRI Shapefile")


# Download WECC GRID shapefile ----------------------------- 
# url = "https://services2.arcgis.com/z5VWEXupJpBJnZRq/arcgis/rest/services/All_Transmission_WECC_JUL2021/FeatureServer/0/query"

# params = {"where": "1=1",
#           "outFields": "*",
#           "outSR": "4326",
#           "f": "geojson"}

# data = requests.get(url, params=params).json()

# WECC_TX_ = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")

# WECC_TX_ = WECC_TX_.rename(columns={
#     "Shape__Area": "ShapeArea",
#     "Shape__Length": "ShapeLen",
# })

# # Make directory if does not exits
# os.makedirs(path_to_maps + 'weec_tx_lines/', exist_ok=True)

# # Save
# WECC_TX_.to_file(path_to_maps + 'weec_tx_lines/' + "wecc_tx_lines.shp", 
#               driver="ESRI Shapefile")


