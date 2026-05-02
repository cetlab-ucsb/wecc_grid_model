
import datetime, calendar, glob, os, requests, pickle

import pandas as pd
import geopandas as gpd

from shapely.geometry import box
from shapely.ops import unary_union

path_to_data = "/Users/Guille/Desktop/map/"

# Download WECC shapefile
url = "https://services2.arcgis.com/z5VWEXupJpBJnZRq/ArcGIS/rest/services/WECC_Balancing_Authorities/FeatureServer/0/query"

params = {"where": "1=1",
          "outFields": "*",
          "outSR": "4326",
          "f": "geojson"}

data = requests.get(url, params=params).json()

WECC_ = gpd.GeoDataFrame.from_features(data["features"], crs="EPSG:4326")

# Make directory if does not exits
os.makedirs(path_to_data + 'WECC/', exist_ok=True)

# Save
WECC_.to_file(path_to_data + 'WECC/' + "WECC.shp", 
              driver="ESRI Shapefile")

# Download WECC shapefile
