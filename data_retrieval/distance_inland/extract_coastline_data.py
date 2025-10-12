
from datetime import datetime
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Geod


raster_path_coastline = "data_retrieval/ne_10m_coastline/ne_10m_coastline.shp"

def extract_inland_distance(query, result):

    # parse query
    parts = query.split('/')
    state = parts[6]
    county = parts[7]
    date_str = parts[3]  + '/' + parts[4] + '/' + parts[5]  # YYYY-MM-DD
    lat_query = float(parts[8])
    lon_query = float(parts[9])

    print(f"Searching for distance inland for {date_str} near ({lat_query}, {lon_query})")

    coastline = gpd.read_file(raster_path_coastline)

    # merge all line geometries
    merged_coast = coastline.geometry.union_all()
    # create point geometry
    point = Point(lon_query, lat_query)
    #find nearest point
    nearest = merged_coast.interpolate(merged_coast.project(point))

    # find geodesic distance
    geod = Geod(ellps="WGS84")
    _, _, dist_m = geod.inv(lon_query, lat_query, nearest.x, nearest.y)

    result['distance_inland'] = round(dist_m / 1000.0, 4)
    
    return result

