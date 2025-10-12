from pyproj import Geod
import rasterio
import numpy as np
from rasterio.transform import rowcol
from rasterio.windows import Window


raster_path_reservoir = "data_retrieval/reservoir_raster/glwd_3/w001001.adf"
#radius in pixels to consider afround the point
window_size = 5

def extract_reservoir_distance(query, result):
    # fefine geodesic calculator for distance (WGS84)
    geod = Geod(ellps="WGS84")

    # parse query
    parts = query.split('/')
    state = parts[6]
    county = parts[7]
    date_str = parts[3]  + '/' + parts[4] + '/' + parts[5]  # YYYY-MM-DD
    lat_query = float(parts[8])
    lon_query = float(parts[9])

    print(f"Searching for reservoir dist for {date_str} near ({lat_query}, {lon_query})")

    with rasterio.open(raster_path_reservoir) as src:
        data = src.read(1)
        # find resevoirs (value =2) or lakes (value=1)i
        water_mask = (
            (data == 1) |
            (data == 2) 
        )

        # row/col of resevoir cells
        rows, cols = np.where(water_mask)
        # convert to geographic coordinates
        reservoir_coords = [src.xy(r, c) for r, c in zip(rows, cols)]

        # calculate distance using geodesic
        min_distance_km = float("inf")
        for lon, lat in reservoir_coords:
            _, _, distance = geod.inv(lon_query, lat_query, lon, lat)
            min_distance_km = min(min_distance_km, distance / 1000.0)
        print(f"Shortest distance to nearest reservoir: {min_distance_km:.2f} km")

        result['distance_to_water'] = round(min_distance_km, 2)
    return result

def extract_reservoir_percentage(query, result):

    # parse query
    parts = query.split('/')
    state = parts[6]
    county = parts[7]
    date_str = parts[3]  + '/' + parts[4] + '/' + parts[5]  # YYYY-MM-DD
    lat_query = float(parts[8])
    lon_query = float(parts[9])

    print(f"Searching for reservoir percentage for {date_str} near ({lat_query}, {lon_query})")


    '''
    Cell Value | Lake or Wetland type
    1 Lake
    2 Reservoir
    3 River
    4 Freshwater Marsh, Floodplain
    5 Swamp Forest, Flooded Forest
    6 Coastal Wetland (incl. Mangrove, Estuary, Delta, Lagoon)
    7 Pan, Brackish/Saline Wetland
    8 Bog, Fen, Mire (Peatland)
    9 Intermittent Wetland/Lake
    10 50-100% Wetland
    11 25-50% Wetland
    12 Wetland Compex (0-25% Wetland)
    '''
    # all types of wetlands, lakes, reservoirs
    wetland_classes = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

    with rasterio.open(raster_path_reservoir) as src:
        # get pixel location
        row, col = rowcol(src.transform, lon_query, lat_query)
        # define window around pixel
        win = Window(col - window_size, row - window_size, 2 * window_size + 1, 2 * window_size + 1)
        win = win.intersection(Window(0, 0, src.width, src.height))
        data = src.read(1, window=win)

        # mask out not used pixels
        data = data[data != src.nodata]

        if data.size == 0:
            result['percentage_wetland'] = 0
            return result

        #if in wetland calculate percentage
        wetland_count = np.isin(data, list(wetland_classes)).sum()
        total_count = data.size
        print(f"total count {total_count} and wetland count = {wetland_count}")

        percent = (wetland_count / total_count) * 100
        
        result['percentage_wetland'] = percent
    return result
