
from math import hypot
from datetime import datetime, timedelta
import os
import csv
import xarray as xr
import cdsapi
import zipfile

DATA_SET = "sis-agrometeorological-indicators"
features = ["2m_temperature", "2m_relative_humidity", "precipitation_flux"]


def extract_cds_data(year):
    # check if Data folder exists in current directory and is not empty
    # if exists:
    #     return "Data Extracted Already: continue with search"
    # read through "US_States_Coordinates.csv" file for each state, north, south, east, west value

    # Read through "US_States_Coordinates.csv" and call api_request for each state
    with open("US_States_Coordinates.csv", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            state = row["State"]
            north = float(row["North"])
            south = float(row["South"])
            east = float(row["East"])
            west = float(row["West"])

            # call api request function
            for variable in features:
                target_dir = os.path.join("Data", state, variable)
                if os.path.exists(target_dir) and any(os.scandir(target_dir)):
                    print(f"Skipping {state} - {variable} (already exists)")
                    continue
                api_request(state, north, south, east, west, variable, year)

    return "Data extraction complete."

def api_request(state, north, south, east, west, variable, year):
    # download data
    client = cdsapi.Client()
    area = [north, west, south, east]
    request = {
    "variable": [variable],
    "year": [year],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "version": "1_1",
    "area": area,
    }

    # add extra parameter for 2m_temperature
    if variable == "2m_temperature":
        request["statistic"] = ["24_hour_mean"]
    if variable == "2m_relative_humidity":
        request["time"] = ["12_00"]

    # Retrieve the data
    result = client.retrieve(DATA_SET, request)
    temp_filename = result.download() 

    #create target directory: Data/{state}/{variable}
    target_dir = os.path.join("Data", state, variable)
    os.makedirs(target_dir, exist_ok=True)


    # Extract contents
    with zipfile.ZipFile(temp_filename, 'r') as zip_ref:
        zip_ref.extractall(target_dir)

    os.remove(temp_filename)
    print(f"Data for {state} ({variable}, {year}) extracted to {target_dir}")

    # convert netcdf files to csv and delete original .nc files
    for file in os.listdir(target_dir):
        if file.endswith(".nc"):
            nc_path = os.path.join(target_dir, file)
            csv_path = os.path.join(target_dir, file.replace(".nc", ".csv"))

            try:
                ds = xr.open_dataset(nc_path)
                df = ds.to_dataframe().reset_index()
                df.to_csv(csv_path, index=False)
                print(f"Converted {file} to CSV.")
                os.remove(nc_path)  # delete .nc file after conversion
            except Exception as e:
                print(f"Failed to convert {file}: {e}")

    # store in correct format


# search for closest value by distance
def extract_climate(query):
    # parse query
    parts = query.split('/')
    state = parts[6]
    county = parts[7]
    date_str = parts[3]  + '/' + parts[4] + '/' + parts[5]  # YYYY-MM-DD
    lat_query = float(parts[8])
    lon_query = float(parts[9])
    date_obj = datetime.strptime(date_str, "%Y-%m-%d") 

    print(f"Searching for: {state}, {county}, {date_str} near ({lat_query}, {lon_query})")

    # search through data for matching date value for given state
    full_results = [
    {
        "Location": f"{lat_query}/{lon_query}",
        f"Day {i+1}": "",
        "2m_temperature": "",
        "2m_relative_humidity": "",
        "precipitation_flux": ""
    }
    for i in range(14)
    ]

    for variable in features:
        variable_dir = os.path.join("Data", state, variable)
        if not os.path.exists(variable_dir):
            print(f"Directory not found: {variable_dir}")
            continue
        results = []
        for day_offset in range(14):
            current_date = date_obj - timedelta(days=day_offset)
            date_str_fmt = current_date.strftime("%Y%m%d")

            # find file with matching date
            for fname in os.listdir(variable_dir):
                if date_str_fmt in fname and fname.endswith(".csv"):
                    filepath = os.path.join(variable_dir, fname)

                    with open(filepath, 'r') as f:
                        reader = csv.DictReader(f)
                        min_dist = float('inf')
                        closest_row = None

                        # find row with minimum distance from location
                        for row in reader:
                            lat = float(row["lat"])
                            lon = float(row["lon"])
                            dist = hypot(lat - lat_query, lon - lon_query) 
                            if dist < min_dist:
                                min_dist = dist
                                closest_row = row

                    if closest_row:
                        if variable == "2m_temperature":
                            results.append({"date": date_obj.strftime("%Y-%m-%d"), "value": closest_row['Temperature_Air_2m_Mean_24h']})
                        elif variable == "2m_relative_humidity":
                            results.append({"date": date_obj.strftime("%Y-%m-%d"), "value": closest_row['Relative_Humidity_2m_12h']})
                        else:
                            results.append({"date": date_obj.strftime("%Y-%m-%d"), "value": closest_row['Precipitation_Flux']})

        for index, result in enumerate(results):
            full_results[index][f"Day {index+1}"] = result['date']
            full_results[index][f"{variable}"] = result['value']

    return full_results