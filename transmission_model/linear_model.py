from datetime import datetime
import json
import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from transmission_model.get_env_inputs import get_environmental_inputs


# with open("transmission_model/output_json/sei_backward_seeding_outputs_sample_augmented.json", "r") as f:
#     sei_outputs = json.load(f)

# Load results_2022.json with predictor features
# with open("results_2022.json", "r") as f:
#     results_2022 = json.load(f)

# TODO scale abundance for wild


def train_model(train_filepath):
    '''parse and format (into json input form with abundance) json poultry training data
    check if output file has all values from csv, then check if it has 14 days of data
    call get_environmental_inputs and append to new file
    '''

    df = pd.read_json("poultry_beta_scaled.json").T.reset_index()
    df.columns = ['key', 'county', 'state', 'date', 'total_abundance', 'species', 'beta', 'latitude', 'longitude', 'scaled_abundance'] 

    filename ="poultry_outbreak_features.json"
    try:
        with open(filename, "r") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}



    # Parse date if needed


    # get environmental features for every outbreak, store in "poultry_outbreaks_features.json"
    for _, row in df.iterrows():
        key = f"WAHIS/Poultry/{row['species']}/{row['date']}/{row['state']}/{row['county']}/{row['latitude']}/{row['longitude']}"

        # check if query in "poultry_outbreak_features.json"
        if key in existing_data:
            continue
        lat = row['latitude']
        lon = row['longitude']
        date_str = row['date']
        abundance = row['scaled_abundance']


        if isinstance(date_str, str):
            date = datetime.strptime(date_str, "%d/%m/%Y").date()
        else:
            date = date_str
            
        result, temp_C, rh_percent, precip_mm, dist_to_reservoir_km = get_environmental_inputs(key, filename)

        existing_data[key] = {
            "county": row["county"],
            "state": row["state"],
            "date": str(date),
            "total_abundance": float(row["scaled_abundance"]),
            "species": row["species"],
            "beta": float(row["beta"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "climate": result.get("climate", {})  # safely attach weather data
        }


        with open(filename, "w") as f:
            json.dump(existing_data, f, indent=2)


    
    
#     # load sei_outputs_random.json with true beta values
#     with open("transmission_model/output_json/sei_backward_seeding_outputs_sample_augmented.json", "r") as f:
#         sei_outputs = json.load(f)

#     # Load results_2022.json with predictor features
#     with open("results_2022.json", "r") as f:
#         results_2022 = json.load(f)

#     X, y = [], []

#     count = 0

#     # train on half the values
#     for entry in results_2022:
#         for uuid, sei_entry in sei_outputs.items():
#             if count >= (len(sei_outputs.items()))/2:
#                 break
#             uuid_parts = uuid.split('/')
#             if uuid_parts[6].isalpha() or uuid_parts[7].isalpha():
#                 uuid = uuid_parts[8] + '/' + uuid_parts[9] + '/' + uuid_parts[5] + '-' + uuid_parts[3] + '-' + uuid_parts[4]
#             else:
#                 uuid = uuid_parts[6] + '/' + uuid_parts[7] + '/' + uuid_parts[5] + '-' + uuid_parts[3] + '-' + uuid_parts[4]

#             if entry['uuid'] == uuid:
#                 result = entry

#         beta = sei_entry["metadata"].get("beta")
#         if beta is None:
#             continue

#         climate = result["climate"]["outbreak_day"]
#         X.append([
#             climate["2m_temperature"],
#             climate["2m_relative_humidity"],
#             climate["precipitation_flux"],
#             result.get("distance_inland", 0),
#             result.get("distance_to_water", 0),
#             result.get("percentage_wetland", 0),
#         ])
#         y.append(beta)

#     model = LinearRegression()
#     model.fit(X, y)

#     # Save trained model
#     joblib.dump(model, "trained_beta_model.pkl")
#     print("Trained beta regression model saved to trained_beta_model.pkl")


# # Load trained regression model

def calculate_beta_linear_from_model(lat, lon, date_str, info):
    pass
#     beta_model = joblib.load("trained_beta_model.pkl")
#     for entry in results_2022:
#         if entry['uuid'] == str(lat) + "/" + str(lon) + "/" + date_str:
#             info = entry
#     climate = info["climate"]["outbreak_day"]
#     X = [[
#         climate["2m_temperature"],
#         climate["2m_relative_humidity"],
#         climate["precipitation_flux"],
#         info.get("distance_inland", 0),
#         info.get("distance_to_water", 0),
#         info.get("percentage_wetland", 0),
#     ]]
#     return beta_model.predict(X)[0]
