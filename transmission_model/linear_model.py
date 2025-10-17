from datetime import datetime
import json
import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from transmission_model.get_env_inputs import get_environmental_inputs
import pycatch22 as catch22

from sklearn.preprocessing import StandardScaler

CATCH_22 = False
scaler = StandardScaler()
# with open("transmission_model/output_json/sei_backward_seeding_outputs_sample_augmented.json", "r") as f:
#     sei_outputs = json.load(f)

# Load results_2022.json with predictor features
# with open("results_2022.json", "r") as f:
#     results_2022 = json.load(f)

# TODO scale abundance for wild

def add_time_series(outbreak_data):

    for key in outbreak_data:
        climate = outbreak_data[key]['climate']
        sorted_days = sorted(climate.keys(), key=lambda x: int(x.split('-')[-1]) if '-' in x else -1)
        sorted_days.reverse()

        temp_series = [climate[d]["2m_temperature"] for d in sorted_days if "2m_temperature" in climate[d]]
        humidity_series = [climate[d]["2m_relative_humidity"] for d in sorted_days if "2m_relative_humidity" in climate[d]]
        precip_series = [climate[d]["precipitation_flux"] for d in sorted_days if "precipitation_flux" in climate[d]]

        catch22_feats = {}
        if len(temp_series) > 0:
            temp_feats = catch22.catch22_all(temp_series)['values']
            humidity_feats = catch22.catch22_all(humidity_series)['values']
            precip_feats = catch22.catch22_all(precip_series)['values']
            
            combined_feats = np.concatenate([temp_feats, humidity_feats, precip_feats])
            catch22_feats = {"catch22_features": combined_feats.tolist()}

        # Add the features to the outbreak data
        outbreak_data[key]['catch22'] = catch22_feats
    return outbreak_data

def train_model(catch_22):
    '''parse and format (into json input form with abundance) json poultry training data
    check if output file has all values from csv, then check if it has 14 days of data
    call get_environmental_inputs and append to new file
    '''
    CATCH_22 = catch_22
    df = pd.read_json("poultry_beta_scaled.json").T.reset_index()
    df.columns = ['key', 'county', 'state', 'date', 'total_abundance', 'species', 'beta', 'latitude', 'longitude', 'scaled_abundance'] 

    filename ="poultry_outbreak_features.json"
    try:
        with open(filename, "r") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}



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
            
        result, temp_C, rh_percent, precip_mm, dist_to_reservoir_km = get_environmental_inputs(key)

        existing_data[key] = {
            "county": row["county"],
            "state": row["state"],
            "date": str(date),
            "scaled_abundance": float(row["scaled_abundance"]),
            "species": row["species"],
            "beta": float(row["beta"]),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "climate": result.get("climate", {}),  # safely attach weather data
            "distance_inland": result.get("distance_inland"),  
            "distance_to_water": result.get("distance_to_water")
        }

        with open(filename, "w") as f:
            json.dump(existing_data, f, indent=2)

        
    # Reload data for training
    # df = pd.read_json("poultry_outbreak_features.json").T.reset_index()
    # df.columns = ['key', 'county', 'state', 'date', 'scaled_abundance', 'species', 'beta', 'latitude', 'longitude', 'climate', 'distance_inland', 'distance_to_water'] 

    # Reload data for training
    with open("poultry_outbreak_features.json", "r") as f:
        poultry_features = json.load(f)
    

    if CATCH_22:
        poultry_features = add_time_series(poultry_features)

    X, y = [], []

    for key, v in poultry_features.items():
        climate = v.get("climate", {})
        abundance = v.get("scaled_abundance")
        beta = v.get("beta")
        dist_reservoir = v.get("distance_to_water") 
        dist_inland = v.get("distance_inland")
        if CATCH_22:
            catch_22_features = v.get("catch22")["catch22_features"]
        outbreak_day = climate.get("outbreak_day", {})
        temp = outbreak_day.get("2m_temperature")
        rh = outbreak_day.get("2m_relative_humidity")
        precip = outbreak_day.get("precipitation_flux")

        # Skip if climate data missing or empty
        if not climate:
            continue
        if CATCH_22:
            all_features = [temp, rh, precip, dist_reservoir, dist_inland, abundance] + catch_22_features
            if all(isinstance(x, (int, float)) and not np.isnan(x) for x in all_features):
                X.append(all_features)
                y.append(beta)
        else:
            if all(isinstance(x, (int, float)) for x in ([temp, rh, precip, dist_reservoir, dist_inland, abundance, beta] )):
                X.append([temp, rh, precip, dist_reservoir, dist_inland, abundance])
                y.append(beta)
            

    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split first 200 for training, rest for validation


    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_scaled, y)


    if len(X) == 0:
        print("No valid data points found for training.")
        return
    y_train_pred = model.predict(X_scaled)
    rmse_train = np.sqrt(np.mean((y - y_train_pred)**2))
    mae_train = np.mean(np.abs(y - y_train_pred))
    r2_train = model.score(X_scaled, y)

    print("\nTraining Metrics:")
    print(f"  RMSE: {rmse_train:.4f}")
    print(f"  MAE:  {mae_train:.4f}")
    print(f"  R^2:   {r2_train:.4f}")
    # TODO: include in explanation did not include MAPE since its not sensitive to very small values


    # Save model
    joblib.dump(model, "poultry_beta_model.joblib")

    joblib.dump(scaler, "scaler.joblib")
    print(f"Model trained and saved with {len(X)} samples.")

    # Print feature importances
    print("\nFeature coefficients:")
    if CATCH_22:
        base_features = ["temperature", "relative humidity", "precipitation", "distance inland", "scaled_abundance"]
        catch22_names = [f"catch22_{i}" for i in range(len(model.coef_) - len(base_features))]

        for name, coef in zip(base_features + catch22_names, model.coef_):
            print(f"  {name}: {coef:.4f}")
    else:
        if not CATCH_22: 
            for name, coef in zip(["temperature", "relative humidity", "precipitation", "distance inland", "scaled_abundance"], model.coef_): 
                print(f" {name}: {coef:.4f}")




def calculate_beta_linear_from_model(lat, lon, date_str, info, catch_22):
    beta_model = joblib.load("poultry_beta_model.joblib")
    scaler = joblib.load("scaler.joblib") 

    climate = info["climate"]["outbreak_day"]
    temp = climate.get("2m_temperature", 0)
    rh = climate.get("2m_relative_humidity", 0)
    precip = climate.get("precipitation_flux", 0)
    dist_to_water = info.get("distance_to_water", 0)
    dist_inland = info.get("distance_inland", 0)
    scaled_abundance = info.get("scaled_abundance", 0)

    if catch_22:
        catch22_features = (info.get("catch22", {}))["catch22_features"]
        # print(catch22_features)
        all_features = [temp, rh, precip, dist_to_water, dist_inland, scaled_abundance] + catch22_features
        # print(X)
    else:
        all_features = [temp, rh, precip, dist_to_water, dist_inland, scaled_abundance]
    
    X_scaled = scaler.transform([all_features])

    return beta_model.predict(X_scaled)[0]
