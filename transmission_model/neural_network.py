# train_nn.py
from datetime import datetime
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd

from transmission_model.get_env_inputs import get_environmental_inputs

scaler = StandardScaler()  # global so training + prediction use same scaler


# with open("results_2022.json", "r") as f:
#     data = json.load(f)


# with open("transmission_model/output_json/sei_backward_seeding_outputs_sample_augmented.json", "r") as f:
#     sei_data = json.load(f)
    
def train_nn():


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

    X, y = [], []

    for key, v in poultry_features.items():
        climate = v.get("climate", {})
        abundance = v.get("scaled_abundance")
        beta = v.get("beta")
        dist_reservoir = v.get("distance_to_water") 
        dist_inland = v.get("distance_inland")
        outbreak_day = climate.get("outbreak_day", {})
        temp = outbreak_day.get("2m_temperature")
        rh = outbreak_day.get("2m_relative_humidity")
        precip = outbreak_day.get("precipitation_flux")

        # Skip if climate data missing or empty
        if not climate:
            continue
        
        
        if all(isinstance(x, (int, float)) for x in [temp, rh, precip, dist_reservoir, dist_inland, abundance, beta]):
            X.append([temp, rh, precip, dist_reservoir, dist_inland, abundance])
            y.append(beta)


    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)

    # Scale inputs
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Define neural network
    model = Sequential([
        Dense(32, input_dim=X.shape[1], activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="linear")  # regression output
    ])

    # Compile
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

    # Train
    model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2, verbose=1)

    # Evaluate
    loss, mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {mae:.4f}")

    # Save model + scaler
    model.save("nn_beta_predictor.keras")
    np.save("scaler_mean.npy", scaler.mean_)
    np.save("scaler_scale.npy", scaler.scale_)
    print("Model + scaler saved!")


def calculate_beta_from_nn(lat, lon, date_str, info):
    """
    Predicts beta given input features:
    features = [temperature, humidity, precipitation, distance_inland, distance_reservoir, percent_wetland]
    """
    # Load model
    model = load_model("nn_beta_predictor.keras")
    scaler.mean_ = np.load("scaler_mean.npy")
    scaler.scale_ = np.load("scaler_scale.npy")

    # extract features
    climate = info["climate"]["outbreak_day"]
    X = [[
        climate["2m_temperature"],
        climate["2m_relative_humidity"],
        climate["precipitation_flux"],
        info.get("distance_to_water", 0),
        info.get("distance_inland", 0),
        info.get("scaled_abundance", 0),
    ]]
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)
    return float(prediction[0][0])


if __name__ == "__main__":
    # Run training if script is executed directly
    train_nn()
