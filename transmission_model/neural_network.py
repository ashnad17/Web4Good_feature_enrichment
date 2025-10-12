# train_nn.py
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

scaler = StandardScaler()  # global so training + prediction use same scaler


# with open("results_2022.json", "r") as f:
#     data = json.load(f)


# with open("transmission_model/output_json/sei_backward_seeding_outputs_sample_augmented.json", "r") as f:
#     sei_data = json.load(f)
    
def train_nn():
    # Load data
    with open("results_2022.json", "r") as f:
        data = json.load(f)


    with open("transmission_model/output_json/sei_backward_seeding_outputs_sample_augmented.json", "r") as f:
        sei_data = json.load(f)

    # Extract inputs and outputs
    X, y = [], []
    for result in data:
        for uuid, sei_entry in sei_data.items():
            uuid_parts = uuid.split('/')
            if uuid_parts[6].isalpha() or uuid_parts[7].isalpha():
                uuid = uuid_parts[8] + '/' + uuid_parts[9] + '/' + uuid_parts[5] + '-' + uuid_parts[3] + '-' + uuid_parts[4]
            else:
                uuid = uuid_parts[6] + '/' + uuid_parts[7] + '/' + uuid_parts[5] + '-' + uuid_parts[3] + '-' + uuid_parts[4]

            if uuid == result["uuid"]:
                beta = sei_entry["metadata"]["beta"]
        entry = result["climate"]["outbreak_day"]
        features = [
            entry["2m_temperature"],
            entry["2m_relative_humidity"],
            entry["precipitation_flux"],
            result.get("distance_inland", 0),
            result.get("distance_to_water", 0),
            result.get("percentage_wetland", 0),
        ]
        X.append(features)
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

    # Load scaler params
    scaler_mean = np.load("scaler_mean.npy")
    scaler_scale = np.load("scaler_scale.npy")
    scaler = StandardScaler()
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale

    # Transform features

    for result in data:
        for uuid, sei_entry in sei_data.items():
            uuid_parts = uuid.split('/')
            if uuid_parts[6].isalpha() or uuid_parts[7].isalpha():
                uuid = uuid_parts[8] + '/' + uuid_parts[9] + '/' + uuid_parts[5] + '-' + uuid_parts[3] + '-' + uuid_parts[4]
            else:
                uuid = uuid_parts[6] + '/' + uuid_parts[7] + '/' + uuid_parts[5] + '-' + uuid_parts[3] + '-' + uuid_parts[4]

        entry = result["climate"]["outbreak_day"]
        features = [
            entry["2m_temperature"],
            entry["2m_relative_humidity"],
            entry["precipitation_flux"],
            result.get("distance_inland", 0),
            result.get("distance_to_water", 0),
            result.get("percentage_wetland", 0),
        ]

    X = np.array([features], dtype=float)
    X_scaled = scaler.transform(X)

    # Predict
    prediction = model.predict(X_scaled)
    return float(prediction[0][0])


if __name__ == "__main__":
    # Run training if script is executed directly
    train_nn()
