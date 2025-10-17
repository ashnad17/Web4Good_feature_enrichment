from datetime import datetime
import json
import os
import numpy as np
import sys
from typing import Dict, Tuple
import pycatch22 as catch22 # type: ignore

from sklearn.preprocessing import MinMaxScaler
from transmission_model.get_env_inputs import get_environmental_inputs
from transmission_model.h5n1_beta_modulation import calculate_beta_with_regime
from transmission_model.linear_model import calculate_beta_linear_from_model, train_model
from transmission_model.neural_network import train_nn, calculate_beta_from_nn

MODEL = ""
CATCH_22 = True

def simulate_sei(S0: float, E0: float, I0: float, beta: float, sigma: float, days: int, N: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SEI simulation for specified days."""
    S, E, I = [S0], [E0], [I0]

    for _ in range(days):
        St, Et, It = S[-1], E[-1], I[-1]
        lambda_t = beta * It / N if N > 0 else 0

        new_exposed = min(lambda_t * St, St)
        new_infectious = min(sigma * Et, Et)

        S_next = St - new_exposed
        E_next = Et + new_exposed - new_infectious
        I_next = It + new_infectious

        # Keep total within population limits
        total = S_next + E_next + I_next
        if total > N:
            excess = total - N
            S_next = max(S_next - excess, 0)
            total = S_next + E_next + I_next
            if total > N:
                excess = total - N
                E_next = max(E_next - excess, 0)
                total = S_next + E_next + I_next
                if total > N:
                    excess = total - N
                    I_next = max(I_next - excess, 0)

        S.append(S_next)
        E.append(E_next)
        I.append(I_next)

    return np.array(S), np.array(E), np.array(I)

def run_sei_simulation(data: Dict, sigma: float, days_back: int, days_forward: int) -> Dict:
    """Run SEI model per outbreak site using backward seeding."""
    results = {}

    for outbreak_id, info in data.items():
        print(outbreak_id)
        try:
            N = float(info.get("total_abundance", 0))
            if N < 2:
                print(f"[SKIP] {outbreak_id} — insufficient host population.")
                continue
            lat = info["latitude"]
            lon = info["longitude"]
            date_str = info["date"]

            if MODEL == "wms":
                beta = calculate_beta_with_regime(date_str, info)
            if MODEL == "linear":
                beta = calculate_beta_linear_from_model(lat, lon, date_str, info, CATCH_22)
            if MODEL == "nn":
                beta = calculate_beta_from_nn(lat, lon, date_str, info)


            # BACKWARD SEEDING
            S0 = N - 1  # 1 infected bird seeded days_back before detection
            E0 = 0
            I0 = 1

            # Simulate from t = -days_back to detection
            S, E, I = simulate_sei(S0, E0, I0, beta, sigma, days_back, N)
            S_det, E_det, I_det = S[-1], E[-1], I[-1]

            # Optionally simulate forward after detection
            S_future, E_future, I_future = simulate_sei(S_det, E_det, I_det, beta, sigma, days_forward, N)

            results[outbreak_id] = {
                "metadata": {
                    "date": date_str,
                    "lat": lat,
                    "lon": lon,
                    "beta": beta,
                    "total_abundance": N,
                    "days_back": days_back,
                    "days_forward": days_forward
                },
                "SEI_back_to_detection": {
                    "S": S.tolist(),
                    "E": E.tolist(),
                    "I": I.tolist()
                },
                "SEI_post_detection": {
                    "S": S_future.tolist(),
                    "E": E_future.tolist(),
                    "I": I_future.tolist()
                }
            }

        except Exception as e:
            print(f"[ERROR] {outbreak_id}: {e}")

    return results

def search_outbreak_features(outbreak_data):
    '''iterate through each outbreak
    check if data is store cfor that feature
    call find env features for each
    scale abundance and return outbreak data'''
    filename ="WAHIS_outbreaks_features.json"
    try:
        with open(filename, "r") as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        existing_data = {}
    abundances = []
    for key, details in outbreak_data.items():
        abundance = details.get("total_abundance")
        if abundance is not None:
            abundances.append(abundance)
    if abundances:
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(np.array(abundances).reshape(-1, 1)).flatten()
        scaled_map = {k: v for k, v in zip(outbreak_data.keys(), scaled_values)}
    else:
        scaled_map = {}
        
    for key, details in outbreak_data.items():
        if key in existing_data:
            continue

        print(f"Key: {key}")
        county = details.get("county")
        state = details.get("state")
        date_str = details.get('date')
        date_str = details.get("date")
        beta = details.get("beta")
        abundance = details.get("total_abundance")
        lat = details.get("latitude")
        lon = details.get("longitude")

        if isinstance(date_str, str):
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            date = date_str
            
        result, temp_C, rh_percent, precip_mm, dist_to_reservoir_km = get_environmental_inputs(key)

        scaled_abundance = scaled_map.get(key, 0)


        # Store all info
        existing_data[key] = {
            "county": county,
            "state": state,
            "date": str(date),
            "latitude": lat,
            "longitude": lon,
            "scaled_abundance": scaled_abundance,
            "total_abundance": abundance,
            "beta": details.get("beta", 0),
            "climate": result.get("climate", {}),
            "distance_inland": result.get("distance_inland"),
            "distance_to_water": result.get("distance_to_water")
        }


        with open(filename, "w") as f:
            json.dump(existing_data, f, indent=2)
    
    return existing_data

def safe_catch22(ts):
    if np.all(np.isclose(ts, ts[0])):  # constant or all same
        return np.zeros(22)  # or np.full(22, np.nan)
    feats = np.array(catch22.catch22_all(ts)['values'])
    feats[np.isnan(feats)] = 0
    feats[np.isinf(feats)] = 0
    return feats

def add_time_series(outbreak_data):


    for key in outbreak_data:
        climate = outbreak_data[key]['climate']
        sorted_days = sorted(climate.keys(), key=lambda x: int(x.split('-')[-1]) if '-' in x else -1)
        sorted_days.reverse()
        temp_series = [climate[d]["2m_temperature"] for d in sorted_days if "2m_temperature" in climate[d]]
        humidity_series = [climate[d]["2m_relative_humidity"] for d in sorted_days if "2m_relative_humidity" in climate[d]]
        precip_series = [climate[d]["precipitation_flux"] for d in sorted_days if "precipitation_flux" in climate[d]]
        if not CATCH_22:
            temp_series, humidity_series, precip_series = [], [], []

        catch22_feats = {}
        if CATCH_22 and len(temp_series) > 0:
            temp_feats = safe_catch22(temp_series)
            humidity_feats = safe_catch22(humidity_series)
            precip_feats = safe_catch22(precip_series)


            if MODEL != "wms":
                combined_feats = np.concatenate([temp_feats, humidity_feats, precip_feats])
                catch22_feats = {"catch22_features": combined_feats.tolist()}
            else:
                weights = np.array([1.0, 1.0, 1.0])
                weighted_feats = (temp_feats * weights[0] +
                                  humidity_feats * weights[1] +
                                  precip_feats * weights[2]) / sum(weights)
                catch22_feats = {"catch22_features": weighted_feats.tolist()}

        # Add the features to the outbreak data
        outbreak_data[key]['catch22'] = catch22_feats
    return outbreak_data
        
if __name__ == "__main__":

    INPUT_JSON = "transmission_model/data/wahis_abundance_with_beta_sample.json"
    # INPUT_JSON = "transmission_model/data/wahis_abundance_with_beta.json"
    # INPUT_JSON = "sei_outputs_sorted.json"
    
    SIGMA = 1 / 2.5
    DAYS_BACK = 7  # Days before detection to seed 1 infection
    DAYS_FORWARD = 10  # Days after detection to simulate

    ## add ifs here to check which type of model is running
    len_atleast_one = True
    if len(sys.argv) == 1:
        print("no model provided. quitting")
        print("no catch 22 specification provided. quitting")
        quit()
    elif len(sys.argv) == 2:
        print("incorrect arguememts provided, quitting")
        quit()
    elif sys.argv[1] == "linear":
        MODEL = "linear"
        OUTPUT_JSON = "transmission_model/output_json/sei_linear_model_outputs_sample.json"
        if sys.argv[2].lower() == "false":
            print("not using catch 22")
            CATCH_22 = False
        elif sys.argv[2].lower() == "true":
            print("now using catch 22")
            CATCH_22 = True
        train_model(CATCH_22)
    elif sys.argv[1] == "neural":
        MODEL = "nn"
        OUTPUT_JSON = "transmission_model/output_json/sei_nn_outputs_sample.json"
        if sys.argv[2].lower() == "false":
            print("not using catch 22")
            CATCH_22 = False
        elif sys.argv[2].lower() == "true":
            print("now using catch 22")
            CATCH_22 = True
        train_nn()
    elif sys.argv[1] == "wms":
        MODEL = "wms"
        OUTPUT_JSON = "transmission_model/output_json/sei_backward_seeding_outputs_sample.json"
        if sys.argv[2].lower() == "false":
            print("not using catch 22")
            CATCH_22 = False
        elif sys.argv[2].lower() == "true":
            print("now using catch 22")
            CATCH_22 = True
    else:
        model = "wms"
        print("wrong model provided. defaulting to weighted multiplicative scaling")
        OUTPUT_JSON = "transmission_model/output_json/sei_backward_seeding_outputs_sample.json"



    with open(INPUT_JSON, "r") as f:
        outbreak_data = json.load(f)

    outbreak_data = search_outbreak_features(outbreak_data)

    outbreak_data = add_time_series(outbreak_data)

    simulation_results = run_sei_simulation(
        data=outbreak_data,
        sigma=SIGMA,
        days_back=DAYS_BACK,
        days_forward=DAYS_FORWARD
    )
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(simulation_results, f, indent=2)

    print(f"Simulated SEI dynamics for {len(simulation_results)} outbreak sites.")
