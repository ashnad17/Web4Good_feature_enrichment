import json
import os
import numpy as np
import sys
from typing import Dict, Tuple
from transmission_model.h5n1_beta_modulation import calculate_beta_with_regime
from transmission_model.linear_model import calculate_beta_linear_from_model, train_model
from transmission_model.neural_network import train_nn, calculate_beta_from_nn

MODEL = ""

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
        try:
            if MODEL == "linear" or MODEL == "nn":
                if info['metadata'] != None:
                    info = info['metadata']

            N = float(info.get("total_abundance", 0))
            if N < 2:
                print(f"[SKIP] {outbreak_id} — insufficient host population.")
                continue
            try:
                lat = info["latitude"]
                lon = info["longitude"]
                date_str = info["date"]
            except:
                lat = info["lat"]
                lon = info["lon"]
                date_str = info["date"]

            if MODEL == "wms":
                beta = calculate_beta_with_regime(lat, lon, date_str, N)
            if MODEL == "linear":
                beta = calculate_beta_linear_from_model(lat, lon, date_str, info)

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

if __name__ == "__main__":

    INPUT_JSON = "transmission_model/data/wahis_abundance_with_beta_sample.json"
    # INPUT_JSON = "transmission_model/data/wahis_abundance_with_beta.json"
    # INPUT_JSON = "sei_outputs_sorted.json"
    
    SIGMA = 1 / 2.5
    DAYS_BACK = 7  # Days before detection to seed 1 infection
    DAYS_FORWARD = 10  # Days after detection to simulate

    ## add ifs here to check which type of model is running
    if len(sys.argv) == 1:
        MODEL = "wms"
        print("no model provided. defaulting to weighted multiplicative scaling")
        OUTPUT_JSON = "transmission_model/output_json/sei_backward_seeding_outputs_sample.json"
    elif sys.argv[1] == "linear":
        MODEL = "linear"
        OUTPUT_JSON = "transmission_model/output_json/sei_linear_model_outputs_sample.json"
        train_model("poultry_training.csv")
    elif sys.argv[1] == "neural":
        MODEL = "nn"
        OUTPUT_JSON = "transmission_model/output_json/sei_nn_outputs_sample.json"
        train_nn()
    elif sys.argv[1] == "wms":
        MODEL = "nn"
        OUTPUT_JSON = "transmission_model/output_json/sei_nn_outputs_sample.json"
        train_nn()
    else:
        model = "wms"
        print("wrong model provided. defaulting to weighted multiplicative scaling")
        OUTPUT_JSON = "transmission_model/output_json/sei_backward_seeding_outputs_sample.json"

    with open(INPUT_JSON, "r") as f:
        outbreak_data = json.load(f)

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
