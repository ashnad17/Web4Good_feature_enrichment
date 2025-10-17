import json
import matplotlib.pyplot as plt
import os
import pandas as pd

nn_betas = "transmission_model/output_json/sei_nn_outputs_sample.json"
linear_beta = "transmission_model/output_json/sei_linear_model_outputs_sample.json"
wms_betas = "transmission_model/output_json/sei_backward_seeding_outputs_sample.json"




if os.path.getsize(nn_betas) == 0 or os.path.getsize(linear_beta) == 0 or os.path.getsize(wms_betas) == 0:
    print("one or more files empty, cannot compare")
    quit()

with open(nn_betas, "r") as f:
    nn_data = json.load(f)
with open(linear_beta, "r") as f:
    linear_data = json.load(f)
with open(wms_betas, "r") as f:
    wms_data = json.load(f)

# --- Extract (lat, lon) and beta values ---
def extract_betas(data):
    coords = []
    betas = []
    for key, value in data.items():
        meta = value.get("metadata", {})
        lat = meta.get("lat")
        lon = meta.get("lon")
        beta = meta.get("beta")
        if lat is not None and lon is not None and beta is not None:
            coords.append(f"({lat:.2f}, {lon:.2f})")
            betas.append(beta)
    return coords, betas

coords1, beta_method1 = extract_betas(nn_data)
coords2, beta_method2 = extract_betas(linear_data)
coords3, beta_method3 = extract_betas(wms_data)

# --- Align by coordinate order (using NN’s as reference) ---
# (Assumes all three have the same coordinate keys)
coords = coords1
x = range(len(coords)) 

plt.figure(figsize=(14, 6))
plt.scatter(x, beta_method1, label='NN β', alpha=0.8, marker='o')
plt.scatter(x, beta_method2, label='Linear Model β', alpha=0.8, marker='s')
plt.scatter(x, beta_method3, label='Backward Seeding β', alpha=0.8, marker='^')

plt.legend()
plt.xlabel("Latitude, Longitude")
plt.ylabel("Predicted β")
plt.title("Predicted β Comparison Across Locations")
plt.tight_layout()
plt.show()