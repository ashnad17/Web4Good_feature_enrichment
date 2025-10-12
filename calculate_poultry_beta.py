import pandas as pd
import numpy as np
import json
from datetime import datetime

# Load filtered CSV
df = pd.read_csv("poultry_outbreaks_usa_2022.csv")

results = {}

for _, row in df.iterrows():
    cases = row.get("cases")
    dead = row.get("dead")
    killed = row.get("killed_disposed")
    susceptible = row.get("susceptible")
    species = row.get("Species")
    state = row.get("level1_name")
    county = (str(row.get("Location_name")))
    lat = row.get("Latitude")
    lon = row.get("Longitude")
    start = row.get("Outbreak_start_date")
    print(start)
    end = row.get("Outbreak_end_date")

    # Parse outbreak duration
    try:
        duration = (pd.to_datetime(end) - pd.to_datetime(start)).days
        if duration <= 0:
            duration = 1
    except:
        duration = 1

    # Determine abundance (priority: susceptible, else cases+dead+killed)
    if pd.notna(susceptible) and susceptible > 0:
        abundance = susceptible
    elif any(pd.notna(x) for x in [cases, dead, killed]):
        abundance = sum([x for x in [cases, dead, killed] if pd.notna(x)])
    else:
        continue  # skip if nothing usable

    # Determine beta (cases preferred → dead → susceptible)
    if pd.notna(cases) and cases > 0:
        beta = (cases / duration) / abundance
    elif pd.notna(dead) and dead > 0:
        beta = (dead / duration) / abundance
    elif pd.notna(susceptible) and susceptible > 0:
        beta = (susceptible / duration) / abundance
    else:
        continue

    # Store clean date
    try:
        date_obj = pd.to_datetime(start)
        date_str = date_obj.strftime("%d/%m/%Y")
    except:
        date_str = str(start)

    # Build entry key
    key = f"WAHIS/Domestic/{species}/{date_str}/{state}/{county}/{lat}/{lon}"

    # Store results
    results[key] = {
        "county": county,
        "state": state,
        "date": date_str,
        "total_abundance": abundance,  # raw abundance before scaling
        "species": {},
        "beta": beta,
        "latitude": lat,
        "longitude": lon
    }

# Convert to DataFrame for scaling
if results:
    abundances = [v["total_abundance"] for v in results.values()]
    min_abund, max_abund = min(abundances), max(abundances)

    # Scale abundance for all entries
    for v in results.values():
        v["scaled_abundance"] = (
            (v["total_abundance"] - min_abund) / (max_abund - min_abund)
            if max_abund > min_abund else 0
        )

# Save to JSON
with open("poultry_beta_scaled.json", "w") as f:
    json.dump(results, f, indent=2)

print("Saved results → poultry_beta_scaled.json")
