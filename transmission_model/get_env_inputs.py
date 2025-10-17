# get_env_inputs.py
from data_retrieval.environmental_data import begin_data_extraction
import json
import os


def get_environmental_inputs(query):
    result = begin_data_extraction(query)
    

    temp_C = float(result['climate']['outbreak_day']['2m_temperature'])
    rh_percent = float(result['climate']['outbreak_day']['2m_relative_humidity'])
    precip_mm = float(result['climate']['outbreak_day']['precipitation_flux'])
    dist_to_reservoir_km = float(result['distance_to_water'])
    return result, temp_C, rh_percent, precip_mm, dist_to_reservoir_km

# def write_to_file(result, filename):
#     # Load existing results if the file exists
#     if i > 0:
#         with open(filename, "r") as f:
#             try:
#                 existing_data = json.load(f)
#             except json.JSONDecodeError:
#                 existing_data = []  
#     else:
#         existing_data = []

#     # Index existing data by (Location, Day 1)
#     existing_data.append(result)

#     # Write updated data back to file
#     with open(filename, "w") as f:
#         json.dump(existing_data, f, indent=2)


