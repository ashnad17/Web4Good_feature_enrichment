# Avian Influenza SEI Model — README

## Overview

This repository provides a complete workflow for modelling Highly Pathogenic Avian Influenza (HPAI) outbreaks in poultry populations using an SEI (Susceptible–Exposed–Infectious) framework.  
It includes data preprocessing, parameter computation, feature extraction, and model execution.

---

## **1. Data Source**

- Data is downloaded from the **WAHIS private repository**.
- The main input file is:

Compute the transmission parameter β using the equation:
β
=
(
cases
/
duration
)
abundance
β=
abundance
(cases/duration)
​

Scale the abundance values appropriately.
The resulting dataset (with a new column for β and scaled abundance) is saved as:
poultry_beta_scaled.json

3. Running the SEI Model
   Install Dependencies
   Before running the model, install all required dependencies:
   pip install -r requirements.txt
   Run the Model
   Use the following command to execute the SEI model:
   python3 -m transmission_model.seimodel MODEL_NAME C22
   Where:
   MODEL_NAME can be:
   linear
   neural
   wms
   or None (defaults to wms)
   where C22 is true or false, saying do you want to run with catch 22 or not
   Example:
   python3 -m transmission_model.seimodel neural
4. Automatic Feature Collection
   You do not need to manually collect environmental features.
   The SEI model automatically retrieves and stores data for each outbreak, including:
   Temperature
   Relative humidity
   Precipitation
   Distance inland
   Distance to nearest water reservoir
   Output Files
   For poultry used in training:
   poultry_outbreak_features.json
   For wild birds used in testing:
   wild_birds_features.json
5. Notes
   Ensure that the WAHIS data is correctly formatted before filtering.
   All scripts assume latitude and longitude columns are available for environmental data retrieval.
   Cached API calls are stored locally in the .cache directory to reduce redundant requests.
6. File Summary
   File Name Description
   infur_20251006 Raw WAHIS outbreak data
   filter_poultry_data.py Filters poultry data for HPAI (USA, 2022)
   poultry_beta_scaled.json Output file with scaled abundance and β values
   transmission_model/seimodel.py Core SEI model script
   poultry_outbreak_features.json Automatically generated features for poultry
   wild_birds_features.json Automatically generated features for wild bird testing
7. Example Workflow

# 1. Filter raw outbreak data

python3 filter_poultry_data.py

# 2. Run SEI model (example: neural version)

python3 -m transmission_model.seimodel neural


USE plot_beta.py to plot how different teh betas are to each other in a scatter plot