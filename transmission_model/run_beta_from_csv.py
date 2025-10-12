# run_beta_from_csv.py

import pandas as pd
from transmission_model.h5n1_beta_modulation import calculate_beta_with_regime

#Data file import
df = pd.read_json("bird_wahis_outbreaks_2022.json").T.reset_index()
df.columns = ['key', 'state', 'county', 'latitude', 'longitude'] 

key_parts = df['key'].str.split('/', expand = True)

df['day'] = key_parts[3]
df['month'] = key_parts[4]
df['year'] = key_parts[5]

# Ensure dates are parsed
df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
df = df[['date', 'latitude', 'longitude']]

# Calculate β for each row
beta_list = []
for _, row in df.iterrows():
    lat = row['latitude']
    lon = row['longitude']
    date_str = row['date'].strftime('%Y-%m-%d')
    beta = calculate_beta_with_regime(lat, lon, date_str)
    beta_list.append(beta)

# Add β to the dataframe
df['beta'] = beta_list

print(f" beta value: {beta_list}")
# Output
df.to_csv("cases_with_beta.csv", index=False)

 