import pandas as pd

# Load CSV
df = pd.read_csv("infur_20251006.csv", dtype=str)

# Convert event_start date to datetime to filter by year
df["event_start date"] = pd.to_datetime(df["event_start date"], errors="coerce")

# Apply filters
filtered_df = df[
    (df["disease_eng"] == "High pathogenicity avian influenza viruses (poultry) (Inf. with)") &
    (df["country"] == "United States of America") &
    (df["event_start date"].dt.year == 2022)
]

# Save filtered results
filtered_df.to_csv("poultry_outbreaks_usa_2022.csv", index=False)

print(f"Filtered {len(filtered_df)} rows saved to poultry_outbreaks_usa_2022.csv")
