import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import urllib3
import pandas as pd
import os

# Disable warnings for insecure requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure retry strategy
def setup_session():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

# Function to download weekly raster data
def download_weekly_data(speciesCode, month, day, output_folder):
    # Create a directory within the output folder for the date if it doesn't exist
    date_directory = os.path.join(output_folder, f'{day}-{month}')
    os.makedirs(date_directory, exist_ok=True)

    # Define the file path
    file_path = os.path.join(date_directory, f'{speciesCode}_abundance_median_2022-{month}-{day}.tif')

    # Skip download if the file already exists
    if os.path.exists(file_path):
        print(f"File {file_path} already exists. Skipping download.")
        return

    # Define the URL for the raster file
    url = (f"https://st-download.ebird.org/v1/fetch?"
           f"objKey=2022/{speciesCode}/web_download/weekly/{speciesCode}_abundance_median_2022-{month}-{day}.tif"
           f"&key=")

    # Attempt to download the file
    session = setup_session()
    response = session.get(url, stream=True, verify=False)

    if response.status_code == 200:
        # Save the file to disk
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {file_path}")
    else:
        print(f"Failed to download {file_path}. HTTP status code: {response.status_code}")

# Main script
if __name__ == "__main__":
    # Define input and output paths
    input_folder = "data_retrieval/ebird/input"
    output_folder = "data_retrieval/ebird/output"

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load species data and time variables
    species_file = os.path.join(input_folder, 'output_allbirdsUS.xlsx')
    time_file = os.path.join(input_folder, 'Time_variables.xlsx')

    df_species = pd.read_excel(species_file)
    df_time = pd.read_excel(time_file)

    # Iterate over each species and time combination to download raster data
    for _, row_species in df_species.iterrows():
        speciesCode = row_species['speciesCode']
        for _, row_time in df_time.iterrows():
            month = row_time['month']
            day = row_time['day']
            download_weekly_data(speciesCode, month, day, output_folder)

    print("All downloads completed.")
