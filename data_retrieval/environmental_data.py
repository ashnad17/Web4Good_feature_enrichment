import re
import json
import os
from pathlib import Path

from data_retrieval.climate import extract_from_open_meteo
from data_retrieval.reservoir import extract_reservoir_data
from data_retrieval.distance_inland import extract_coastline_data

def begin_data_extraction(query):
    

    # extract data for temp, humidity, precipitation
    result = extract_from_open_meteo.extract_climate(query)

    # extract distance inland for point
    result = extract_coastline_data.extract_inland_distance(query, result)

    # change to whichever one is needed:
    result = extract_reservoir_data.extract_reservoir_distance(query, result)

    result = extract_reservoir_data.extract_reservoir_percentage(query, result)

    return result


    # extract_cds_data('2023')