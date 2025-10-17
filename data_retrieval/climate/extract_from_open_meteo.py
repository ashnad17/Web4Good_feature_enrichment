from datetime import datetime, timedelta
import pandas as pd
import requests_cache
from retry_requests import retry
import openmeteo_requests

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

def get_weather_data(lat, lon, outbreak_date):
    outbreak_dt = pd.to_datetime(outbreak_date, dayfirst=True).date()
    start_dt = outbreak_dt - timedelta(days=13)  # 14 days including outbreak day

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": outbreak_dt.strftime("%Y-%m-%d"),
        "hourly": ["temperature_2m", "relative_humidity_2m", "precipitation"],
        "timezone": "auto"
    }

    responses = openmeteo.weather_api(url, params=params) 
    response = responses[0] 
    
    # Extract hourly data 
    hourly = response.Hourly() 
    
    times = pd.to_datetime(hourly.Time(), unit="s", utc=True) 
    times_end = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True) 
    
    interval_seconds = hourly.Interval() 
    df_hourly = pd.DataFrame({
        "time": pd.date_range(start=times, end=times_end, freq=pd.Timedelta(seconds=interval_seconds), inclusive="left"), 
        "temperature_2m": hourly.Variables(0).ValuesAsNumpy(), 
        "relative_humidity_2m": hourly.Variables(1).ValuesAsNumpy(), 
        "precipitation": hourly.Variables(2).ValuesAsNumpy() 
        }) 
    
    # Compute daily 24-hour means 
    df_hourly["date"] = df_hourly["time"].dt.date 
    df_daily = df_hourly.groupby("date").agg({
        "temperature_2m": "mean", 
        "relative_humidity_2m": "mean", 
        "precipitation": "mean" }).reset_index() 
    
    # Build results dictionary 
    result = { "climate": {} } 
    
    # Add daily climate data as labeled dictionary: day1…day14, outbreak_day 
    daily_dates = list(df_daily["date"]) 
    
    for i, row in enumerate(df_daily.itertuples(index=False)):
        label = f"outbreak_day-{13 - i}" if i < len(df_daily[1:])-1 else "outbreak_day" 
        result["climate"][label] = {
            "2m_temperature": float(row.temperature_2m),
            "2m_relative_humidity": float(row.relative_humidity_2m),
            "precipitation_flux": float(row.precipitation)
            } 
    return result

def extract_climate(query):
    # Parse the query

    parts = query.split('/')
    state = parts[6]
    county = parts[7]
    date_str = f"{parts[3]}/{parts[4]}/{parts[5]}"  # YYYY/MM/DD
    lat_query = float(parts[8])
    lon_query = float(parts[9])

    print(f"Fetching 14-day weather history up to {date_str} near ({lat_query}, {lon_query})")
    weather_data = get_weather_data(lat_query, lon_query, date_str)
    return weather_data
