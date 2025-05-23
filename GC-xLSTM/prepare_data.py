import numpy as np
import pandas as pd
from typing import Tuple
from synthetic import simulate_lorenz_96, simulate_var
from datasets.acatis.prepare_acatis_data import fetch_acatis_data

def get_fmri_data(timeseries_file: str, gc_file: str):
    # Read the time series data from the CSV file
    time_series = pd.read_csv(timeseries_file, header=0).to_numpy()

    # Print the shape of the time series array
    print(f"Shape of time series array: {time_series.shape}")

    # Read the Granger causality data from the CSV file, the format being 
    # [cause, effect, lag]. there is no header in the file
    gc_data = pd.read_csv(gc_file, header=None).to_numpy()
    # Print the shape of the Granger causality data
    print(f"Shape of Granger causality data: {gc_data.shape}")
    # convert the gc data to p x p boolean matrix, with rows representing effect and columns representing cause
    p = time_series.shape[1]
    gc_matrix = np.zeros((p, p), dtype=int)
    for i in range(gc_data.shape[0]):
        cause = int(gc_data[i, 0])
        effect = int(gc_data[i, 1])
        lag = int(gc_data[i, 2])
        gc_matrix[effect, cause] = 1
    
    return time_series, gc_matrix

def fetch_molene_data(field: str, num_stations: int | str = 'all') -> Tuple[np.ndarray, list]:
    # Load the data from a CSV file
    df = pd.read_csv("/home/harsh/xlstm/Neural-GC/datasets/molene/Original_Data/aggregated_data.csv")

    # Load the weather stations data
    weather_stations_df = pd.read_csv("/home/harsh/xlstm/Neural-GC/datasets/molene/Original_Data/weather_stations.csv")
    valid_stations = weather_stations_df['numer_sta'].unique()

    # Filter rows where temperature (t) is not missing (mq) and station is valid
    filtered_df = df[(df[field] != 'mq') & (df['numer_sta'].isin(valid_stations))]

    # Calculate counts per station for non-missing temperature data
    station_counts = filtered_df.groupby('numer_sta').size()

    # Find the maximum count of temperature readings
    max_count = station_counts.max()

    # Retain only stations with counts equal to the maximum
    stations_with_max_counts = station_counts[station_counts == max_count].index
    filtered_df = filtered_df[filtered_df['numer_sta'].isin(stations_with_max_counts)]

    # Ensure no NaNs in the temperature field
    filtered_df = filtered_df.dropna(subset=[field])

    # Sort by station and date for proper time series alignment
    filtered_df = filtered_df.sort_values(by=['numer_sta', 'date'])

    # Create a pivot table to align time series for each station
    pivot_df = filtered_df.pivot(index='date', columns='numer_sta', values=field)
    
    # Randomly select num_stations stations
    if num_stations != 'all':
        if num_stations > len(pivot_df.columns):
            raise ValueError(f"Number of stations requested ({num_stations}) exceeds available stations ({len(pivot_df.columns)})")
        pivot_df = pivot_df.sample(n=num_stations, axis=1)

    # Convert to a NumPy array of shape (T, p)
    time_series = pivot_df.to_numpy().astype(float)

    # Normalize the time series along each row using min-max scaling
    min_vals = np.min(time_series, axis=1, keepdims=True)
    max_vals = np.max(time_series, axis=1, keepdims=True)
    time_series = (time_series - min_vals) / (max_vals - min_vals)
    if (max_vals == min_vals).any():
        print("Warning: Some fields have constant values")
        print(time_series)
        exit(0)

    # Get the list of stations
    stations = pivot_df.columns.tolist()

    # Print the resulting array shape
    print(f"Shape of time series array: {time_series.shape}")

    return time_series, stations


def return_data(dataset: dict) -> Tuple[np.ndarray, ...]:
    if "molene" in dataset["name"]:
        fields = dataset['dataset_config']['field']
        num_stations = dataset['dataset_config']['num_stations']
        data, stations = fetch_molene_data(fields, num_stations)
        return data, None, stations
    elif "lorenz" in dataset['name']:
        p = dataset['dataset_config']['p']
        T = dataset['dataset_config']['T']
        F = dataset['dataset_config']['F']
        return simulate_lorenz_96(p=p, F=F, T=T)
    elif "var" in dataset["name"]:
        p = dataset['dataset_config']['p']
        T = dataset['dataset_config']['T']
        lag = dataset['dataset_config']['lag']
        sparsity = dataset['dataset_config']['sparsity']
        return simulate_var(p=p, T=T, lag=lag, sparsity=sparsity)
    elif "mocap" in dataset["name"]:
        dataset_path = dataset["dataset_config"]["dataset_path"]
        time_series = np.load(dataset_path)
        time_series_data = time_series['time_series_data']
        joint_info = time_series['joint_info']
        return time_series_data, None, joint_info
    elif "fmri" in dataset["name"]:
        timeseries_file = dataset["dataset_config"]["timeseries_file"]
        gc_file = dataset["dataset_config"]["gc_file"]
        time_series, gc_matrix = get_fmri_data(timeseries_file, gc_file)
        return time_series, gc_matrix, None
    elif "acatis" in dataset["name"]:
        acatis_time_series, gc_matrix, all_feature_names = fetch_acatis_data(subset=False)
        # time series shape is (num_companies, time_steps, num_features)
        return acatis_time_series, gc_matrix, all_feature_names
    else:
        raise ValueError(f"Unsupported dataset name: {dataset['name']}")

def write_to_csv(data: np.ndarray, file_path: str) -> None:
    """
    Write the given data to a CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    print(f"Data written to {file_path}")