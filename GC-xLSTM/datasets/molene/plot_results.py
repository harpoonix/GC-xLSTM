import pandas as pd
import numpy as np
import re
import folium
from folium.plugins import FloatImage, PolyLineTextPath
from typing import List
from prepare_data import return_data

def dms_to_decimal(dms_str):
    """Convert DMS coordinate string to decimal degrees"""
    match = re.match(
        r"""([+-]?\d+\.?\d*)°        # Degrees
                         (\d+\.?\d*)'            # Minutes
                         (\d+\.?\d*)"?            # Seconds (optional)
                         ([NSEW])                 # Direction""",
        dms_str,
        re.VERBOSE,
    )
    if not match:
        raise ValueError(f"Invalid DMS format: {dms_str}")
    degrees, minutes, seconds, direction = match.groups()
    seconds = seconds or 0  # Handle missing seconds
    dd = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    if direction in ["S", "W"]:
        dd *= -1
    return dd

def create_weather_map(station_ids: List, adj_matrix: np.ndarray, output_file="weather_map.html", display_empty_stations=True):
    """
    Create a weather station map with directed connections
    
    Parameters:
    - station_ids: list of numer_sta values to include
    - adj_matrix: numpy array adjacency matrix (n x n where n = len(station_ids))
    - output_file: path for saving the HTML map
    """
    # Load and process data
    df = pd.read_csv("GC-xLSTMdatasets/molene/Original_Data/weather_stations.csv")
    df["Latitude"] = df["Latitude"].apply(dms_to_decimal)
    df["Longitude"] = df["Longitude"].apply(dms_to_decimal)

    # Filter selected stations
    selected_df = df[df['numer_sta'].isin(station_ids)].reset_index(drop=True)
    
    if len(selected_df) == 0:
        raise ValueError("No stations found with the provided IDs")
        
    if selected_df.shape[0] != adj_matrix.shape[0] or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError(f"Adjacency matrix must be square matrix of size {len(selected_df)}x{len(selected_df)}")

    # Create the map
    m = folium.Map(
        location=[selected_df["Latitude"].mean(), selected_df["Longitude"].mean()],
        zoom_start=8,
    )
    folium.TileLayer("Stamen Terrain", attr="Map").add_to(m)
    
    has_edges = np.zeros(adj_matrix.shape[0])

    # Add connections
    for i in range(adj_matrix.shape[0]):
        for j in range(adj_matrix.shape[1]):
            if adj_matrix[i][j] == 1 and i != j:
                has_edges[i] = 1
                has_edges[j] = 1
                points = [
                    [selected_df.iloc[i]["Latitude"], selected_df.iloc[i]["Longitude"]],
                    [selected_df.iloc[j]["Latitude"], selected_df.iloc[j]["Longitude"]]
                ]
                line = folium.PolyLine(points, color="blue", weight=2, opacity=0.8).add_to(m)
                
                PolyLineTextPath(
                    line,
                    "➤",
                    repeat=False,
                    center=True,
                    offset=0,
                    attributes={
                        "fill": "#0000ff",
                        "font-weight": "bold",
                        "font-size": "24",
                        "alignment-baseline": "middle",
                        "dominant-baseline": "middle",
                    },
                ).add_to(m)
    
    # Add stations
    for idx, row in selected_df.iterrows():
        if display_empty_stations or has_edges[idx] == 1:
            folium.Marker(
                location=[row["Latitude"], row["Longitude"]],
                popup=f"<b>{row['Nom']}</b><br>Altitude: {row['Altitude']}m<br>ID: {row['numer_sta']}",
                icon=folium.Icon(color='red', icon='cloud', size=(3, 3)),
            ).add_to(m)

    # # Add scale bar
    # scale_bar = folium.plugins.ScaleBar(position='bottomleft')
    # m.add_child(scale_bar)

    # Save the map
    m.save(output_file)
    print(f"Map saved as {output_file}")

# Example usage
if __name__ == "__main__":
    # Sample input data
    station_ids = return_data({'name': "molene/temp", "dataset_config": {"field": "t", "num_stations": "all"}})  # Replace with your station IDs
    adj_matrix = np.array([
        [0, 1, 0],  # Connections from station 1
        [0, 0, 1],  # Connections from station 2
        [1, 0, 0]   # Connections from station 3
    ])
    
    create_weather_map(
        station_ids=station_ids,
        adj_matrix=adj_matrix,
        output_file="custom_weather_map.html"
    )
