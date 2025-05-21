#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to plot traffic scats locations onto an interactive map.
Uses GeoPy for geographical calculations and Folium for visualization.
"""

import os
import sys
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWebEngineWidgets import QWebEngineView
qt_webengine_available = True

def read_map_csv(csv_file_path):
    """
    Read the map.csv file containing Scats data with longitude and latitude
    
    Args:
        csv_file_path (str): Path to the map.csv file
        
    Returns:
        pandas.DataFrame: DataFrame containing the Scats data
    """
    try:
        df = pd.read_csv(csv_file_path)
        print(f"Successfully read {len(df)} locations from {csv_file_path}")
        # Display first few rows to verify
        print(df.head())
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        sys.exit(1)

def create_map(df, center=None):
    """
    Create a minimal interactive map with markers for each location in the DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame containing locations with 'latitude' and 'longitude' columns
        center (tuple, optional): Center coordinates for the map (lat, lon). If None, uses the mean of all points.
        
    Returns:
        folium.Map: The created map object
    """
    if center is None:
        # Calculate the center as the mean of all points
        center = [df['latitude'].mean(), df['longitude'].mean()]
    
    # Create a minimal map with increased zoom capability
    m = folium.Map(
        location=center,
        zoom_start=10,
        max_zoom=22,  # Increase maximum zoom level
        tiles='CartoDB positron',  # Use a minimal, faster loading tile layer
        control_scale=True  # Add scale for better navigation
    )
    
    # Create a marker cluster with optimized settings for performance
    marker_cluster = MarkerCluster(
        options={
            'disableClusteringAtZoom': 16,  # Stop clustering at high zoom levels
            'maxClusterRadius': 80,  # Increase clustering radius for fewer markers at low zoom
            'spiderfyOnMaxZoom': True  # Spread out markers when clicking on a cluster at max zoom
        }
    ).add_to(m)
    
    # Add minimal markers for each location
    for idx, row in df.iterrows():
        popup_text = f"Scats ID: {row['Scats']}"
        folium.CircleMarker(  # Use CircleMarker instead of Marker with icon for better performance
            location=[row['latitude'], row['longitude']],
            popup=popup_text,
            radius=4,
            color='#3186cc',
            fill=True,
            fill_color='#3186cc',
            fill_opacity=0.7
        ).add_to(marker_cluster)
    
    return m

def main():
    """
    Main function to read data and create the map visualization
    """
    # Get the absolute path to the project directory
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Path to the map.csv file
    map_csv_path = os.path.join(project_dir, 'Data', 'map.csv')
    
    # Read the CSV file
    df = read_map_csv(map_csv_path)
    
    # Create the map
    m = create_map(df)
    
    # Display statistics
    print(f"Total locations: {len(df)}")
    
    # Save map to HTML file
    output_file = os.path.join(project_dir, 'map_output.html')
    m.save(output_file)

    # Create a PyQt5 application to display the map
    app = QtWidgets.QApplication(sys.argv)
    
    # Create a QWebEngineView instance
    view = QWebEngineView()
    view.setWindowTitle("Traffic Scats Locations")
    view.setGeometry(100, 100, 800, 600)
    
    # Load the HTML file
    view.load(QtCore.QUrl.fromLocalFile(os.path.abspath(output_file)))
    view.show()
    
    sys.exit(app.exec_())
    
    
if __name__ == "__main__":
    main()