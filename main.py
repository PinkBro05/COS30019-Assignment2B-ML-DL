#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to plot traffic scats locations onto an interactive map.
Uses GeoPy for geographical calculations and Folium for visualization.
"""

import os
import sys
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster, Search
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWebEngineWidgets import QWebEngineView

def read_geojson_file(geojson_file_path):
    """
    Read the GeoJSON file containing traffic light data with coordinates
    
    Args:
        geojson_file_path (str): Path to the GeoJSON file
        
    Returns:
        geopandas.GeoDataFrame: GeoDataFrame containing the traffic light data
    """
    try:
        # Read GeoJSON using geopandas
        gdf = gpd.read_file(geojson_file_path)
        print(f"Successfully read {len(gdf)} traffic light locations from {geojson_file_path}")
        
        # Extract longitude and latitude from geometry for easier access
        gdf['longitude'] = gdf.geometry.x
        gdf['latitude'] = gdf.geometry.y
        
        # Display first few rows to verify
        print(gdf[['SITE_NO', 'SITE_NAME', 'SITE_TYPE', 'longitude', 'latitude']].head())
        
        return gdf
    except Exception as e:
        print(f"Error reading GeoJSON file: {e}")
        sys.exit(1)

def create_map(gdf, center=None):
    """
    Create an interactive map with markers for each traffic light location
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame containing locations with geometry
        center (tuple, optional): Center coordinates for the map (lat, lon). If None, uses the mean of all points.
        
    Returns:
        folium.Map: The created map object
    """
    if center is None:
        # Calculate the center as the mean of all points
        center = [gdf['latitude'].mean(), gdf['longitude'].mean()]
    
    # Create a map with increased zoom capability
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
            'maxClusterRadius': 100,  # Increase clustering radius for even fewer markers at low zoom
            'spiderfyOnMaxZoom': True,  # Spread out markers when clicking on a cluster at max zoom
            'chunkedLoading': True  # Enable chunked loading for better performance
        }
    ).add_to(m)
    
    # Add markers for each location directly to the cluster without GeoJSON layer
    # This improves performance significantly
    for idx, row in gdf.iterrows():
        # Create popup text with relevant information
        popup_text = f"Site No: {row['SITE_NO']}<br>Name: {row['SITE_NAME']}<br>Type: {row['SITE_TYPE']}"
        if pd.notna(row['COMMENTS']) and row['COMMENTS']:
            popup_text += f"<br>Comments: {row['COMMENTS']}"
            
        # Add circle marker to cluster - more efficient than regular markers
        circle_marker = folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            popup=popup_text,
            radius=4,
            color='#3186cc',
            fill=True,
            fill_color='#3186cc',
            fill_opacity=0.7,
            tooltip=f"Site: {row['SITE_NO']} - {row['SITE_NAME']}"  # Add tooltip for hover information
        )
        circle_marker.add_to(marker_cluster)
    
    # Create a search control for the map
    create_search(gdf, m)

    # Add a layer control to toggle the visibility
    folium.LayerControl(
        position='topright',
        collapsed=True,
    ).add_to(m)
        
    return m

def create_search(gdf, map_obj):
    """
    Create a search control for the map to find traffic light locations by name
    
    Args:
        gdf (geopandas.GeoDataFrame): GeoDataFrame containing locations with geometry
        map_obj (folium.Map): The map object to add the search control to
        
    Returns:
        None
    """
    # Create a search control for the map but ensure it's not visible initially
    scats_layer = folium.FeatureGroup(name='Traffic Lights', show=False)
    scats = folium.GeoJson(
        gdf,
        name='Traffic Lights',
        show=False,
    )
    
    # Add search functionality
    search = Search(
        layer=scats,
        geom_type='Point',
        placeholder='Search by Site No or Name',
        collapsed=True,
        search_label='SITE_NO',
        position='topright'
    ).add_to(map_obj)

    # Add the layer to the map
    scats.add_to(scats_layer)
    scats_layer.add_to(map_obj)
    
    
def main():
    """
    Main function to read data and create the map visualization
    """
    # Path to the GeoJSON file
    geojson_path = os.path.join('Data', 'Traffic_Lights.geojson')
    
    # Read the GeoJSON file
    gdf = read_geojson_file(geojson_path)
    
    # Create the map
    m = create_map(gdf)
    
    # Display statistics
    print(f"Total traffic light locations: {len(gdf)}")
    
    # Save map to HTML file
    output_file = 'map_output.html'
    m.save(output_file)
    print(f"Map saved to {output_file}")

    # Create a PyQt5 application to display the map
    app = QtWidgets.QApplication(sys.argv)
    
    # Create a QWebEngineView instance
    view = QWebEngineView()
    view.setWindowTitle("Traffic Light Locations")
    view.setGeometry(100, 100, 1024, 768)  # Slightly larger window for better visibility
    
    # Load the HTML file
    view.load(QtCore.QUrl.fromLocalFile(os.path.abspath(output_file)))
    view.show()
    
    sys.exit(app.exec_())
    
    
if __name__ == "__main__":
    main()