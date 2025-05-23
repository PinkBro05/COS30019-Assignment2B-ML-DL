"""
Script to create a graph.txt file for the traffic flow suggestion system.
Uses OSMnx to read Victoria, Australia road network and create a graph with
SCATS sites as nodes and road distances as edges.

For efficiency with large datasets, this script creates direct edges between 
SCATS sites based on haversine distances rather than calculating all shortest paths.
"""

import os
import networkx as nx
import osmnx as ox
import geopandas as gpd
from haversine import haversine, Unit
import numpy as np
from scipy.spatial import cKDTree
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        
        # Display sample data
        print(gdf[['SITE_NO', 'SITE_NAME', 'SITE_TYPE', 'longitude', 'latitude']].sample(5))
        
        return gdf
    except Exception as e:
        print(f"Error reading GeoJSON file: {e}")
        return None

def get_victoria_map(place_name="Victoria, Australia"): # WE WON't USE THIS DUE TO THE SCOPE OF THE PROJECT
    """
    Use OSMnx to fetch the Victoria, Australia road network
    
    Args:
        place_name (str): The name of the place to fetch
        
    Returns:
        networkx.MultiDiGraph: Road network graph
    """
    print(f"Fetching road network for {place_name}...")
    try:
        # Get the drive network for Victoria with drive_service type and simplify=True
        graph = ox.graph_from_place(place_name, 
                                   network_type='drive_service',
                                   simplify=True)
        
        print(f"Road network fetched: {len(graph.nodes)} nodes and {len(graph.edges)} edges")
        
        # Save a plot of the network for visualization
        try:
            fig, ax = ox.plot_graph(graph, figsize=(10, 10), node_size=0, edge_linewidth=0.5, 
                                   show=False, close=False)
            plt.title(f"Road Network: {place_name}")
            plt.tight_layout()
            plt.savefig(os.path.join('Utils', 'road_network.png'), dpi=300)
            plt.close()
        except Exception as e:
            print(f"Could not save network visualization: {e}")
        
        return graph
    except Exception as e:
        print(f"Error fetching road network: {e}")
        # Try with bounding box as fallback
        try:
            print("Trying with bounding box instead...")
            # Victoria, Australia bounding box (approximate)
            north, south, east, west = -33.9806, -39.1700, 149.9763, 140.9617
            graph = ox.graph_from_bbox(north, south, east, west, 
                                     network_type='drive_service',
                                     simplify=True)
            print(f"Road network fetched using bbox: {len(graph.nodes)} nodes and {len(graph.edges)} edges")
            return graph
        except Exception as e2:
            print(f"Error fetching road network with bounding box: {e2}")
            return None

def create_scats_graph(scats_gdf, max_distance_km=5):
    """
    Create a graph directly from SCATS sites, connecting nearby sites
    
    Args:
        scats_gdf (geopandas.GeoDataFrame): GeoDataFrame with SCATS sites
        max_distance_km (float): Maximum distance in kilometers to connect nodes
        
    Returns:
        networkx.Graph: Graph of SCATS sites
        dict: Dictionary of edge distances
    """
    print(f"Creating SCATS site graph (connecting sites within {max_distance_km} km)...")
    
    # Create a new graph
    scats_graph = nx.Graph()
    
    # Add nodes
    for idx, row in scats_gdf.iterrows():
        site_no = int(row['SITE_NO'])
        scats_graph.add_node(site_no, 
                           longitude=row['longitude'],
                           latitude=row['latitude'],
                           name=row['SITE_NAME'])
    
    # Create edges between nearby nodes
    edge_distances = {}
    nodes = list(scats_graph.nodes(data=True))
    
    # Use KD-tree for efficient nearest neighbor search
    coords = np.array([[data['latitude'], data['longitude']] for _, data in nodes])
    tree = cKDTree(coords)
    
    # Query all pairs of points within max_distance_km
    print("Finding nearby SCATS sites...")
    pairs = tree.query_pairs(max_distance_km * 0.01)  # Rough conversion to degrees
    
    print(f"Creating edges between {len(pairs)} pairs of nearby sites...")
    for i, j in tqdm(pairs):
        site1 = nodes[i][0]
        site2 = nodes[j][0]
        
        # Calculate haversine distance in meters
        lat1, lon1 = coords[i][0], coords[i][1]
        lat2, lon2 = coords[j][0], coords[j][1]
        
        distance = int(haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS))
        
        # Check if the distance not 0 (duplicate edges or long = lat)
        if distance == 0:
            continue
        
        # Add edge to graph
        scats_graph.add_edge(site1, site2, weight=distance)
        
        # Store distance
        edge_distances[(site1, site2)] = distance
        edge_distances[(site2, site1)] = distance
    
    print(f"Created graph with {scats_graph.number_of_nodes()} nodes and {scats_graph.number_of_edges()} edges")
    
    # Make sure the graph is connected
    if not nx.is_connected(scats_graph):
        print("Warning: Graph is not fully connected")
        components = list(nx.connected_components(scats_graph))
        print(f"Graph has {len(components)} connected components")
        print(f"Largest component has {len(components[0])} nodes")
    
    return scats_graph, edge_distances

def create_graph_txt(scats_gdf, edge_distances, output_path):
    """
    Create graph.txt file with the specified format
    
    Args:
        scats_gdf (geopandas.GeoDataFrame): GeoDataFrame with SCATS sites
        edge_distances (dict): Dictionary mapping (source, target) to distance
        output_path (str): Path to output file
    """
    print(f"Creating graph.txt file at {output_path}...")
    
    with open(output_path, 'w') as f:
        # Write header
        f.write("Nodes:\n")
        
        # Write nodes
        for _, row in scats_gdf.iterrows():
            site_no = int(row['SITE_NO'])
            lon = row['longitude']
            lat = row['latitude']
            f.write(f"{site_no}: ({lon},{lat})\n")
        
        # Write edges header
        f.write("Edges:\n")
        
        # Write edges
        for (source, target), distance in edge_distances.items():
            f.write(f"({source},{target}): {distance}\n")
            
    print(f"Graph file created with {len(scats_gdf)} nodes and {len(edge_distances)} edges")

def main():
    """
    Main function to read data, create graph, and write to file
    """
    start_time = time.time()
    
    # Path to the GeoJSON file
    geojson_path = os.path.join('Data', 'Traffic_Lights.geojson')
    
    # Output path for graph.txt
    output_path = os.path.join('Data', 'graph.txt')
    
    # Read the GeoJSON file
    scats_gdf = read_geojson_file(geojson_path)
    if scats_gdf is None:
        return
    
    # Create graph directly from SCATS sites
    scats_graph, edge_distances = create_scats_graph(scats_gdf, max_distance_km=5)
    
    # Create graph.txt
    create_graph_txt(scats_gdf, edge_distances, output_path)
    
    end_time = time.time()
    print(f"Done! Execution time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
