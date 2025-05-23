"""
Helper functions for using search algorithms from GUI applications.
"""
import os
import sys
import importlib.util

# Add the necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "data_reader"))
sys.path.append(os.path.join(current_dir, "Custom_Search"))
sys.path.append(os.path.join(current_dir, "Custom_Search", "aco_routing"))

from parser import parse_graph_file
from Custom_Search.aco_routing.network import Network

def find_paths(graph_file_path, origin, destination, algorithm="AS", top_k=5):
    """
    Find paths from origin to destination using specified algorithm.
    
    Args:
        graph_file_path (str): Path to the graph file
        origin (str): Origin node ID
        destination (str or list): Destination node ID or list of destination node IDs
        algorithm (str): Algorithm to use for path finding
                         Options: "DIJK" (Dijkstra), "ACO", "AS" (A* - fallback to Dijkstra)
        top_k (int): Number of paths to return (most algorithms only return 1 path)
        
    Returns:
        list: List of tuples (path, cost) in ranked order, where path is a list of node IDs
              and cost is the total path cost
    """
    # Normalize the algorithm name
    algorithm = algorithm.upper()
    
    if algorithm == "DIJK" or algorithm == "DIJKSTRA":
        # Use Dijkstra's algorithm
        try:
            from Custom_Search.Dijkstras_Algorithm.dijk import run_dijkstra
            return run_dijkstra(graph_file_path, origin, destination, top_k)
        except Exception as e:
            print(f"Error running Dijkstra's algorithm: {e}")
            return []
    
    elif algorithm == "ACO":
        # Use ACO algorithm
        try:
            from Custom_Search.aco_search import run_aco
            return run_aco(graph_file_path, origin, destination, top_k)
        except Exception as e:
            print(f"Error running ACO algorithm: {e}")
            return []
    
    else:  # Default to Dijkstra for all other algorithms
        try:
            from Custom_Search.Dijkstras_Algorithm.dijk import run_dijkstra
            print(f"Algorithm {algorithm} not fully implemented, falling back to Dijkstra's algorithm")
            return run_dijkstra(graph_file_path, origin, destination, top_k)
        except Exception as e:
            print(f"Error running fallback algorithm: {e}")
            return []

def highlight_path_on_map(m, path_nodes, node_positions, edges, color='red', width=4):
    """
    Highlight a path on the map.
    
    Args:
        m (folium.Map): The map object
        path_nodes (list): List of node IDs in the path
        node_positions (dict): Dictionary mapping node IDs to (lon, lat)
        edges (dict): Dictionary mapping (node1, node2) to edge weight
        color (str): Color of the path
        width (int): Width of the path line
        
    Returns:
        folium.FeatureGroup: The feature group containing the path
    """
    import folium
    
    # Create a feature group for the path
    path_group = folium.FeatureGroup(name=f"Path (Cost: {sum(edges.get((path_nodes[i], path_nodes[i+1]), 0) for i in range(len(path_nodes)-1))})")
      # Add path segments
    for i in range(len(path_nodes) - 1):
        start_node = path_nodes[i]
        end_node = path_nodes[i+1]
        # Get node coordinates (lon, lat) with error handling
        # Use a default position for missing nodes or skip the segment
        missing_node = False
        try:
            start_pos = node_positions[start_node]
        except KeyError:
            print(f"Warning: Node {start_node} not found in node positions data.")
            missing_node = True
            
        try:
            end_pos = node_positions[end_node]
        except KeyError:
            print(f"Warning: Node {end_node} not found in node positions data.")
            missing_node = True
            
        if missing_node:
            # Skip this segment
            continue
        
        # Get edge weight
        weight = edges.get((start_node, end_node), 0)
        
        # Add line
        folium.PolyLine(
            locations=[[start_pos[1], start_pos[0]], [end_pos[1], end_pos[0]]],
            color=color,
            weight=width,
            opacity=0.8,
            tooltip=f"Distance: {weight}m"
        ).add_to(path_group)
        
        # Add markers for each node in the path
        folium.CircleMarker(
            location=[start_pos[1], start_pos[0]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            tooltip=f"Node: {start_node}"
        ).add_to(path_group)
    
    # Add marker for the last node
    last_node = path_nodes[-1]
    try:
        last_pos = node_positions[last_node]
        folium.CircleMarker(
            location=[last_pos[1], last_pos[0]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            tooltip=f"Node: {last_node}"
        ).add_to(path_group)
    except KeyError:
        print(f"Warning: Last node {last_node} not found in node positions data.")
    
    # Add the path group to the map
    path_group.add_to(m)
    
    return path_group
