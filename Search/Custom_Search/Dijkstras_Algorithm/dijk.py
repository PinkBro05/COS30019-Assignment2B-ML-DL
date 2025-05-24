import os
import sys
import argparse

# Set up path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "..", "data_reader"))
from parser import parse_graph_file

# Import the DijkstraNetwork class
sys.path.append(os.path.join(current_dir, "entity"))
from DijkstraNetwork import DijkstraNetwork


def run_dijkstra(graph_file_path, origin, destination, top_k=1):
    """
    Run Dijkstra's algorithm for path finding.
    
    Args:
        graph_file_path (str): Path to the graph file
        origin (str): Origin node ID
        destination (str or list): Destination node ID or list of destination node IDs
        top_k (int): Number of paths to return (currently only supports finding 1 path)
        
    Returns:
        list: List of tuples (path, cost) in ranked order, where path is a list of node IDs
              and cost is the total path cost
    """
    try:
        # Parse the graph file
        nodes, edges, _, _ = parse_graph_file(graph_file_path)
        
        # Create the DijkstraNetwork instance
        network = DijkstraNetwork()
        network.build_from_data(nodes, edges)
        
        # Convert destination to list if it's a single node
        if isinstance(destination, str):
            destination = [destination]
        
        # Find the shortest path to any destination
        shortest_path, shortest_dest, shortest_cost = network.find_shortest_path_to_destinations(origin, destination)
        
        # Return the result as a list of tuples (path, cost)
        if shortest_path:
            return [(shortest_path, shortest_cost)]
        else:
            return []
            
    except Exception as e:
        print(f"Error running Dijkstra's algorithm: {e}")
        return []