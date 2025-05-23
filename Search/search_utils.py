"""
Helper functions for using search algorithms from GUI applications.
"""
import os
import sys

# Add the necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import modules using the new package structure
from .data_reader import parse_graph_file

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
            from .Custom_Search.Dijkstras_Algorithm import run_dijkstra
            return run_dijkstra(graph_file_path, origin, destination, top_k)
        except Exception as e:
            print(f"Error running Dijkstra's algorithm: {e}")
            return []
    
    elif algorithm == "ACO":
        # Use ACO algorithm
        try:
            from .Custom_Search import run_aco
            return run_aco(graph_file_path, origin, destination, top_k)
        except Exception as e:
            print(f"Error running ACO algorithm: {e}")
            return []
        
    elif algorithm == "BFS":
        # Use BFS algorithm
        try:
            from .Uninformed_Search import run_bfs
            return run_bfs(graph_file_path, origin, destination, top_k)
        except Exception as e:
            print(f"Error running BFS algorithm: {e}")
            return []
            
    elif algorithm == "DFS":
        # Use DFS algorithm
        try:
            from .Uninformed_Search import run_dfs
            return run_dfs(graph_file_path, origin, destination, top_k)
        except Exception as e:
            print(f"Error running DFS algorithm: {e}")
            return []
            
    elif algorithm == "ASTAR" or algorithm == "A*":
        # Use A* algorithm
        try:
            from .Informed_Search import run_astar
            return run_astar(graph_file_path, origin, destination, top_k)
        except Exception as e:
            print(f"Error running A* algorithm: {e}")
            return []
            
    elif algorithm == "GBFS":
        # Use GBFS algorithm
        try:
            from .Informed_Search import run_gbfs
            return run_gbfs(graph_file_path, origin, destination, top_k)
        except Exception as e:
            print(f"Error running GBFS algorithm: {e}")
            return []
    
    else:  # Default to Dijkstra for all other algorithms
        try:
            from Custom_Search.Dijkstras_Algorithm.dijk import run_dijkstra
            print(f"Algorithm {algorithm} not fully implemented, falling back to Dijkstra's algorithm")
            return run_dijkstra(graph_file_path, origin, destination, top_k)
        except Exception as e:
            print(f"Error running fallback algorithm: {e}")
            return []
