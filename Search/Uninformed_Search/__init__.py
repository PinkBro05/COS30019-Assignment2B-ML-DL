"""
Package for uninformed search algorithms including BFS and DFS.
"""
from .bfs import BfsNetwork, main as bfs_main

def run_bfs(graph_file_path, origin, destination, top_k=1):
    """
    Run Breadth-First Search algorithm on a graph.
    
    Args:
        graph_file_path (str): Path to the graph file
        origin (str): Origin node ID
        destination (str or list): Destination node ID or list of destination node IDs
        top_k (int): Number of paths to return (BFS only returns the shortest path)
        
    Returns:
        list: List containing a tuple (path, cost) where path is a list of node IDs
              and cost is the total path cost
    """
    from ..data_reader import parse_graph_file
    
    # Parse the graph file
    nodes, edges, parsed_origin, destinations = parse_graph_file(graph_file_path, origin, destination)
    
    # Create and set up the BFS network
    network = BfsNetwork()
    network.build_from_data(nodes, edges)
    
    # Convert single destination to list if needed
    if isinstance(destination, str):
        dest_list = [destination]
    else:
        dest_list = destination
        
    # Find the shortest path to any destination
    shortest_path, shortest_dest, shortest_cost = network.find_shortest_path_to_destinations(origin, dest_list)
    
    # Return the result in the expected format
    if shortest_path:
        return [(shortest_path, shortest_cost)]
    else:
        return []

def run_dfs(graph_file_path, origin, destination, top_k=1):
    """
    Run Depth-First Search algorithm on a graph.
    
    Args:
        graph_file_path (str): Path to the graph file
        origin (str): Origin node ID
        destination (str or list): Destination node ID or list of destination node IDs
        top_k (int): Number of paths to return (DFS only returns the first path found)
        
    Returns:
        list: List containing a tuple (path, cost) where path is a list of node IDs
              and cost is the total path cost
    """
    from ..data_reader import parse_graph_file
    
    # Parse the graph file
    nodes, edges, parsed_origin, destinations = parse_graph_file(graph_file_path, origin, destination)
    
    # Create and set up the DFS network
    network = BfsNetwork()
    network.build_from_data(nodes, edges)
    
    # Convert single destination to list if needed
    if isinstance(destination, str):
        dest_list = [destination]
    else:
        dest_list = destination
        
    # Find the first path to any destination
    dfs_path, dfs_dest, dfs_cost = network.find_shortest_path_to_destinations(origin, dest_list)
    
    # Return the result in the expected format
    if dfs_path:
        return [(dfs_path, dfs_cost)]
    else:
        return []
