"""
Package for informed search algorithms including A* and Greedy Best-First Search (GBFS).
"""

def run_astar(graph_file_path, origin, destination, top_k=1):
    """
    Run A* search algorithm on a graph.
    
    Args:
        graph_file_path (str): Path to the graph file
        origin (str): Origin node ID
        destination (str or list): Destination node ID or list of destination node IDs
        top_k (int): Number of paths to return (A* returns the optimal path)
        
    Returns:
        list: List containing a tuple (path, cost) where path is a list of node IDs
              and cost is the total path cost
    """
    from .A_Star import a_star_search
    from ..data_reader import parse_graph_file
    
    # Parse the graph file
    nodes, edges, parsed_origin, destinations = parse_graph_file(graph_file_path, origin, destination)
    
    # Convert single destination to list if needed
    if isinstance(destination, str):
        dest_list = [destination]
    else:
        dest_list = destination
    
    # Run A* search for each destination and return the best path
    best_path = None
    best_cost = float('inf')
    
    for dest in dest_list:
        path, cost = a_star_search(nodes, edges, origin, dest)
        if path and cost < best_cost:
            best_path = path
            best_cost = cost
    
    # Return the result in the expected format
    if best_path:
        return [(best_path, best_cost)]
    else:
        return []

def run_gbfs(graph_file_path, origin, destination, top_k=1):
    """
    Run Greedy Best-First Search algorithm on a graph.
    
    Args:
        graph_file_path (str): Path to the graph file
        origin (str): Origin node ID
        destination (str or list): Destination node ID or list of destination node IDs
        top_k (int): Number of paths to return (GBFS returns a single path)
        
    Returns:
        list: List containing a tuple (path, cost) where path is a list of node IDs
              and cost is the total path cost
    """
    from .GBFS import gbfs_search
    from ..data_reader import parse_graph_file
    
    # Parse the graph file
    nodes, edges, parsed_origin, destinations = parse_graph_file(graph_file_path, origin, destination)
    
    # Convert single destination to list if needed
    if isinstance(destination, str):
        dest_list = [destination]
    else:
        dest_list = destination
    
    # Run GBFS search for each destination and return the best path
    best_path = None
    best_cost = float('inf')
    
    for dest in dest_list:
        path, cost = gbfs_search(nodes, edges, origin, dest)
        if path and cost < best_cost:
            best_path = path
            best_cost = cost
    
    # Return the result in the expected format
    if best_path:
        return [(best_path, best_cost)]
    else:
        return []
