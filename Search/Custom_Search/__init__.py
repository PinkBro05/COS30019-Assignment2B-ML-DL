"""
Package for custom search algorithms including Ant Colony Optimization (ACO) and Dijkstra's Algorithm.
"""

def run_aco(graph_file_path, origin, destination, top_k=5):
    """
    Run Ant Colony Optimization algorithm on a graph.
    
    Args:
        graph_file_path (str): Path to the graph file
        origin (str): Origin node ID
        destination (str or list): Destination node ID or list of destination node IDs
        top_k (int): Number of paths to return
        
    Returns:
        list: List of tuples (path, cost) in ranked order, where path is a list of node IDs
              and cost is the total path cost
    """
    from .aco_search import run_aco
    
    try:
        return run_aco(graph_file_path, origin, destination, top_k)
    except Exception as e:
        print(f"Error in ACO search: {e}")
        return []

def run_dijkstra(graph_file_path, origin, destination, top_k=5):
    """
    Run Dijkstra's algorithm on a graph.
    
    Args:
        graph_file_path (str): Path to the graph file
        origin (str): Origin node ID
        destination (str or list): Destination node ID or list of destination node IDs
        top_k (int): Number of paths to return
        
    Returns:
        list: List of tuples (path, cost) in ranked order, where path is a list of node IDs
              and cost is the total path cost
    """
    from .Dijkstras_Algorithm.dijk import run_dijkstra as _run_dijkstra
    
    try:
        return _run_dijkstra(graph_file_path, origin, destination, top_k)
    except Exception as e:
        print(f"Error in Dijkstra's algorithm: {e}")
        return []
