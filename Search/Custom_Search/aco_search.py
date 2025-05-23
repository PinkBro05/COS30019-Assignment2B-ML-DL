import os
import sys
import traceback
import argparse
import time
import multiprocessing

# Set up correct import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add path to find aco_routing module
sys.path.append(current_dir)

# Add path to find data_reader module
sys.path.append(parent_dir)

# Import ACO modules
from aco_routing.aco import ACO
from aco_routing.network import Network

# Import parser from data_reader
from data_reader.parser import parse_graph_file

def run_aco(graph_file_path, origin, destination, top_k=1):
    """
    Run ACO search algorithm for path finding.
    
    Args:
        graph_file_path (str): Path to the graph file
        origin (str): Origin node ID
        destination (str or list): Destination node ID or list of destination node IDs
        top_k (int): Number of paths to return (currently only returns 1 path)
        
    Returns:
        list: List of tuples (path, cost) in ranked order, where path is a list of node IDs
              and cost is the total path cost
    """
    try:
        # Parse the graph file
        nodes, edges, _, _ = parse_graph_file(graph_file_path)
        
        # Create the graph - optimize memory usage
        G = Network()
        
        # Pre-allocate graph memory
        G.graph = {node: [] for node in nodes}
        G.pos = nodes

        # Add edges 
        for (start, end), weight in edges.items():
            G.add_edge(start, end, cost=float(weight))

        # Calculate adaptive parameters
        node_count = G.number_of_nodes()
        use_floyd_warshall = False
        visualize = False
        iterations = 3
        ant_max_steps = node_count + 1
        num_ants = 100
        alpha = 1
        beta = 2
        evaporation_rate = 0.5
        
        # Determine optimal parameters based on graph size
        use_local_search = True
        local_search_frequency = 10  # Apply local search every 10 iterations
        num_threads = min(multiprocessing.cpu_count(), 4)  # Use available CPU cores efficiently
        
        # Convert destination to list if it's a single node
        if isinstance(destination, str):
            destination = [destination]
        
        # Initialize ACO with optimized parameters
        aco = ACO(G, 
            ant_max_steps=ant_max_steps,
            num_iterations=iterations, 
            evaporation_rate=evaporation_rate, 
            alpha=alpha, 
            beta=beta, 
            mode=0, # 0: any destination, 1: all destinations, 2: TSP mode
            log_step=None, # Setting log, Int or None
            visualize=visualize,
            visualization_step=None,
            use_floyd_warshall=use_floyd_warshall,
            use_local_search=use_local_search,
            local_search_frequency=local_search_frequency,
            num_threads=num_threads
        )
        
        # Find shortest path using ACO
        aco_path, aco_cost = aco.find_shortest_path(
            source=origin,
            destination=destination,
            num_ants=num_ants,
        )
        
        # Return the result as a list of tuples (path, cost)
        if aco_path and aco_cost > 0:
            return [(aco_path, aco_cost)]
        else:
            return []
            
    except Exception as e:
        print(f"Error running ACO algorithm: {e}")
        traceback.print_exc()
        return []