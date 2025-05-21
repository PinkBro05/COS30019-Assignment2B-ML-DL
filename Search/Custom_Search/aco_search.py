import os
import sys
import traceback
import argparse
import time
import multiprocessing

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from aco_routing.aco import ACO
from aco_routing.network import Network

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "data_reader"))

from parser import parse_graph_file

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ACO Search Algorithm')
    parser.add_argument('file_path', nargs='?', default="Data/PathFinder-Test.txt",
                        help='Path to the graph file (default: Data/PathFinder-Test.txt)')
    
    # Check if the script was called directly or through search.py
    if len(sys.argv) > 1:
        args = parser.parse_args()
        file_path = args.file_path
    else:
        # Default values if no arguments provided
        file_path = "Data/PathFinder-test.txt"

    
    try:
        nodes, edges, origin, destinations = parse_graph_file(file_path)
    except Exception as e:
        print(f"Error parsing graph file: {e}")
        traceback.print_exc()
        return

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
    iterations = 20
    ant_max_steps = node_count + 1
    num_ants = node_count
    alpha = 1
    beta = 2
    evaporation_rate = 0.5
    
    # Determine optimal parameters based on graph size
    use_local_search = True
    local_search_frequency = 10  # Apply local search every 10 iterations
    num_threads = min(multiprocessing.cpu_count(), 32)  # Use available CPU cores efficiently
    
    # Measure execution time
    start_time = time.time()
    
    # Initialize ACO with optimized parameters including thread-based parallelization and local search
    aco = ACO(G, 
        ant_max_steps=ant_max_steps,
        num_iterations=iterations, 
        evaporation_rate=evaporation_rate, 
        alpha=alpha, 
        beta=beta, 
        mode=0, # 0: any destination, 1: all destinations, 2: TSP mode
        log_step=None, # Setting log, Int or None
        visualize=visualize,  # Enable visualization
        visualization_step=None,  # Update visualization every 10 iterations
        use_floyd_warshall=use_floyd_warshall,  # Use Floyd-Warshall preprocessing
        use_local_search=use_local_search,  # Enable local search optimization
        local_search_frequency=local_search_frequency,  # Apply local search every N iterations
        num_threads=num_threads  # Use thread-based parallelization
    )
    
    aco_path, aco_cost = aco.find_shortest_path(
        source=origin,
        destination=destinations,
        num_ants=num_ants,
    )
    
    # Calculate execution time
    # execution_time = time.time() - start_time
    # print(f"\nExecution time: {execution_time:.4f} seconds")
    
    # Print path stats
    # if aco_path:
    #     print(f"Path length: {len(aco_path)} nodes")
    #     print(f"Path cost: {aco_cost}")
    #     print("------------------------------")

    # Format the output for the assignment requirements
    if aco_cost == 0:
        print(f"{file_path} CUS2")
        print(f"[{', '.join(destinations)}] {G.number_of_nodes()}")
        print(f"Destination already reached: Origin {origin} to destination {destinations}")
        print("0.0")
    elif not aco_path:
        # No path found but no exception thrown
        print(f"{file_path} CUS2")
        print(f"[{', '.join(destinations)}] {G.number_of_nodes()}")
        print("No path found")
        print("0.0")
    else:
        # Normal output
        aco_path = [node for node in aco_path]
        goal_str = aco_path[-1]
        number_of_nodes = G.number_of_nodes()
        path_str = ", ".join(aco_path)
        
        print(f"{file_path} CUS2")
        print(f"{goal_str} {number_of_nodes}")
        print(f"[{path_str}]")
        # print(f"{aco_cost}")
        
        # Only visualize final result if not already visualized during execution
        # if not visualize:
        #     aco.graph_api.visualize_graph(aco_path, aco_cost)

if __name__ == "__main__":
    main()
