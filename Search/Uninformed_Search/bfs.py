import os
import sys
import argparse
import traceback

# Set up path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "data_reader"))

try:
    from data_reader.parser import parse_graph_file
except ImportError:
    try:
        from parser import parse_graph_file
    except ImportError:
        print("Could not import parser module. Check paths and module availability.")
        sys.exit(1)

# Import the BfsNetwork class
sys.path.append(os.path.join(current_dir, "entity"))
try:
    from Uninformed_Search.entity.BfsNetwork import BfsNetwork
except ImportError:
    try:
        from entity.BfsNetwork import BfsNetwork
    except ImportError:
        print("Could not import BfsNetwork class. Check paths and module availability.")
        sys.exit(1)

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Breadth-First Search Algorithm for path finding')
    parser.add_argument('file_path', nargs='?', default="Data/PathFinder-test.txt",
                            help='Path to the graph file (default: Data/PathFinder-test.txt)')
    parser.add_argument('--start', help='Starting node (if not using origin from file)')
    parser.add_argument('--end', help='Target node (if not using destinations from file)')
    parser.add_argument('--k', type=int, default=5, help='Number of paths to find (default: 5)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')

    # Check if the script was called directly or through search.py
    if len(sys.argv) > 1:
        args = parser.parse_args()
        file_path = args.file_path
    else:
        # Default file path if no arguments provided
        file_path = "Data/PathFinder-test.txt"
        args = parser.Namespace(start=None, end=None, k=5, debug=False)

    try:
        nodes, edges, origin, destinations = parse_graph_file(file_path)
        
        # Get start node (from command line or file)
        if hasattr(args, 'start') and args.start:
            # Use string values for consistency with the graph data
            start_node = str(args.start)
        else:
            start_node = str(origin)
            
        # Get destination nodes (from command line or file)
        if hasattr(args, 'end') and args.end:
            # Use string values for consistency with the graph data
            end_node = str(args.end)
            end_nodes = [end_node]
        else:
            # Destinations already come as strings from parser
            end_nodes = destinations
            
        # Print goals and number of nodes
        print(f"{file_path} BFS")
        print(f"[{', '.join(map(str, end_nodes))}]", len(nodes))
        
        # Create the BfsNetwork instance
        network = BfsNetwork()
        network.build_from_data(nodes, edges)
        
        # Get number of paths to find
        k_paths = args.k
        
        # Add some debug info
        debug_mode = args.debug
        if debug_mode:
            print(f"\nDEBUG INFO:")
            print(f"Start node: {start_node} (type: {type(start_node)})")
            print(f"End nodes: {end_nodes} (type: {type(end_nodes[0]) if end_nodes else None})")
            print(f"Network nodes: {list(network.graph.keys())[:5]}...")
            print(f"Network edge example: {list(edges.items())[:2]}")
            print(f"K paths: {k_paths}")
            
        # Find and display the k shortest paths to any destination
        shortest_paths = network.find_k_shortest_paths_to_destinations(start_node, end_nodes, k=k_paths)
        
        # Show the results
        if shortest_paths:
            for i, (path, dest, cost) in enumerate(shortest_paths, 1):
                print(f"Path {i} to {dest}: {' '.join(map(str, path))}")
                print(f"Cost {i}: {cost}")
        else:
            print("\nNo paths found to any destination.")
    except Exception as e:
        print(f"Error parsing graph file: {e}")

if __name__ == "__main__":
    main()