#!/usr/bin/env python3
# Modified Dijkstra implementation with proper imports and error handling
import os
import sys
import argparse
import traceback

# Get absolute path to the project root (Search directory)
search_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Go up one more level to get the actual project root
project_root = os.path.dirname(search_root)

# Add relevant paths
sys.path.append(project_root)
sys.path.append(search_root)
data_reader_dir = os.path.join(search_root, "data_reader")
sys.path.append(data_reader_dir)
entity_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entity")
sys.path.append(entity_dir)

# Import parser using explicit absolute paths
parser_path = os.path.join(search_root, "data_reader", "parser.py")
if not os.path.exists(parser_path):
    print(f"Parser module not found at: {parser_path}")
    sys.exit(1)

# Import network class and entity modules
dijkstra_network_path = os.path.join(entity_dir, "DijkstraNetwork.py")
if not os.path.exists(dijkstra_network_path):
    print(f"DijkstraNetwork not found at: {dijkstra_network_path}")
    sys.exit(1)

# Try importing the modules
try:
    # Try both possible ways to import the parser
    try:
        from data_reader.parser import parse_graph_file
    except ImportError:
        # Fallback to direct import if module approach fails
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location("parser", parser_path)
        parser_module = importlib.util.module_from_spec(spec)
        sys.modules["parser"] = parser_module
        spec.loader.exec_module(parser_module)
        parse_graph_file = parser_module.parse_graph_file
except ImportError as e:
    print(f"Error importing parser: {e}")
    print(f"sys.path: {sys.path}")        
    sys.exit(1)

try:
    # Try both possible ways to import DijkstraNetwork
    try:
        sys.path.append(os.path.dirname(dijkstra_network_path))
        from DijkstraNetwork import DijkstraNetwork
    except ImportError:
        # Fallback to direct import
        import importlib.util
        spec = importlib.util.spec_from_file_location("DijkstraNetwork", dijkstra_network_path)
        if spec is not None:  # Check if spec is not None
            network_module = importlib.util.module_from_spec(spec)
            sys.modules["DijkstraNetwork"] = network_module
            if spec.loader is not None:  # Check if loader is not None
                spec.loader.exec_module(network_module)
                DijkstraNetwork = network_module.DijkstraNetwork
            else:
                print(f"Error: spec.loader is None for DijkstraNetwork")
                sys.exit(1)
        else:
            print(f"Error: Could not create module spec for DijkstraNetwork")
            sys.exit(1)
except ImportError as e:
    print(f"Error importing DijkstraNetwork: {e}")
    sys.exit(1)

def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Dijkstra\'s Algorithm for path finding')
    parser.add_argument('file_path', nargs='?', default=os.path.join(project_root, "Data", "graph.txt"),
                        help='Path to the graph file')
    parser.add_argument('--start', help='Starting node (if not using origin from file)')
    parser.add_argument('--end', help='Target node (if not using destinations from file)')
    parser.add_argument('--k', type=int, default=5, help='Number of paths to find (default: 5)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    # Parse command line args or use defaults
    args = parser.parse_args()
    file_path = args.file_path
    
    # Handle relative paths
    if not os.path.isabs(file_path):
        file_path = os.path.join(project_root, file_path)
    
    try:
        # Debug info
        print(f"Project root: {project_root}")
        print(f"Loading graph from: {file_path}")
        
        # Parse the graph file
        nodes, edges, origin, destinations = parse_graph_file(file_path)
        
        # Get start node
        start_node = str(args.start) if args.start else str(origin)
        
        # Get end nodes
        if args.end:
            end_nodes = [str(args.end)]
        else:
            end_nodes = destinations
        
        # Print header info
        print(f"{os.path.basename(file_path)} CUS1")
        print(f"[{', '.join(map(str, end_nodes))}]", len(nodes))
        
        # Create network
        network = DijkstraNetwork()
        network.build_from_data(nodes, edges)
        
        # Number of paths to find
        k_paths = args.k
        
        # Debug info
        if args.debug:
            print("\nDEBUG INFO:")
            print(f"Start node: {start_node} (type: {type(start_node).__name__})")
            if end_nodes:
                first_end = next(iter(end_nodes)) if isinstance(end_nodes, set) else end_nodes[0]
                print(f"End nodes: {end_nodes} (type: {type(first_end).__name__})")
            else:
                print(f"End nodes: {end_nodes} (empty)")
            
            node_sample = list(network.graph.keys())[:5] if len(network.graph) > 5 else list(network.graph.keys())
            print(f"Network nodes sample: {node_sample}...")
            
            edge_sample = list(edges.items())[:2] if edges else []
            print(f"Network edge example: {edge_sample}")
            print(f"K paths: {k_paths}")
        
        # Find paths
        shortest_paths = network.find_k_shortest_paths_to_destinations(start_node, end_nodes, k=k_paths)
        
        # Display results
        if shortest_paths:
            for i, (path, dest, cost) in enumerate(shortest_paths, 1):
                print(f"Path {i} to {dest}: {' '.join(map(str, path))}")
                print(f"Cost {i}: {cost}")
        else:
            print("\nNo paths found to any destination.")
            
    except FileNotFoundError:
        print(f"Error: The graph file '{file_path}' was not found.")
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
