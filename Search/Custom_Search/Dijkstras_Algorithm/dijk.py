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


def main():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Dijkstra\'s Algorithm for path finding')
    parser.add_argument('file_path', nargs='?', default="Data/PathFinder-test.txt",
                        help='Path to the graph file (default: Data/PathFinder-test.txt)')
        
    # Check if the script was called directly or through search.py
    if len(sys.argv) > 1:
        args = parser.parse_args()
        file_path = args.file_path
    else:
        # Default file path if no arguments provided
        file_path = "Data/PathFinder-test.txt"

    try:
        nodes, edges, origin, destinations = parse_graph_file(file_path)

        # Print goals and number of nodes
        print(f"{file_path} CUS1")
        print(f"[{', '.join(destinations)}]", len(nodes))

        # Create the DijkstraNetwork instance
        network = DijkstraNetwork()
        network.build_from_data(nodes, edges)

        
        # Find and display the shortest path to any destination
        shortest_path, shortest_dest, shortest_cost = network.find_shortest_path_to_destinations(origin, destinations)

        # Show the result
        if shortest_path:
            print(f"{' '.join(map(str, shortest_path))}")
            print(f"{shortest_cost}")
        else:
            print("\nNo paths found to any destination.")
    except Exception as e:
        print(f"Error parsing graph file: {e}")

if __name__ == "__main__":
    main()