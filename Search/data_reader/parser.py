import re

def parse_graph_file(file_path, origin=None, destinations=None):
    """
    Parses a text file containing graph data and extracts nodes, edges, origin, and destinations.
    
    Args:
        file_path (str): Path to the text file containing the graph data.
        origin (str, optional): If provided, overrides the origin node from the file.
        destinations (list or set, optional): If provided, overrides the destinations from the file.
    
    Returns:
        tuple: A tuple containing:
            - nodes (dict): Mapping of node IDs to coordinates {node_id: (x, y)}.
            - edges (dict): Mapping of edge pairs to weights {(node1, node2): weight}.
            - origin (str): The origin node.
            - destinations (set): List of destination nodes.
    """
    nodes = {}
    edges = {}
    file_origin = None
    file_destinations = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        section = None
        
        for line in lines:
            line = line.strip()
            
            if line.startswith("Nodes:"):
                section = "nodes"
                continue
            elif line.startswith("Edges:"):
                section = "edges"
                continue
            elif line.startswith("Origin:"):
                section = "origin"
                continue
            elif line.startswith("Destinations:"):
                section = "destinations"
                continue
            
            if section == "nodes":
                match = re.match(r"(\d+): \((\d+),(\d+)\)", line)
                if match:
                    node_id = str(match.group(1))
                    x, y = int(match.group(2)), int(match.group(3))
                    nodes[node_id] = (x, y)
            
            elif section == "edges":
                match = re.match(r"\((\d+),(\d+)\): (\d+)", line)
                if match:
                    node1, node2, weight = str(match.group(1)), str(match.group(2)), int(match.group(3))
                    edges[(node1, node2)] = weight
            
            elif section == "origin":
                file_origin = str(line)
            
            elif section == "destinations":
                file_destinations = set([d.strip() for d in line.split(';') if d.strip()])
    
    # Use provided origin and destinations if available, otherwise use what was in the file
    actual_origin = origin if origin is not None else file_origin
    actual_destinations = set(destinations) if destinations is not None else file_destinations
    
    return nodes, edges, actual_origin, actual_destinations

# Example usage:
if __name__ == "__main__":
    file_path = "Data/Modified_TSP/test_30.txt"  # Replace with your actual file name
    nodes, edges, origin, destinations = parse_graph_file(file_path)
    
    print("Nodes:", nodes)
    print("Edges:", edges)
    print("Origin:", origin)
    print("Destinations:", destinations)