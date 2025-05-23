"""
Chunking strategy implementation to reduce search space for path finding algorithms.
Creates a bounding box around origin and destination points to filter the graph.
"""

import math
import os
import sys
from typing import Dict, Tuple, Set, List, Optional

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on Earth (specified in decimal degrees).
    
    Args:
        lat1, lon1: Latitude and longitude of point 1
        lat2, lon2: Latitude and longitude of point 2
        
    Returns:
        Distance in meters
    """
    # Convert decimal degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of Earth in meters
    r = 6371000
    
    return c * r

def create_bounding_box(origin_coords: Tuple[float, float], 
                       dest_coords: Tuple[float, float], 
                       margin_factor: float = 0.2) -> Tuple[float, float, float, float]:
    """
    Create a bounding box around origin and destination coordinates with a margin.
    
    Args:
        origin_coords: (longitude, latitude) of origin point
        dest_coords: (longitude, latitude) of destination point
        margin_factor: Factor to expand the bounding box (0.2 = 20% margin)
        
    Returns:
        Tuple of (min_lon, min_lat, max_lon, max_lat)
    """
    # Extract coordinates
    origin_lon, origin_lat = origin_coords
    dest_lon, dest_lat = dest_coords
    
    # Find the min/max coordinates
    min_lon = min(origin_lon, dest_lon)
    max_lon = max(origin_lon, dest_lon)
    min_lat = min(origin_lat, dest_lat)
    max_lat = max(origin_lat, dest_lat)
    
    # Calculate margins based on the bounding box size
    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat
    
    # Add minimum margin to avoid too small bounding boxes
    min_margin_deg = 0.01  # About 1km
    lon_margin = max(lon_range * margin_factor, min_margin_deg)
    lat_margin = max(lat_range * margin_factor, min_margin_deg)
    
    # Expand the bounding box
    min_lon -= lon_margin
    max_lon += lon_margin
    min_lat -= lat_margin
    max_lat += lat_margin
    
    return min_lon, min_lat, max_lon, max_lat

def point_in_bounding_box(point_coords: Tuple[float, float], 
                         bounding_box: Tuple[float, float, float, float]) -> bool:
    """
    Check if a point is within the bounding box.
    
    Args:
        point_coords: (longitude, latitude) of the point
        bounding_box: (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        True if point is within bounding box
    """
    point_lon, point_lat = point_coords
    min_lon, min_lat, max_lon, max_lat = bounding_box
    
    return (min_lon <= point_lon <= max_lon and 
            min_lat <= point_lat <= max_lat)

def filter_nodes_by_chunk(nodes: Dict[str, Tuple[float, float]], 
                         origin_id: str, 
                         destination_id: str, 
                         margin_factor: float = 0.2) -> Set[str]:
    """
    Filter nodes to include only those within the bounding box chunk.
    
    Args:
        nodes: Dictionary mapping node IDs to (longitude, latitude) coordinates
        origin_id: ID of the origin node
        destination_id: ID of the destination node
        margin_factor: Factor to expand the bounding box
        
    Returns:
        Set of node IDs that are within the chunk
    """
    # Validate that origin and destination exist
    if origin_id not in nodes:
        raise ValueError(f"Origin node '{origin_id}' not found in nodes")
    if destination_id not in nodes:
        raise ValueError(f"Destination node '{destination_id}' not found in nodes")
    
    # Get coordinates for origin and destination
    origin_coords = nodes[origin_id]
    dest_coords = nodes[destination_id]
    
    # Create bounding box
    bounding_box = create_bounding_box(origin_coords, dest_coords, margin_factor)
    
    # Filter nodes within the bounding box
    filtered_nodes = set()
    for node_id, coords in nodes.items():
        if point_in_bounding_box(coords, bounding_box):
            filtered_nodes.add(node_id)
    
    # Always include origin and destination (safety check)
    filtered_nodes.add(origin_id)
    filtered_nodes.add(destination_id)
    
    return filtered_nodes

def filter_edges_by_nodes(edges: Dict[Tuple[str, str], int], 
                         valid_nodes: Set[str]) -> Dict[Tuple[str, str], int]:
    """
    Filter edges to include only those connecting nodes within the valid set.
    
    Args:
        edges: Dictionary mapping (node1, node2) to edge weight
        valid_nodes: Set of valid node IDs
        
    Returns:
        Filtered edges dictionary
    """
    filtered_edges = {}
    for (node1, node2), weight in edges.items():
        # Include any edge that has at least one endpoint inside the valid nodes set
        # This ensures all nodes inside the bounding box remain connected through their edges
        if node1 in valid_nodes or node2 in valid_nodes:
            filtered_edges[(node1, node2)] = weight
    
    return filtered_edges

def create_chunked_graph(graph_file_path: str, 
                        origin_id: str, 
                        destination_id: str, 
                        margin_factor: float = 0.2) -> Tuple[Dict[str, Tuple[float, float]], 
                                                            Dict[Tuple[str, str], int], 
                                                            str, 
                                                            Set[str]]:
    """
    Create a chunked version of the graph containing only nodes and edges within 
    the bounding box of origin and destination.
    
    Args:
        graph_file_path: Path to the graph file
        origin_id: ID of the origin node
        destination_id: ID of the destination node
        margin_factor: Factor to expand the bounding box (default 0.2 = 20% margin)
        
    Returns:
        Tuple of (filtered_nodes, filtered_edges, origin, destinations)
        Same format as the original parser but with reduced search space
    """    # Import the parser using dynamic import to avoid path issues
    import importlib.util
    search_dir = os.path.join(os.path.dirname(__file__), '..', 'Search')
    parser_path = os.path.join(search_dir, 'data_reader', 'parser.py')
    
    spec = importlib.util.spec_from_file_location("parser", parser_path)
    parser_module = importlib.util.module_from_spec(spec)
    sys.path.append(search_dir)
    spec.loader.exec_module(parser_module)
    parse_graph_file = parser_module.parse_graph_file
    
    # Parse the original graph
    nodes, edges, _, _ = parse_graph_file(graph_file_path, origin_id, {destination_id})
      # Filter nodes by chunk
    valid_nodes = filter_nodes_by_chunk(nodes, origin_id, destination_id, margin_factor)
    
    # Filter edges to include all connections to nodes in the bounding box
    filtered_edges = filter_edges_by_nodes(edges, valid_nodes)
    
    # Update valid nodes to include any nodes connected by our filtered edges
    for (node1, node2), _ in filtered_edges.items():
        valid_nodes.add(node1)
        valid_nodes.add(node2)
      # Create filtered nodes dictionary with all needed nodes
    filtered_nodes = {node_id: coords for node_id, coords in nodes.items() 
                     if node_id in valid_nodes}
    
    # print(f"Chunking reduced search space:")
    # print(f"  Original nodes: {len(nodes)}")
    # print(f"  Filtered nodes: {len(filtered_nodes)}")
    # print(f"  Reduction: {(1 - len(filtered_nodes)/len(nodes))*100:.1f}%")
    # print(f"  Original edges: {len(edges)}")
    # print(f"  Filtered edges: {len(filtered_edges)}")
    # print(f"  Reduction: {(1 - len(filtered_edges)/len(edges))*100:.1f}%")
    
    return filtered_nodes, filtered_edges, origin_id, {destination_id}

def write_chunked_graph_to_file(filtered_nodes: Dict[str, Tuple[float, float]], 
                               filtered_edges: Dict[Tuple[str, str], int], 
                               origin_id: str, 
                               destination_ids: Set[str], 
                               output_file_path: str):
    """
    Write the chunked graph to a temporary file for use by search algorithms.
    
    Args:
        filtered_nodes: Filtered nodes dictionary
        filtered_edges: Filtered edges dictionary
        origin_id: Origin node ID
        destination_ids: Set of destination node IDs
        output_file_path: Path to write the chunked graph
    """
    with open(output_file_path, 'w') as f:
        # Write nodes
        f.write("Nodes:\n")
        for node_id, (lon, lat) in filtered_nodes.items():
            f.write(f"{node_id}: ({lon},{lat})\n")
        
        # Write edges
        f.write("Edges:\n")
        for (node1, node2), weight in filtered_edges.items():
            f.write(f"({node1},{node2}): {weight}\n")
        
        # Write origin
        f.write("Origin:\n")
        f.write(f"{origin_id}\n")
        
        # Write destinations
        f.write("Destinations:\n")
        f.write(";".join(destination_ids) + "\n")

def get_chunk_statistics(nodes: Dict[str, Tuple[float, float]], 
                        origin_id: str, 
                        destination_id: str, 
                        margin_factor: float = 0.2) -> Dict[str, any]:
    """
    Get statistics about the chunking process.
    
    Args:
        nodes: Original nodes dictionary
        origin_id: Origin node ID
        destination_id: Destination node ID
        margin_factor: Margin factor used for chunking
        
    Returns:
        Dictionary with chunking statistics
    """
    # Get coordinates
    origin_coords = nodes[origin_id]
    dest_coords = nodes[destination_id]
    
    # Calculate direct distance
    direct_distance = haversine_distance(
        origin_coords[1], origin_coords[0],  # lat, lon
        dest_coords[1], dest_coords[0]       # lat, lon
    )
    
    # Create bounding box
    bounding_box = create_bounding_box(origin_coords, dest_coords, margin_factor)
    min_lon, min_lat, max_lon, max_lat = bounding_box
    
    # Calculate bounding box dimensions
    box_width = haversine_distance(min_lat, min_lon, min_lat, max_lon)
    box_height = haversine_distance(min_lat, min_lon, max_lat, min_lon)
    box_area = box_width * box_height  # Approximate area in m²
    
    # Count nodes in chunk
    valid_nodes = filter_nodes_by_chunk(nodes, origin_id, destination_id, margin_factor)
    
    return {
        'origin_coords': origin_coords,
        'destination_coords': dest_coords,
        'direct_distance_m': direct_distance,
        'bounding_box': bounding_box,
        'box_width_m': box_width,
        'box_height_m': box_height,
        'box_area_m2': box_area,
        'total_nodes': len(nodes),
        'chunked_nodes': len(valid_nodes),
        'reduction_percent': (1 - len(valid_nodes)/len(nodes)) * 100,
        'margin_factor': margin_factor
    }

# Example usage and testing
if __name__ == "__main__":
    # Test the chunking functionality
    import importlib.util
    
    graph_file = os.path.join('Data', 'graph.txt')
    origin = "2015"
    destination = "3629"
    
    try:
        # Create chunked graph
        filtered_nodes, filtered_edges, origin_id, dest_ids = create_chunked_graph(
            graph_file, origin, destination, margin_factor=0.2
        )
          # Get statistics using dynamic import
        search_dir = os.path.join('..', 'Search')
        parser_path = os.path.join(search_dir, 'data_reader', 'parser.py')
        
        spec = importlib.util.spec_from_file_location("parser", parser_path)
        parser_module = importlib.util.module_from_spec(spec)
        sys.path.append(search_dir)
        spec.loader.exec_module(parser_module)
        
        original_nodes, _, _, _ = parser_module.parse_graph_file(graph_file)
        
        stats = get_chunk_statistics(original_nodes, origin, destination)
        
        print("\nChunking Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        # Test writing to file
        temp_file = "temp_chunked_graph.txt"
        write_chunked_graph_to_file(filtered_nodes, filtered_edges, origin_id, dest_ids, temp_file)
        # print(f"\nChunked graph written to: {temp_file}")
        
    except Exception as e:
        print(f"Error testing chunking: {e}")