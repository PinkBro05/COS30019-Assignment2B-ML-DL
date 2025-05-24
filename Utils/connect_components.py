"""
Functions to connect disconnected components in a graph using multiple strategies
to ensure complete graph connectivity for path finding.
"""

import networkx as nx
import numpy as np
from haversine import haversine, Unit
from tqdm import tqdm
import random

def connect_components(scats_graph, edge_distances, max_fallback_distance=5, force_connect=True, max_attempts=3):
    """
    Connect disconnected components in the graph using a multi-strategy approach.
    
    Args:
        scats_graph (networkx.Graph): The current graph with disconnected components
        edge_distances (dict): Dictionary of current edge distances
        max_fallback_distance (float): Maximum distance in km for connecting components
        force_connect (bool): If True, will ensure graph is connected even if it means adding
                              longer-distance edges
        max_attempts (int): Maximum number of connection attempts with increasingly relaxed constraints
        
    Returns:
        networkx.Graph: Connected graph
        dict: Updated edge distances dictionary
    """
    if nx.is_connected(scats_graph):
        print("Graph is already connected")
        return scats_graph, edge_distances
    
    original_graph = scats_graph.copy()
    
    # First attempt: Try MST-based connection with given max_fallback_distance
    print(f"Attempt 1: Connecting components using MST approach (max distance: {max_fallback_distance} km)...")
    scats_graph, edge_distances = connect_with_mst(
        scats_graph, edge_distances, max_fallback_distance)
    
    # If graph is connected or we don't want to force connections, return
    if nx.is_connected(scats_graph) or not force_connect:
        if nx.is_connected(scats_graph):
            print("Successfully connected graph after first attempt")
        return scats_graph, edge_distances
    # Second attempt: Try with a more flexible distance constraint
    relaxed_distance = max_fallback_distance * 2
    print(f"Attempt 2: Graph still disconnected, trying with increased distance ({relaxed_distance} km)...")
    scats_graph, edge_distances = connect_with_mst(
        scats_graph, edge_distances, relaxed_distance)
    
    if nx.is_connected(scats_graph):
        print("Successfully connected graph after second attempt")
        return scats_graph, edge_distances
    
    # Third attempt: Use the k-nearest neighbor approach to guarantee connectivity
    print("Attempt 3: Using k-nearest neighbors approach to connect remaining components...")
    scats_graph, edge_distances = connect_with_knn(
        scats_graph, edge_distances, k=3)
    
    if nx.is_connected(scats_graph):
        print("Successfully connected graph after k-nearest neighbors approach")
        return scats_graph, edge_distances
    
    # Final attempt: Connect all remaining components forcefully
    print("Final attempt: Forcefully connecting all remaining components...")
    scats_graph, edge_distances = connect_remaining_forcefully(scats_graph, edge_distances)
    
    # Verify final connectivity
    if nx.is_connected(scats_graph):
        print("Graph is now fully connected")
    else:
        # This should never happen with force_connect=True
        print("Warning: Graph is still not fully connected despite force connection")
        # Last resort: create a direct connection between known problematic nodes
        scats_graph, edge_distances = ensure_direct_connections(scats_graph, edge_distances)
    
    return scats_graph, edge_distances

def connect_with_mst(scats_graph, edge_distances, max_distance_km):
    """
    Connect components using a Minimum Spanning Tree approach with distance constraints
    
    Args:
        scats_graph (networkx.Graph): The graph to connect
        edge_distances (dict): Current edge distances
        max_distance_km (float): Maximum distance in km for connecting components
        
    Returns:
        networkx.Graph: Updated graph
        dict: Updated edge distances
    """
    # Get the current connected components
    components = list(nx.connected_components(scats_graph))
    print(f"Found {len(components)} disconnected components")
    
    # If only one component, graph is already connected
    if len(components) == 1:
        return scats_graph, edge_distances
    
    # Create a mapping of nodes to their component index
    component_map = {}
    for i, component in enumerate(components):
        for node in component:
            component_map[node] = i
    
    # For each component, find possible connections to other components
    component_bridges = []
    
    for node1, node1_data in tqdm(scats_graph.nodes(data=True), desc="Finding bridges between components"):
        comp1 = component_map[node1]
        for node2, node2_data in scats_graph.nodes(data=True):
            comp2 = component_map[node2]
            
            # Only process nodes from different components
            if comp1 >= comp2:
                continue
                
            # Calculate distance
            lat1, lon1 = node1_data['latitude'], node1_data['longitude']
            lat2, lon2 = node2_data['latitude'], node2_data['longitude']
            
            distance = haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)
            
            # Only consider connections within the fallback distance
            if distance <= max_distance_km * 1000:  # Convert to meters
                component_bridges.append((node1, node2, distance, comp1, comp2))
    
    if not component_bridges:
        print(f"No bridges found within {max_distance_km} km. Consider increasing the distance threshold.")
        return scats_graph, edge_distances
    
    # Sort bridges by distance
    component_bridges.sort(key=lambda x: x[2])
    
    # Use a Union-Find data structure to track connected components
    parent = {i: i for i in range(len(components))}
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        parent[find(x)] = find(y)
    
    # Add minimal set of edges to connect components
    bridges_added = 0
    for node1, node2, distance, comp1, comp2 in component_bridges:
        if find(comp1) != find(comp2):
            # Add this edge to connect components
            scats_graph.add_edge(node1, node2, weight=distance)
            edge_distances[(node1, node2)] = int(distance)
            edge_distances[(node2, node1)] = int(distance)
            
            # Merge components
            union(comp1, comp2)
            bridges_added += 1
    
    # Count how many connected components remain
    remaining_roots = set()
    for i in range(len(components)):
        remaining_roots.add(find(i))
    
    print(f"Added {bridges_added} bridges to connect components")
    print(f"Reduced from {len(components)} to {len(remaining_roots)} connected components")
    
    return scats_graph, edge_distances

def connect_remaining_forcefully(scats_graph, edge_distances):
    """
    Force connection of all remaining components by connecting to nearest nodes,
    regardless of distance
    
    Args:
        scats_graph (networkx.Graph): The graph with disconnected components
        edge_distances (dict): Current edge distances
        
    Returns:
        networkx.Graph: Connected graph
        dict: Updated edge distances
    """
    # Get remaining components
    components = list(nx.connected_components(scats_graph))
    print(f"Forcefully connecting {len(components)} remaining components")
    
    if len(components) <= 1:
        return scats_graph, edge_distances
    
    # Sort components by size (largest first)
    components.sort(key=len, reverse=True)
    
    # Get the largest component
    largest_component = components[0]
    
    # For each smaller component, connect it to the largest component
    for i, component in enumerate(components[1:], 1):
        print(f"Connecting component {i} (size: {len(component)}) to main component")
        
        # Find the closest node pair between this component and the largest component
        min_distance = float('inf')
        closest_pair = None
        
        # Sample nodes if components are large to speed up processing
        large_component_nodes = largest_component
        if len(largest_component) > 100:
            large_component_nodes = list(largest_component)
            large_component_nodes = np.random.choice(large_component_nodes, 100, replace=False)
            
        current_component_nodes = component
        if len(component) > 100:
            current_component_nodes = list(component)
            current_component_nodes = np.random.choice(current_component_nodes, 100, replace=False)
        
        # Find the closest node pair
        for node1 in tqdm(current_component_nodes, desc=f"Finding connection for component {i}"):
            node1_data = scats_graph.nodes[node1]
            lat1, lon1 = node1_data['latitude'], node1_data['longitude']
            
            for node2 in large_component_nodes:
                node2_data = scats_graph.nodes[node2]
                lat2, lon2 = node2_data['latitude'], node2_data['longitude']
                
                distance = haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_pair = (node1, node2, distance)
        
        if closest_pair:
            node1, node2, distance = closest_pair
            # Add connection
            scats_graph.add_edge(node1, node2, weight=distance)
            edge_distances[(node1, node2)] = int(distance)
            edge_distances[(node2, node1)] = int(distance)
            print(f"  Added connection between nodes {node1} and {node2} (distance: {distance/1000:.2f} km)")
            
            # Update largest component to include the newly connected component
            largest_component = set(largest_component) | set(component)
    
    # Final verification
    if nx.is_connected(scats_graph):
        print("Graph is now fully connected after forceful connection")
    else:
        # This should never happen
        print("Warning: Graph still has disconnected components after forceful connection")
        components = list(nx.connected_components(scats_graph))
        print(f"Remaining components: {len(components)}")
        
        # Last resort: add edges between ALL components
        connect_all_components_brute_force(scats_graph, edge_distances, components)
    
    return scats_graph, edge_distances

def connect_all_components_brute_force(scats_graph, edge_distances, components):
    """
    Absolutely ensure connectivity by connecting all components with brute force.
    This is a last resort method.
    
    Args:
        scats_graph (networkx.Graph): The graph
        edge_distances (dict): Edge distances
        components (list): List of components
    """
    print("!!! APPLYING BRUTE FORCE CONNECTION - THIS IS A LAST RESORT !!!")
    
    # Extract one representative node from each component
    component_representatives = [list(component)[0] for component in components]
    
    # Connect them all in a star pattern to the first component's representative
    center_node = component_representatives[0]
    center_data = scats_graph.nodes[center_node]
    center_lat, center_lon = center_data['latitude'], center_data['longitude']
    
    for i, node in enumerate(component_representatives[1:], 1):
        node_data = scats_graph.nodes[node]
        node_lat, node_lon = node_data['latitude'], node_data['longitude']
        
        # Calculate distance
        distance = haversine((center_lat, center_lon), (node_lat, node_lon), unit=Unit.METERS)
        
        # Add edge
        scats_graph.add_edge(center_node, node, weight=distance)
        edge_distances[(center_node, node)] = int(distance)
        edge_distances[(node, center_node)] = int(distance)
        
        print(f"  Emergency connection added between components 0 and {i}")
    
    if nx.is_connected(scats_graph):
        print("Graph is now fully connected after brute force connection")
    else:
        print("CRITICAL ERROR: Graph still not connected after all attempts!")
        # This should be impossible
    
    return scats_graph, edge_distances

def connect_with_knn(scats_graph, edge_distances, k=3):
    """
    Connect components using k-nearest neighbors approach for each component.
    This creates multiple connections between components for better connectivity.
    
    Args:
        scats_graph (networkx.Graph): The graph to connect
        edge_distances (dict): Current edge distances
        k (int): Number of nearest neighbors to connect from each component
        
    Returns:
        networkx.Graph: Updated graph
        dict: Updated edge distances
    """
    # Get the current connected components
    components = list(nx.connected_components(scats_graph))
    print(f"Using k-nearest neighbors approach with k={k} for {len(components)} components")
    
    if len(components) <= 1:
        return scats_graph, edge_distances
        
    # Sort components by size
    components.sort(key=len, reverse=True)
    
    # Create a mapping of nodes to their component
    component_map = {}
    for i, component in enumerate(components):
        for node in component:
            component_map[node] = i
    
    # Process each component
    for comp_idx, component in enumerate(components):
        print(f"Processing component {comp_idx+1}/{len(components)} (size: {len(component)})")
        
        # For smaller components, include all nodes
        sample_size = min(len(component), 10)
        sampled_nodes = random.sample(list(component), sample_size)
        
        for node1 in sampled_nodes:
            node1_data = scats_graph.nodes[node1]
            lat1, lon1 = node1_data['latitude'], node1_data['longitude']
            
            # Find candidates from other components
            candidates = []
            for node2, node2_data in scats_graph.nodes(data=True):
                if component_map[node2] == comp_idx:
                    continue  # Skip nodes in the same component
                
                lat2, lon2 = node2_data['latitude'], node2_data['longitude']
                distance = haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)
                candidates.append((node2, distance))
            
            # Sort candidates by distance and connect to k nearest
            candidates.sort(key=lambda x: x[1])
            for node2, distance in candidates[:k]:
                if not scats_graph.has_edge(node1, node2):
                    scats_graph.add_edge(node1, node2, weight=distance)
                    edge_distances[(node1, node2)] = int(distance)
                    edge_distances[(node2, node1)] = int(distance)
    
    # Check if the graph is now connected
    if nx.is_connected(scats_graph):
        print("Graph successfully connected using k-nearest neighbors approach")
    else:
        remaining = list(nx.connected_components(scats_graph))
        print(f"Graph still has {len(remaining)} components after k-nearest neighbors")
    
    return scats_graph, edge_distances

def ensure_direct_connections(scats_graph, edge_distances):
    """
    Ensure that specific problematic nodes are directly connected.
    This is a last-resort method for handling known problematic node pairs.
    
    Args:
        scats_graph (networkx.Graph): The graph to modify
        edge_distances (dict): Current edge distances
        
    Returns:
        networkx.Graph: Updated graph
        dict: Updated edge distances
    """
    print("Ensuring direct connections between potentially problematic nodes...")
    
    # List of known problematic node pairs (add more if needed)
    problem_pairs = [(4974, 1767)]
    
    for node1, node2 in problem_pairs:
        # Check if both nodes exist in the graph
        if node1 not in scats_graph.nodes or node2 not in scats_graph.nodes:
            print(f"Warning: One or both nodes ({node1}, {node2}) not in graph")
            continue
            
        # Check if there's already a path between them
        if nx.has_path(scats_graph, node1, node2):
            print(f"Nodes {node1} and {node2} are already connected")
            continue
            
        # Connect them directly
        node1_data = scats_graph.nodes[node1]
        node2_data = scats_graph.nodes[node2]
        
        lat1, lon1 = node1_data['latitude'], node1_data['longitude']
        lat2, lon2 = node2_data['latitude'], node2_data['longitude']
        
        distance = haversine((lat1, lon1), (lat2, lon2), unit=Unit.METERS)
        
        # Add direct edge
        scats_graph.add_edge(node1, node2, weight=distance)
        edge_distances[(node1, node2)] = int(distance)
        edge_distances[(node2, node1)] = int(distance)
        
        print(f"Added direct connection between nodes {node1} and {node2} (distance: {distance/1000:.2f} km)")
    
    return scats_graph, edge_distances
