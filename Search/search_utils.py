"""
Helper functions for using search algorithms from GUI applications.
"""
import os
import sys
import importlib.util

# Add the necessary paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "data_reader"))
sys.path.append(os.path.join(current_dir, "Informed_Search"))
sys.path.append(os.path.join(current_dir, "Uninformed_Search"))
sys.path.append(os.path.join(current_dir, "Custom_Search"))
sys.path.append(os.path.join(current_dir, "Custom_Search", "aco_routing"))
sys.path.append(os.path.join(current_dir, "Custom_Search", "Dijkstras_Algorithm"))

from parser import parse_graph_file
from Custom_Search.aco_routing.network import Network

def find_paths(graph_file_path, origin, destination, algorithm="AS", top_n=5):
    """
    Find paths between origin and destination using the specified algorithm.
    
    Args:
        graph_file_path (str): Path to the graph file
        origin (str): Origin node ID
        destination (str): Destination node ID
        algorithm (str): Algorithm to use (AS, GBFS, BFS, DFS, CUS1, CUS2)
        top_n (int): Number of top paths to return
        
    Returns:
        list: List of tuples (path, cost) sorted by cost
    """
    # Parse the graph file with the provided origin and destination
    nodes, edges, _, _ = parse_graph_file(graph_file_path, origin=origin, destinations=[destination])
    
    # Create the graph structure
    G = Network()
    G.graph = {node: [] for node in nodes}
    G.pos = nodes  # Store node positions
    
    # Add edges
    for (start, end), weight in edges.items():
        G.add_edge(start, end, cost=float(weight))
    
    # Initialize results
    path_results = []
    
    if algorithm == "AS" or algorithm == "ASTAR":
        # Import A* algorithm
        module_path = os.path.join(current_dir, "Informed_Search", "A_Star.py")
        spec = importlib.util.spec_from_file_location("A_Star", module_path)
        a_star_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(a_star_module)
        
        # Run A* algorithm
        path = a_star_module.a_star(G.graph, nodes, origin, destination, edges)
        
        # Calculate path cost
        cost = 0
        if len(path) > 1:
            for i in range(len(path) - 1):
                cost += edges.get((path[i], path[i+1]), float('inf'))
            
            path_results.append((path, cost))
    
    elif algorithm == "GBFS":
        # Import GBFS algorithm
        module_path = os.path.join(current_dir, "Informed_Search", "GBFS.py")
        spec = importlib.util.spec_from_file_location("GBFS", module_path)
        gbfs_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gbfs_module)
        
        # Run GBFS algorithm
        path = gbfs_module.greedy_best_first_search(G.graph, nodes, origin, destination, edges)
        
        # Calculate path cost
        cost = 0
        if len(path) > 1:
            for i in range(len(path) - 1):
                cost += edges.get((path[i], path[i+1]), float('inf'))
            
            path_results.append((path, cost))
    
    elif algorithm == "BFS":
        # Import BFS algorithm
        module_path = os.path.join(current_dir, "Uninformed_Search", "bfs.py")
        spec = importlib.util.spec_from_file_location("bfs", module_path)
        bfs_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bfs_module)
        
        # Get BfsNetwork class
        bfs_network_path = os.path.join(current_dir, "Uninformed_Search", "entity", "BfsNetwork.py")
        spec = importlib.util.spec_from_file_location("BfsNetwork", bfs_network_path)
        bfs_network_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bfs_network_module)
        
        # Convert to BFS network format
        bfs_network = bfs_network_module.BfsNetwork(G)
        
        # Run BFS algorithm
        path = bfs_module.bfs(bfs_network, origin, destination)
        
        # Calculate path cost
        cost = 0
        if len(path) > 1:
            for i in range(len(path) - 1):
                cost += edges.get((path[i], path[i+1]), float('inf'))
            
            path_results.append((path, cost))
    
    elif algorithm == "DFS":
        # Import DFS algorithm
        module_path = os.path.join(current_dir, "Uninformed_Search", "dfs.py")
        spec = importlib.util.spec_from_file_location("dfs", module_path)
        dfs_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dfs_module)
        
        # Get DfsNetwork class
        dfs_network_path = os.path.join(current_dir, "Uninformed_Search", "entity", "DfsNetwork.py")
        spec = importlib.util.spec_from_file_location("DfsNetwork", dfs_network_path)
        dfs_network_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dfs_network_module)
        
        # Convert to DFS network format
        dfs_network = dfs_network_module.DfsNetwork(G)
        
        # Run DFS algorithm
        path = dfs_module.dfs(dfs_network, origin, destination)
        
        # Calculate path cost
        cost = 0
        if len(path) > 1:
            for i in range(len(path) - 1):
                cost += edges.get((path[i], path[i+1]), float('inf'))
            
            path_results.append((path, cost))
    
    elif algorithm == "CUS1":  # Dijkstra's
        # Import Dijkstra's algorithm
        module_path = os.path.join(current_dir, "Custom_Search", "Dijkstras_Algorithm", "dijk.py")
        spec = importlib.util.spec_from_file_location("dijk", module_path)
        dijk_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dijk_module)
        
        # Run Dijkstra's algorithm and get results
        dijk_network = dijk_module.convert_to_dijk_format(G)
        paths = dijk_module.dijkstra_all_paths(dijk_network, origin, destination, top_n)
        
        # Process all returned paths
        for path, cost in paths:
            path_results.append((path, cost))
    
    elif algorithm == "CUS2":  # ACO
        # Import ACO algorithm components
        from Custom_Search.aco_routing.aco import ACO
        
        # Set up ACO parameters
        node_count = len(G.graph)
        ant_max_steps = node_count + 1
        iterations = 50  # Reduced for GUI response time
        num_ants = min(node_count, 30)  # Limit the number of ants for performance
        
        # Create ACO instance
        aco = ACO(
            graph=G,
            ant_max_steps=ant_max_steps,
            num_iterations=iterations,
            num_ants=num_ants,
            alpha=1.0,
            beta=2.0,
            evaporation_rate=0.5,
            mode=0,  # Find path to destination
            visualize=False
        )
        
        # Run multiple times to get different paths
        for _ in range(top_n):
            # Find path using ACO
            aco_path, aco_cost = aco.find_path(origin, destination)
            
            if aco_path:
                path_results.append((aco_path, aco_cost))
    
    # Sort the results by cost
    path_results.sort(key=lambda x: x[1])
    
    # Return the top N paths
    return path_results[:top_n]

def highlight_path_on_map(m, path_nodes, node_positions, edges, color='red', width=4):
    """
    Highlight a path on the map.
    
    Args:
        m (folium.Map): The map object
        path_nodes (list): List of node IDs in the path
        node_positions (dict): Dictionary mapping node IDs to (lon, lat)
        edges (dict): Dictionary mapping (node1, node2) to edge weight
        color (str): Color of the path
        width (int): Width of the path line
        
    Returns:
        folium.FeatureGroup: The feature group containing the path
    """
    import folium
    
    # Create a feature group for the path
    path_group = folium.FeatureGroup(name=f"Path (Cost: {sum(edges.get((path_nodes[i], path_nodes[i+1]), 0) for i in range(len(path_nodes)-1))})")
    
    # Add path segments
    for i in range(len(path_nodes) - 1):
        start_node = path_nodes[i]
        end_node = path_nodes[i+1]
        
        # Get node coordinates (lon, lat)
        start_pos = node_positions[start_node]
        end_pos = node_positions[end_node]
        
        # Get edge weight
        weight = edges.get((start_node, end_node), 0)
        
        # Add line
        folium.PolyLine(
            locations=[[start_pos[1], start_pos[0]], [end_pos[1], end_pos[0]]],
            color=color,
            weight=width,
            opacity=0.8,
            tooltip=f"Distance: {weight}m"
        ).add_to(path_group)
        
        # Add markers for each node in the path
        folium.CircleMarker(
            location=[start_pos[1], start_pos[0]],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=1.0,
            tooltip=f"Node: {start_node}"
        ).add_to(path_group)
    
    # Add marker for the last node
    last_node = path_nodes[-1]
    last_pos = node_positions[last_node]
    folium.CircleMarker(
        location=[last_pos[1], last_pos[0]],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=1.0,
        tooltip=f"Node: {last_node}"
    ).add_to(path_group)
    
    # Add the path group to the map
    path_group.add_to(m)
    
    return path_group
