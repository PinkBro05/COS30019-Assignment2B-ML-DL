import os
import sys
from abc import abstractmethod

# Get the path to the project root by going up 2 levels
current_dir = os.path.dirname(os.path.abspath(__file__))  # Current: .../Uninformed_Search/entity
project_root = os.path.dirname(os.path.dirname(current_dir))  # Root: .../cos30019

# Now correctly navigate to the network.py file
aco_routing_dir = os.path.join(project_root, "Custom_Search", "aco_routing")
sys.path.append(aco_routing_dir)

from network import Network

class SearchNetwork(Network):
    """
    Extended Network class with common functionality for search algorithms.
    This acts as a parent class for specific search implementations.
    """
    
    def build_from_data(self, nodes, edges):
        """
        Build the network from nodes list and edges dictionary.
        
        Parameters:
            nodes - a list of node identifiers
            edges - a dictionary where keys are (src, tgt) tuples and values are weights
        """
        # Initialize nodes
        for node in nodes:
            if node not in self.graph:
                self.graph[node] = []
                
        # Process each edge
        for (src, tgt), weight in edges.items():
            self.add_edge(src, tgt, weight=weight)
            
        return self  # Return self for method chaining
    
    @abstractmethod
    def find_path(self, start, goal):
        """
        Find path from start to goal node.
        
        Returns:
            tuple: (path, weight) where path is a list of nodes and weight is the total cost
        """
        pass
    
    def find_shortest_path_to_destinations(self, origin, destinations):
        """
        Find the shortest path from origin to any of the destinations.
        
        Returns:
            tuple: (path, destination, weight) of the shortest path
        """
        shortest_path = None
        shortest_dest = None
        shortest_weight = float('inf')
        
        for dest in destinations:
            path, weight = self.find_path(origin, dest)
            if path and weight < shortest_weight:
                shortest_weight = weight
                shortest_path = path
                shortest_dest = dest
                
        return shortest_path, shortest_dest, shortest_weight