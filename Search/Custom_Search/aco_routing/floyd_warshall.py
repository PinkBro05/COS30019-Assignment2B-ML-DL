from typing import Dict, List, Tuple
import numpy as np
from aco_routing.network import Network

class FloydWarshall:
    """
    Floyd-Warshall algorithm for all-pairs shortest paths.
    This class computes the shortest paths between all pairs of nodes in a graph.
    """
    
    def __init__(self, graph: Network):
        """Initialize with a Network graph.
        
        Args:
            graph: Network graph object
        """
        self.graph = graph
        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.idx_to_node = {i: node for i, node in enumerate(self.nodes)}
        
        # Initialize distance and next matrices
        self.dist_matrix = np.full((self.n, self.n), float('inf'))
        self.next_matrix = np.full((self.n, self.n), None, dtype=object)
        
        # Fill diagonal with zeros
        for i in range(self.n):
            self.dist_matrix[i, i] = 0
        
        # Initialize with direct edges
        for u, v in self.graph.get_edges():
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            cost = self.graph.edges[(u, v)].get("cost", float('inf'))
            self.dist_matrix[i, j] = cost
            self.next_matrix[i, j] = v
    
    def run(self):
        """Run the Floyd-Warshall algorithm."""
        # Main Floyd-Warshall algorithm
        for k in range(self.n):
            for i in range(self.n):
                for j in range(self.n):
                    if self.dist_matrix[i, j] > self.dist_matrix[i, k] + self.dist_matrix[k, j]:
                        self.dist_matrix[i, j] = self.dist_matrix[i, k] + self.dist_matrix[k, j]
                        self.next_matrix[i, j] = self.next_matrix[i, k]
    
    def get_shortest_path(self, source: str, target: str) -> Tuple[List[str], float]:
        """Get the shortest path between source and target.
        
        Args:
            source: Source node
            target: Target node
            
        Returns:
            Tuple containing the path as a list of nodes and the path cost
        """
        if source not in self.node_to_idx or target not in self.node_to_idx:
            return [], float('inf')
        
        i, j = self.node_to_idx[source], self.node_to_idx[target]
        
        if self.dist_matrix[i, j] == float('inf'):
            return [], float('inf')
        
        path = [source]
        current = i
        
        while current != j:
            next_node = self.next_matrix[current, j]
            if next_node is None:
                break
            path.append(next_node)
            current = self.node_to_idx[next_node]
        
        return path, self.dist_matrix[i, j]
    
    def get_all_pairs_shortest_paths(self) -> Dict[Tuple[str, str], Tuple[List[str], float]]:
        """Get all pairs shortest paths.
        
        Returns:
            Dictionary mapping (source, target) pairs to (path, cost) tuples
        """
        result = {}
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    source = self.idx_to_node[i]
                    target = self.idx_to_node[j]
                    path, cost = self.get_shortest_path(source, target)
                    if path:
                        result[(source, target)] = (path, cost)
        
        return result
    
    def update_graph_with_shortest_paths(self):
        """
        Update the original graph with the shortest path information.
        This modifies edge costs in the original network graph.
        """
        all_paths = self.get_all_pairs_shortest_paths()
        
        for (source, target), (path, cost) in all_paths.items():
            # For each shortest path, if there isn't a direct edge between source and target,
            # add a "virtual edge" with the calculated shortest path cost
            if not self.graph.has_edge(source, target):
                self.graph.add_edge(source, target, cost=cost, is_virtual=True, path=path)
            else:
                # If there is an existing edge, update its cost if the calculated path is shorter
                current_cost = self.graph.edges[(source, target)].get("cost", float('inf'))
                if cost < current_cost:
                    self.graph.edges[(source, target)]["cost"] = cost
                    self.graph.edges[(source, target)]["is_virtual"] = True
                    self.graph.edges[(source, target)]["path"] = path