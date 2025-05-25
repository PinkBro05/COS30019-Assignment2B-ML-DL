import os
import sys
from collections import deque

# Get the path to the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the path to the common search network class
common_dir = os.path.join(parent_dir, "Custom_Search", "Dijkstras_Algorithm")
sys.path.append(common_dir)

# Import the intermediate parent class
from SearchNetwork import SearchNetwork

class BfsNetwork(SearchNetwork):
    """
    Extended Network class with BFS functionalities for path finding.
    Follows the requirements for node expansion order.
    """
    
    def bfs_traverse(self, start):
        """
        Perform a BFS traversal from the start node.
        
        Returns:
            List of nodes in BFS traversal order
        """
        visited = set()
        queue = deque([start])
        visited.add(start)
        traversal = []
        
        while queue:
            node = queue.popleft()
            traversal.append(node)
            
            # Get neighbors and sort them in ascending order
            neighbors = sorted(self.neighbors(node))
            
            for neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        return traversal
    
    def bfs_path(self, start, goal, debug=False):
        """
        Find the shortest path from start to goal using BFS.
        
        Parameters:
            start: Starting node
            goal: Target node
            debug: Whether to print debugging information
            
        Returns:
            tuple: (path, weight) where path is a list of nodes and weight is the total path weight.
                   If no path is found, returns ([], float('inf')).
        """
        if start == goal:
            return [start], 0
            
        visited = set()
        # Format: (node, path, cost, step_added)
        queue = deque([(start, [start], 0, 0)])
        step_counter = 1
        
        if debug:
            print(f"Initial queue: {[(n, p) for n, p, _, _ in queue]}")
        
        while queue:
            current, path, cost, added_at = queue.popleft()
            
            if debug:
                print(f"\nStep {step_counter}:")
                step_counter += 1
                print(f"Popped: ({current}, {path}) [added at step {added_at}]")
            
            if current in visited:
                if debug:
                    print(f"Skipped (already visited): {current}")
                continue
            
            visited.add(current)
            
            if current == goal:
                if debug:
                    print(f"GOAL reached: {current}")
                return path, cost
            
            # Get neighbors with their edge weights
            neighbors = []
            for neighbor in self.neighbors(current):
                edge_data = self.get_edge_data(current, neighbor)
                edge_weight = edge_data.get('weight', 1)
                neighbors.append((neighbor, edge_weight))
            
            # Sort neighbors in ascending order
            neighbors.sort(key=lambda x: str(x[0]))
            
            if debug:
                print(f"Exploring neighbors (sorted): {[n for n, _ in neighbors]}")
            
            for neighbor, edge_weight in neighbors:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_cost = cost + edge_weight
                    
                    if debug:
                        print(f"    → Adding to queue: ({neighbor}, {new_path}) [added at step {step_counter}]")
                    
                    queue.append((neighbor, new_path, new_cost, step_counter))
            
            if debug:
                print(f"Queue after expansion: {[(n, p) for n, p, _, _ in queue]}")
        
        # No path found
        return [], float('inf')
    
    def find_path(self, start, goal, debug=False):
        """Implementation of the abstract method using BFS with optional debugging"""
        return self.bfs_path(start, goal, debug)
    
    def find_k_paths(self, start, goal, k=5):
        """
        Find k shortest paths using a modified BFS approach.
        """
        if start not in self.graph or goal not in self.graph:
            return []
        
        # Priority queue: (cost, path)
        paths = []
        queue = deque([(0, [start])])
        visited_paths = set()
        
        # Add iteration counter
        iterations = 0
        
        while queue and len(paths) < k:
            # Increment iteration counter
            iterations += 1
            
            current_cost, current_path = queue.popleft()
            current_node = current_path[-1]
            
            # Convert path to tuple for hashing
            path_tuple = tuple(current_path)
            if path_tuple in visited_paths:
                continue
            visited_paths.add(path_tuple)
            
            if current_node == goal:
                paths.append((current_path.copy(), current_cost))
                continue
            
            # Explore neighbors using the correct method
            for neighbor in self.neighbors(current_node):
                if neighbor not in current_path:  # Avoid cycles
                    edge_data = self.get_edge_data(current_node, neighbor)
                    weight = edge_data.get('weight', 1)
                    new_path = current_path + [neighbor]
                    new_cost = current_cost + weight
                    queue.append((new_cost, new_path))
        
        # Print some debug info about the number of iterations
        print(f"Explored {iterations} paths, found {len(paths)} unique paths to goal")
        
        # Sort paths by cost
        paths.sort(key=lambda x: x[1])
        return paths[:k]