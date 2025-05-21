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