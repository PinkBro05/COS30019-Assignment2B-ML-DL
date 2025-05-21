import os
import sys

# Get the path to the parent directory
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the path to the common search network class
common_dir = os.path.join(parent_dir, "Custom_Search", "Dijkstras_Algorithm")
sys.path.append(common_dir)

# Import the intermediate parent class
from SearchNetwork import SearchNetwork

class DfsNetwork(SearchNetwork):
    """
    Extended Network class with DFS functionalities for path finding.
    Follows the requirements for node expansion order.
    """
    
    def dfs_traverse(self, start):
        """
        Perform a DFS traversal from the start node.
        
        Returns:
            List of nodes in DFS traversal order
        """
        visited = set()
        traversal = []
        stack = [start]
        
        while stack:
            node = stack.pop()
            
            if node not in visited:
                visited.add(node)
                traversal.append(node)
                
                # Get neighbors and sort them in reverse order for stack
                # This ensures they are processed in ascending order when popped
                neighbors = sorted(self.neighbors(node), reverse=True)
                
                for neighbor in neighbors:
                    if neighbor not in visited:
                        stack.append(neighbor)
                    
        return traversal
    
    def dfs_path(self, start, goal, debug=False):
        """
        Find a path from start to goal using DFS.
        
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
        stack = [(start, [start], 0, 0)]
        step_counter = 1
        
        if debug:
            print(f"Initial stack: {[(n, p) for n, p, _, _ in stack]}")
        
        while stack:
            current, path, cost, added_at = stack.pop()
            
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
            
            # Sort neighbors in REVERSE order because we're using a stack (LIFO)
            # This ensures smaller nodes are popped first (ascending order)
            neighbors.sort(key=lambda x: str(x[0]), reverse=True)
            
            if debug:
                # Show neighbors in the order they will be processed (reversed for display)
                print(f"Exploring neighbors (sorted): {[n for n, _ in reversed(neighbors)]}")
            
            for neighbor, edge_weight in neighbors:
                if neighbor not in visited:
                    new_path = path + [neighbor]
                    new_cost = cost + edge_weight
                    
                    if debug:
                        print(f"    → Adding to stack: ({neighbor}, {new_path}) [added at step {step_counter}]")
                    
                    stack.append((neighbor, new_path, new_cost, step_counter))
            
            if debug:
                print(f"Stack after expansion: {[(n, p) for n, p, _, _ in stack]}")
        
        # No path found
        return [], float('inf')
    
    def find_path(self, start, goal, debug=False):
        """Implementation of the abstract method using DFS with optional debugging"""
        return self.dfs_path(start, goal, debug)