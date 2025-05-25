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
    
    def find_k_paths(self, start, goal, k=5):
        """
        Find k shortest paths using an iterative deepening approach.
        
        Parameters:
            start: starting node
            goal: goal node
            k: number of paths to find (default: 5)
            
        Returns:
            list: List of tuples (path, weight) sorted by weight
        """
        if start not in self.graph or goal not in self.graph:
            return []
        
        # Use a max heap for the k shortest paths
        import heapq
        all_paths = []  # List of (cost, path) tuples
        max_cost = float('inf')
        
        # Use a more efficient approach with a priority queue
        # Queue format: (cost, node, path, visited_set)
        queue = [(0, start, [start], {start})]
        
        # Track visited paths for pruning
        visited_paths = set()
        
        # Set a limit on number of iterations to avoid too long processing
        iteration_limit = 100000
        iterations = 0
        
        while queue and iterations < iteration_limit:
            iterations += 1
            
            current_cost, current_node, current_path, visited = heapq.heappop(queue)
            
            # If we've already found k paths and this path's cost exceeds the longest path we have,
            # we can safely ignore it as it won't be in our top k
            if len(all_paths) >= k and current_cost > -all_paths[0][0]:
                continue
                
            # Skip if we've seen this path configuration before
            path_key = (current_node, tuple(sorted(visited)))
            if path_key in visited_paths:
                continue
            visited_paths.add(path_key)
            
            # If we reached the goal, add this path to our collection
            if current_node == goal:
                if len(all_paths) < k:
                    heapq.heappush(all_paths, (-current_cost, current_path.copy(), current_cost))
                elif current_cost < -all_paths[0][0]:
                    # Replace the worst path if this one is better
                    heapq.heappop(all_paths)
                    heapq.heappush(all_paths, (-current_cost, current_path.copy(), current_cost))
                continue
            
            # Process neighbors in sorted order (for consistent DFS behavior)
            neighbors = []
            for neighbor in self.neighbors(current_node):
                if neighbor not in visited:  # Avoid cycles
                    edge_data = self.get_edge_data(current_node, neighbor)
                    weight = edge_data.get('weight', 1)
                    neighbors.append((neighbor, weight))
            
            # We no longer need to sort in reverse order since we're using a priority queue
            neighbors.sort(key=lambda x: str(x[0]))
            
            # Add neighbors to queue
            for neighbor, edge_weight in neighbors:
                new_path = current_path + [neighbor]
                new_cost = current_cost + edge_weight
                
                # Only add paths that have potential to be among the k shortest
                if len(all_paths) < k or new_cost < -all_paths[0][0]:
                    new_visited = visited.copy()
                    new_visited.add(neighbor)
                    heapq.heappush(queue, (new_cost, neighbor, new_path, new_visited))
        
        # Print some debug info about the number of iterations
        print(f"Explored {iterations} paths, found {len(all_paths)} unique paths to goal")
        
        # Extract the results in the correct format
        result_paths = [(path, cost) for _, path, cost in sorted(all_paths, key=lambda x: x[2])]
        return result_paths