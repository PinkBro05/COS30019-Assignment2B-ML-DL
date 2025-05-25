import os
import sys
import heapq
from itertools import count

# Fix the path imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

# Import the intermediate parent class
common_dir = os.path.join(parent_dir, "Uninformed_Search", "entity")
sys.path.append(common_dir)

from SearchNetwork import SearchNetwork

class DijkstraNetwork(SearchNetwork):
    """
    Extended Network class with Dijkstra's algorithm for finding shortest paths.
    """
    
    def dijkstra(self, start, goal, debug=False):
        """
        Find the shortest path from start to goal using Dijkstra's algorithm.
        
        Parameters:
            start: Starting node
            goal: Target node
            debug: Whether to print debugging information
            
        Returns:
            tuple: (path, cost) where path is a list of nodes and cost is the total path cost
        """
        # Handle case where start and goal are the same
        if start == goal:
            return [start], 0
            
        visited = set()
        counter = count()  # tie-breaker for insertion order
        # Format: (cost, counter, node, path, step_added)
        heap = [(0, next(counter), start, [start], 0)]
        step_counter = 1
        
        if debug:
            print(f"Initial heap: {[(c, n) for c, _, n, _, _ in heap]}")
        
        while heap:
            cost, _, current, path, added_at = heapq.heappop(heap)
            
            if debug:
                print(f"\nStep {step_counter}:")
                step_counter += 1
                print(f"Popped: (cost={cost}, node={current}, path={path}) [added at step {added_at}]")
            
            # Skip if we've already visited this node
            if current in visited:
                if debug:
                    print(f"  Skipped (already visited): {current}")
                continue
            
            visited.add(current)
            
            # If we've reached the goal, return the path and cost
            if current == goal:
                if debug:
                    print(f"GOAL reached: {current} with cost {cost}")
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
                print(f"  Exploring neighbors (sorted): {neighbors}")
            
            for neighbor, edge_weight in neighbors:
                if neighbor not in visited:
                    new_cost = cost + edge_weight
                    new_path = path + [neighbor]
                    
                    if debug:
                        print(f"    → Added to heap: (cost={new_cost}, node={neighbor}, path={new_path}) [added at step {step_counter}]")
                        
                    heapq.heappush(heap, (new_cost, next(counter), neighbor, new_path, step_counter))
            
            if debug:
                print(f"  Heap now: {[(c, n) for c, _, n, _, _ in heap]}")
        
        # No path found
        return [], float('inf')
    
    def find_shortest_path_to_destinations(self, origin, destinations, debug=False):
        """
        Find the shortest path from origin to any of the destinations.
        
        Parameters:
            origin: Starting node
            destinations: List of possible target nodes
            debug: Whether to print debugging information
            
        Returns:
            tuple: (path, destination, cost) of the shortest path
        """
        shortest_path = None
        shortest_dest = None
        shortest_cost = float('inf')
        
        for dest in destinations:
            path, cost = self.find_path(origin, dest, debug)
            if path and cost < shortest_cost:
                shortest_cost = cost
                shortest_path = path
                shortest_dest = dest
                
                if debug:
                    print(f"\nNew shortest path found to {dest}: {path}")
                    print(f"Cost: {cost}")
        
        return shortest_path, shortest_dest, shortest_cost
    
    def find_path(self, start, goal, debug=False):
        """Implementation of the abstract method using Dijkstra's algorithm with optional debugging"""
        return self.dijkstra(start, goal, debug)
    
    def find_k_paths(self, start, goal, k=5):
        """
        Find k shortest paths from start to goal using Dijkstra's algorithm.
        
        Parameters:
            start: Starting node
            goal: Goal node
            k: Number of paths to find (default: 5)
            
        Returns:
            list: List of tuples (path, cost) sorted by cost
        """
        if start not in self.graph or goal not in self.graph:
            return []
            
        # List to store completed paths
        paths = []
        visited = set()
        counter = count()  # tie-breaker for insertion order
        
        # Priority queue: (cost, counter, node, path)
        heap = [(0, next(counter), start, [start])]
        
        # Add iteration counter
        iterations = 0
        
        while heap and len(paths) < k:
            # Increment iteration counter
            iterations += 1
            
            cost, _, current, path = heapq.heappop(heap)
            
            # If we've reached the goal, add to paths
            if current == goal:
                paths.append((path, cost))
                continue
                
            # Skip if we've already visited this node
            if current in visited:
                continue
                
            visited.add(current)
            
            # Get neighbors with their edge weights
            for neighbor in self.neighbors(current):
                if neighbor not in path:  # Avoid cycles
                    edge_data = self.get_edge_data(current, neighbor)
                    edge_weight = edge_data.get('weight', 1)
                    new_cost = cost + edge_weight
                    new_path = path + [neighbor]
                    
                    heapq.heappush(heap, (new_cost, next(counter), neighbor, new_path))
        
        # Print some debug info about the number of iterations
        print(f"Explored {iterations} paths, found {len(paths)} unique paths to goal")
        
        # Sort paths by cost (should already be sorted, but just to be safe)
        paths.sort(key=lambda x: x[1])
        return paths[:k]
    
    def find_k_shortest_paths_to_destinations(self, origin, destinations, k=5):
        """
        Find the k shortest paths from origin to any of the destinations.
        
        Parameters:
            origin: Starting node
            destinations: List of possible target nodes
            k: Number of paths to find (default: 5)
            
        Returns:
            list: List of tuples (path, destination, cost) sorted by cost
        """
        all_paths = []
        
        for dest in destinations:
            paths = self.find_k_paths(origin, dest, k)
            for path, cost in paths:
                all_paths.append((path, dest, cost))
                
        # Sort all paths by cost and return top k
        all_paths.sort(key=lambda x: x[2])  # Sort by cost
        return all_paths[:k]