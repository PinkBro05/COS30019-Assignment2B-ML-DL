import random
from typing import List, Tuple
import os
import sys
import matplotlib.pyplot as plt
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import paths setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from aco_routing.ant import Ant
from aco_routing.graph_api import GraphApi
from aco_routing.network import Network
from aco_routing.aco_visualizer import ACOVisualizer  # Import the new visualizer
from aco_routing.floyd_warshall import FloydWarshall  # Import the Floyd-Warshall algorithm

class ACO:
    def __init__(
        self,
        graph: Network,
        ant_max_steps: int,
        num_iterations: int,
        evaporation_rate: float = 0.1,
        alpha: float = 0.7,
        beta: float = 0.3,
        mode: int = 0,
        min_scaling_factor: float = 0.01,
        log_step: int = None,
        visualize: bool = False,
        visualization_step: int = 1,
        use_floyd_warshall: bool = False,  # New parameter for Floyd-Warshall preprocessing
        use_local_search: bool = True,     # Enable local search optimization
        local_search_frequency: int = 5,   # Apply local search every N iterations
        num_threads: int = None            # Number of threads for parallel processing
    ):
        """Initialize the ACO (Ant Colony Optimization) algorithm.
        
        Args:
            graph: Network object containing the graph structure
            ant_max_steps: Maximum number of steps an ant is allowed to take
            num_iterations: Number of cycles/waves of search ants to be deployed
            evaporation_rate: Rate at which pheromones evaporate (0-1)
            alpha: Pheromone bias (importance of pheromone trails)
            beta: Edge cost bias (importance of shorter paths)
            mode: Search mode (0: find any destination, 1: all destinations, 2: TSP mode)
            min_scaling_factor: Minimum pheromone scaling factor (0-1)
            log_step: Number of iterations between logs
            visualize: Whether to visualize the algorithm progress
            visualization_step: Frequency of visualization updates
            use_floyd_warshall: Whether to preprocess the graph using Floyd-Warshall algorithm
            use_local_search: Whether to apply local search optimization
            local_search_frequency: Apply local search every N iterations
            num_threads: Number of threads to use for parallel processing
        """
        # Store all parameters
        self.graph = graph
        self.ant_max_steps = ant_max_steps
        self.num_iterations = num_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.min_scaling_factor = min_scaling_factor
        self.log_step = log_step
        self.visualize = visualize
        self.visualization_step = visualization_step
        if mode == 0:
            self.use_floyd_warshall = True # Auto trigger for any destination mode
        else:
            self.use_floyd_warshall = use_floyd_warshall
        self.use_local_search = use_local_search
        self.local_search_frequency = local_search_frequency
        self.num_threads = num_threads if num_threads else min(multiprocessing.cpu_count(), 32)
        
        # Initialize other fields
        self.search_ants = []
        self.best_path = []
        self.best_path_cost = float("inf")
        
        # Initialize gradient descent parameters
        self.gt = 0.0 # Gradient
        self.acc = 0.0 # Accumulated gradient
        self.d_acc = 0.0 # Delta accumulated gradient
        
        # Preprocess the graph with Floyd-Warshall if requested
        if self.use_floyd_warshall:
            self._preprocess_with_floyd_warshall()
        
        # Initialize the Graph API
        self.graph_api = GraphApi(self.graph, self.evaporation_rate)
        
        # Initialize visualization if needed
        self.visualizer = None
        if self.visualize:
            self.visualizer = ACOVisualizer(self.graph_api)
        
        # Initialize all edges of the graph with a stochastic pheromone value and delta pheromone of 0.0
        max_temp = 1.0
        min_temp = max_temp * self.min_scaling_factor
        k = 0.5 # control distribution
        for u, v in self.graph.get_edges():
            r = random.uniform(min_temp, max_temp) # Random pheromone value between 0.1 and 1.0
            self.graph_api.set_edge_pheromones(u, v, max_temp - k * r * (max_temp-min_temp)) # Stochastic pheromone value
            self.graph_api.set_edge_delta_pheromones(u, v, 0.0)
            
    def _preprocess_with_floyd_warshall(self):
        """Preprocess the graph using the Floyd-Warshall algorithm."""
        # if self.log_step is not None:
        #     print("Preprocessing graph with Floyd-Warshall algorithm...")
        
        # Create a FloydWarshall instance
        fw = FloydWarshall(self.graph)
        
        # Run the algorithm
        fw.run()
        
        # Update the graph with shortest paths
        fw.update_graph_with_shortest_paths()
        
        # if self.log_step is not None:
        #     print(f"Floyd-Warshall preprocessing complete. Graph now has {self.graph.number_of_edges()} edges.")
            
    def process_ant(self, ant):
        """Process a single ant for thread-based parallel execution.
        
        Args:
            ant: The ant to process
            
        Returns:
            Tuple of (ant, best_cost, best_path)
        """
        best_cost = float("inf")
        best_path = None
        
        # Process ant until it reaches destination or max steps
        for _ in range(self.ant_max_steps):
            if ant.reached_destination():
                ant.is_fit = True
                
                if ant.path_cost == 0:
                    return (ant, 0, ant.path.copy())
                
                best_cost = ant.path_cost
                best_path = ant.path.copy()
                break
                
            ant.take_step()
        
        return (ant, best_cost, best_path)

    def _deploy_forward_search_ants(self) -> float:
        """Process ants using thread-based parallelization for better performance"""
        iteration_best_path_cost = float("inf")
        
        # Use ThreadPoolExecutor for efficient thread-based parallelization
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Map requires a function that takes a single parameter, so we need to use a lambda or partial
            futures = [executor.submit(self.process_ant, ant) for ant in self.search_ants]
            
            for future in as_completed(futures):
                try:
                    ant, path_cost, path = future.result()
                    if path is not None and path_cost < float("inf"):
                        if path_cost == 0:
                            self.best_path_cost = 0
                            self.best_path = path
                            return 0
                        
                        if path_cost <= self.best_path_cost:
                            self.best_path = path
                            self.best_path_cost = path_cost
                            
                        if path_cost <= iteration_best_path_cost:
                            iteration_best_path_cost = path_cost
                except Exception as e:
                    print(f"Error processing ant: {e}")
        
        return iteration_best_path_cost
            
    def _deploy_backward_search_ants(self, iteration, iteration_best_path_cost) -> (float, float):
        for ant in self.search_ants:
            if ant.is_fit and ant.path_cost <= iteration_best_path_cost:
                # Max Min Ant System (MMAS) pheromone update
                if float(iteration) / float(self.num_iterations) < 0.75:
                    ant.deposit_pheromones_on_path(elitist_param = 0)
                    if ant.path_cost == self.best_path_cost:
                        ant.deposit_pheromones_on_path(elitist_param = 0.2)
                else:
                    if ant.path_cost == self.best_path_cost:
                        ant.deposit_pheromones_on_path(elitist_param = 0.2)
                    self.graph_api.deposit_pheromones_for_path(self.best_path)
        max_pheromone = self.graph_api.pheromone_deposit_weight/self.best_path_cost
        min_pheromone = self.min_scaling_factor * max_pheromone
        return max_pheromone, min_pheromone
                
    def _deploy_search_ants(self, source: str, destination: str, num_ants: int) -> None:
        for iteration in range(self.num_iterations):
            # Clear previous ants
            self.search_ants.clear()
            
            # Create new ants
            for _ in range(num_ants):
                if self.mode == 2:  # TSP mode
                    # For TSP, randomly select a spawn point
                    spawn_point = random.choice(list(self.graph.nodes()))
                    all_nodes = list(self.graph.nodes())
                    ant = Ant(
                        self.graph_api,
                        spawn_point,  # Random spawn
                        all_nodes,    # All nodes are destinations
                        alpha=self.alpha,
                        beta=self.beta,
                        mode=self.mode
                    )
                    self.search_ants.append(ant)
                else:
                    spawn_point = source
                    ant = Ant(
                        self.graph_api,
                        spawn_point,
                        destination,
                        alpha=self.alpha,
                        beta=self.beta,
                        mode=self.mode
                    )
                    self.search_ants.append(ant)

            # Deploy ants and get progress
            iteration_best_path_cost = self._deploy_forward_search_ants()
            
            # Handle case where destination is origin
            if (iteration_best_path_cost == 0):
                break
            
            max_pheromon, min_pheromon = self._deploy_backward_search_ants(iteration, iteration_best_path_cost)        
            
            # Update pheromones after each iteration
            self.acc, self.d_acc = self.graph_api.update_pheromones(max_pheromon, min_pheromon, self.acc, self.d_acc)
            
            # Apply local search optimization if enabled
            if self.use_local_search and (iteration + 1) % self.local_search_frequency == 0:
                self.best_path, self.best_path_cost = self._apply_2opt_local_search(self.best_path)
            
            # Logging 
            if self.log_step is not None and ((iteration + 1) % self.log_step == 0):
                print(f"Iteration {iteration + 1}/{self.num_iterations} completed. Best path cost: {self.best_path_cost:.2f}")
            
            # Visualization update
            if self.visualize and (iteration % self.visualization_step == 0):
                self.visualizer.update_state(iteration + 1, self.best_path, self.best_path_cost)    

    def find_shortest_path(
        self,
        source: str,
        destination: str,
        num_ants: int,
    ) -> Tuple[List[str], float]:
        """Finds the shortest path according to the current mode
        
        Args:
            source: Source node
            destination: Destination node or list of nodes
            num_ants: Number of ants to deploy
            save_animation: Whether to save the visualization as an animation
            animation_filename: Filename for the animation
            
        Returns:
            Tuple containing the best path and its cost
        """
        # Verify the graph has the required nodes
        if source not in self.graph.nodes():
            raise ValueError(f"Source node {source} cannot access in graph")
        
        if self.mode == 1:
            for dest in destination:
                if dest not in self.graph.nodes():
                    raise ValueError(f"Destination node {dest} cannot access in graph")
        elif self.mode == 0:
            if all(dest not in self.graph.nodes() for dest in destination):
                raise ValueError(f"Destination nodes {destination} cannot access in graph")
            
        # Reset best path and cost
        self.best_path = []
        self.best_path_cost = float("inf")
        
        # Do the actual search
        self._deploy_search_ants(
            source,
            destination,
            num_ants,
        )
        
        # For TSP mode, validate the path includes all nodes
        if self.mode == 2 and self.best_path:
            all_nodes = set(self.graph.nodes())
            path_nodes = set(self.best_path)
            
            # Check for missing nodes
            missing_nodes = all_nodes - path_nodes
            if missing_nodes:
                print(f"Warning: Current best TSP tour is missing nodes: {missing_nodes}")
                
                # If we found any valid path, ensure source is last node
                if self.best_path and self.best_path[-1] != self.best_path[0]:
                    # Ensure tour returns to starting point
                    self.best_path.append(self.best_path[0])

            
        return self.best_path, self.best_path_cost
    
    def _apply_2opt_local_search(self, path):
        """Applies 2-opt local search to improve a path.
        
        The 2-opt algorithm swaps two edges in the path to create a new path
        and keeps the changes if they improve the solution.
        
        Args:
            path: The current path to optimize
            
        Returns:
            Tuple of (improved_path, improved_cost)
        """
        if len(path) < 4:  # Not enough nodes for meaningful swaps
            return path, self._calculate_path_cost(path)
        
        improved = True
        best_path = path.copy()
        best_cost = self._calculate_path_cost(best_path)
        
        # Continue until no improvement can be made
        while improved:
            improved = False
            for i in range(1, len(best_path) - 2):
                for j in range(i + 1, len(best_path) - 1):
                    # Skip if nodes are adjacent
                    if j == i + 1:
                        continue
                    
                    # Create new path with 2-opt swap: reverse the segment between i and j
                    new_path = best_path[:i] + best_path[i:j+1][::-1] + best_path[j+1:]
                    new_cost = self._calculate_path_cost(new_path)
                    
                    # Update if we found a better path
                    if new_cost < best_cost:
                        best_path = new_path
                        best_cost = new_cost
                        improved = True
                        break  # Start the nested loops again with the new path
                
                if improved:
                    break  # break out of outer loop if improved
        
        return best_path, best_cost
    
    def _calculate_path_cost(self, path):
        """Calculate the total cost of a path.
        
        Args:
            path: List of nodes representing a path
            
        Returns:
            The total cost of the path
        """
        cost = 0.0
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            cost += self.graph_api.get_edge_cost(u, v)
        return cost