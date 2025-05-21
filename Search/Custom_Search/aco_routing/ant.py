from typing import Dict, List, Set, Union
import os
import sys

# Import paths setup
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from aco_routing import utils
from aco_routing.graph_api import GraphApi

class Ant:
    def __init__(
        self,
        graph_api: GraphApi,
        source: str,
        destination: Union[str, List[str], Set[str]] = None,
        alpha: float = 0.7,
        beta: float = 0.3,
        mode: int = 0,
    ):
        """Initialize an ant for ACO algorithm.
        
        Args:
            graph_api: The graph API to use for navigation
            source: The starting node
            destination: The destination node(s) to reach
            alpha: Pheromone bias (importance of pheromone trails)
            beta: Edge cost bias (importance of shorter paths)
            mode: Operation mode (0: any destination, 1: all destinations)
        """
        self.graph_api = graph_api
        self.source = source
        self.destination = destination if destination is not None else []
        self.alpha = alpha
        self.beta = beta
        self.visited_nodes = set()
        self.path = []
        self.path_cost = 0.0
        self.is_fit = False
        self.visited_destinations = set()
        self.mode = mode
        
        # Set the spawn node as the current and first node
        self.current_node = self.source
        self.path.append(self.source)
            
        # Initialize visited destinations
        self.visited_destinations = set()
        if self.current_node in self.destination:
            self.visited_destinations.add(self.current_node)

    def reached_destination(self) -> bool:
        """Returns if the ant has reached all specified destinations
        
        For TSP mode (2): returns True when all nodes have been visited and ant has returned to source
        """
        # Update visited destinations
        if self.current_node in self.destination:
            self.visited_destinations.add(self.current_node)
        
        if self.mode == 2:  # TSP mode
            # Get all nodes in the graph
            all_nodes = set(self.graph_api.get_all_nodes())
            
            # Check if we've visited all nodes and returned to source
            has_visited_all = len(self.visited_nodes) >= len(all_nodes)
            
            # For a valid TSP tour, we need to have visited all nodes and returned to our starting point
            return has_visited_all and self.current_node == self.source
        elif self.mode == 0:
            return len(self.visited_destinations) > 0
        else:
            return self.visited_destinations == set(self.destination)

    def _get_unvisited_neighbors(self) -> List[str]:
        """Get unvisited neighbors with optimized set lookup"""
        # Use a set for faster membership testing
        visited_set = self.visited_nodes
        
        # Use list comprehension with faster set lookup
        return [node for node in self.graph_api.get_neighbors(self.current_node)
                if node not in visited_set]

    def _compute_all_edges_desirability(
        self,
        unvisited_neighbors: List[str],
    ) -> float:
        """Computes the denominator of the transition probability equation for the ant

        Args:
            unvisited_neighbors (List[str]): All unvisited neighbors of the current node

        Returns:
            float: The summation of all the outgoing edges (to unvisited nodes) from the current node
        """
        total = 0.0
        for neighbor in unvisited_neighbors:
            edge_pheromones = self.graph_api.get_edge_pheromones(
                self.current_node, neighbor
            )
            edge_cost = self.graph_api.get_edge_cost(self.current_node, neighbor)
            edge_distance = self.graph_api.get_edge_distance(self.current_node, neighbor)
            # Avoid division by zero
            if edge_cost == 0:
                edge_cost = 0.001  # Small value instead of zero
                
            total += utils.compute_edge_desirability(
                edge_pheromones, edge_cost, edge_distance, self.alpha, self.beta, self.mode
            )

        return total

    def _calculate_edge_probabilities(
        self, unvisited_neighbors: List[str]
    ) -> Dict[str, float]:
        """Computes the transition probabilities of all the edges from the current node

        Args:
            unvisited_neighbors (List[str]): A list of unvisited neighbors of the current node

        Returns:
            tuple: A tuple containing (probabilities, transition_values) where both are dictionaries
                  mapping nodes to their respective values
        """
        probabilities: Dict[str, float] = {}
        transition_values: Dict[str, float] = {}
        all_edges_desirability = self._compute_all_edges_desirability(
            unvisited_neighbors
        )
        
        # Guard against division by zero
        if all_edges_desirability == 0:
            # Equal probability for all neighbors
            equal_prob = 1.0 / len(unvisited_neighbors) if unvisited_neighbors else 0
            equal_probs = {neighbor: equal_prob for neighbor in unvisited_neighbors}
            # Return consistent tuple format (probabilities, transition_values)
            return equal_probs, equal_probs

        for neighbor in unvisited_neighbors:
            edge_pheromones = self.graph_api.get_edge_pheromones(
                self.current_node, neighbor
            )
            edge_cost = self.graph_api.get_edge_cost(self.current_node, neighbor)
            edge_distance = self.graph_api.get_edge_distance(self.current_node, neighbor)
            
            # Avoid division by zero
            if edge_cost == 0:
                edge_cost = 0.001  # Small value instead of zero

            current_edge_desirability = utils.compute_edge_desirability(
                edge_pheromones, edge_cost, edge_distance, self.alpha, self.beta, self.mode
            )
            transition_values[neighbor] = current_edge_desirability
            probabilities[neighbor] = current_edge_desirability / all_edges_desirability

        return probabilities, transition_values

    def _choose_next_node(self) -> Union[str, None]:
        """Choose the next node to be visited by the ant"""
        if self.mode == 2:  # TSP mode
            all_nodes = set(self.graph_api.get_all_nodes())
            unvisited = all_nodes - self.visited_nodes
            
            # If we've visited all nodes, try to return to source
            if len(unvisited) == 0:
                if self.source in self.graph_api.get_neighbors(self.current_node):
                    return self.source
                return None
                
            # Prioritize unvisited nodes, especially if few remain
            unvisited_neighbors = [n for n in self.graph_api.get_neighbors(self.current_node) 
                                if n in unvisited]
        else:
            # Original implementation for other modes
            unvisited_neighbors = self._get_unvisited_neighbors()

        # Check if ant has no possible nodes to move to
        if len(unvisited_neighbors) == 0:
            return None

        # For regular ants, use probabilistic selection
        probabilities, transition_values = self._calculate_edge_probabilities(unvisited_neighbors)
        return utils.pseudo_random_proportional_selection(transition_values, probabilities)

    def take_step(self) -> None:
        """Compute and update the ant position"""
        # Mark the current node as visited
        self.visited_nodes.add(self.current_node)
        
        # Update visited destinations
        if self.current_node in self.destination:
            self.visited_destinations.add(self.current_node)

        # Pick the next node of the ant
        next_node = self._choose_next_node()

        # Check if ant is stuck at current node or has reached all destinations
        if not next_node:
            if self.mode == 2:
                # For TSP: check if we've visited all nodes and returned to source
                all_nodes = set(self.graph_api.get_all_nodes())
                if len(self.visited_nodes) == len(all_nodes) and self.current_node == self.source:
                    # Complete successful tour
                    self.is_fit = True
                else:
                    # Incomplete tour - not fit
                    self.is_fit = False
            elif self.reached_destination():
                self.is_fit = True
            return

        # Standard case: add the new node to the path
        self.path.append(next_node)
        self.path_cost += self.graph_api.get_edge_cost(self.current_node, next_node)
        self.current_node = next_node
        
        # For TSP mode, update fitness immediately if we've completed a tour
        if self.mode == 2 and self.reached_destination():
            self.is_fit = True

    def deposit_pheromones_on_path(self, elitist_param) -> None:
        for i in range(len(self.path) - 1):
            u, v = self.path[i], self.path[i + 1]
            if elitist_param == 0:
                deposit_pheromone_value = self.graph_api.pheromone_deposit_weight / self.path_cost
            else:
                deposit_pheromone_value = elitist_param / self.path_cost
            self.graph_api.deposit_pheromones(u, v, deposit_pheromone_value)