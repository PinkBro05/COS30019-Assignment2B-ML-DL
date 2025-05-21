from typing import List, Dict
import matplotlib.pyplot as plt
import os
import sys
import math
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import matplotlib.colors as mcolors
import numpy as np

# Add the parent directory to the system path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from aco_routing.network import Network


class GraphApi:
    def __init__(self, graph: Network, evaporation_rate: float):
        """Initialize the GraphApi with a network and evaporation rate.
        
        Args:
            graph: Network object containing the graph structure
            evaporation_rate: Rate at which pheromones evaporate (0-1)
        """
        self.graph = graph
        self.evaporation_rate = evaporation_rate
        self.gamma = 0.95
        self.epsilon = 1e-7
        self.pheromone_deposit_weight = 1 # Weight for pheromone deposit
        
        # Precompute and cache edge costs
        self._edge_cost_cache = {}
        self._neighbor_cache = {}
        
        # Precompute edge costs
        for u, v in self.graph.get_edges():
            self._edge_cost_cache[(u, v)] = self.graph.edges.get((u, v), {}).get("cost", float('inf'))
        
        # Precompute neighbors for each node
        for node in self.graph.nodes():
            self._neighbor_cache[node] = list(self.graph.neighbors(node))

    def set_edge_pheromones(self, u: str, v: str, pheromone_value: float) -> None:
        if (u, v) in self.graph.edges:
            self.graph.edges[(u, v)]["pheromones"] = pheromone_value
    
    def set_edge_delta_pheromones(self, u: str, v: str, delta_pheromone_value: float) -> None:
        if (u, v) in self.graph.edges:
            self.graph.edges[(u, v)]["delta_pheromones"] = delta_pheromone_value

    def get_edge_pheromones(self, u: str, v: str) -> float:
        return self.graph.edges.get((u, v), {}).get("pheromones", 0.0)

    def deposit_pheromones(self, u: str, v: str, pheromone_amount: float) -> None:
        if (u, v) in self.graph.edges:
            delta_pheromone = pheromone_amount
            self.graph.edges[(u, v)]["delta_pheromones"] += delta_pheromone
            
    def deposit_pheromones_for_path(self, path: List[str]) -> None:
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            
            # This ensures that pheromones are deposited even if the edge cost is zero
            if self.get_edge_cost(u, v) == 0.0:
                self.set_edge_pheromones(u, v, 0.01)
                if hasattr(self, '_edge_cost_cache') and (u, v) in self._edge_cost_cache:
                    self._edge_cost_cache[(u, v)] = 0.01
                    
            pheromone_amount = self.pheromone_deposit_weight / self.get_edge_cost(u, v)
            self.deposit_pheromones(u, v, pheromone_amount)
            
    def update_pheromones(self, max_pheromon, min_pheromon, current_acc, current_d_acc) -> None:
        for u, v in self.graph.get_edges():
            if (u, v) in self.graph.edges:
                # Gradient descent update using Adadelta Optimizer
                pheromones = self.graph.edges[(u, v)].get("pheromones", 0.0)
                gt = pheromones - (self.graph.edges[(u, v)].get("delta_pheromones", 0.0)) * self.pheromone_deposit_weight
                acc = self.gamma * current_acc + (1 - self.gamma) * gt * gt
                update = (gt * math.sqrt(current_d_acc + self.epsilon)) / (math.sqrt(acc + self.epsilon))
                new_pheromone = pheromones - update
                d_acc = self.gamma * current_d_acc + (1 - self.gamma) * update * update
                
                self.graph.edges[(u, v)]["delta_pheromones"] = 0.0  # Reset delta pheromone after updating
                
                # Max min ant system
                if new_pheromone < min_pheromon:
                    new_pheromone = min_pheromon
                elif new_pheromone > max_pheromon:
                    new_pheromone = max_pheromon
                self.graph.edges[(u, v)]["pheromones"] = new_pheromone
        return acc, d_acc

    def get_edge_cost(self, u: str, v: str) -> float:
        """Get edge cost with caching for better performance"""
        # Use cached value if available
        if hasattr(self, '_edge_cost_cache') and (u, v) in self._edge_cost_cache:
            return self._edge_cost_cache[(u, v)]
        
        # Fallback to original implementation
        return self.graph.edges.get((u, v), {}).get("cost", float('inf'))

    def get_edge_distance(self, u: str, v: str) -> float:
        """Get edge distance with caching for better performance"""
        # Use cached value if available
        if hasattr(self, '_edge_distance_cache') and (u, v) in self._edge_distance_cache:
            return self._edge_distance_cache[(u, v)]
        
        # Fallback to original implementation
        return self.graph.edges.get((u, v), {}).get("distance", float('inf'))

    def get_all_nodes(self) -> List[str]:
        return list(self.graph.nodes())

    def get_neighbors(self, node: str) -> List[str]:
        """Get neighbors with caching for better performance"""
        # Use cached value if available
        if hasattr(self, '_neighbor_cache') and node in self._neighbor_cache:
            return self._neighbor_cache[node]
        
        # Fallback to original implementation
        return list(self.graph.neighbors(node))
    
    def get_pheromone_levels(self) -> Dict:
        """Get all pheromone levels in the graph for debugging"""
        result = {}
        for (u, v) in self.graph.get_edges():
            result[(u, v)] = self.get_edge_pheromones(u, v)
        return result

    def visualize_graph(self, shortest_path: List[str], shortest_path_cost=None) -> None:
        """Visualize the graph with the shortest path highlighted
        
        Args:
            shortest_path: List of nodes representing the path to highlight
            shortest_path_cost: The cost of the shortest path (optional)
        """
        try:
            # Create a figure with a specific size
            plt.figure(figsize=(12, 10))
            
            # Create a set of path edges for easy lookup
            path_edges = set()
            for i in range(len(shortest_path) - 1):
                path_edges.add((shortest_path[i], shortest_path[i + 1]))
            
            # Draw ALL nodes in the graph
            path_nodes = set(shortest_path)
            all_nodes = list(self.graph.nodes())
            
            # Get max pheromone value for color normalization
            max_pheromone = 0.0
            for u, v in self.graph.get_edges():
                pheromone = self.graph.edges.get((u, v), {}).get("pheromones", 0)
                max_pheromone = max(max_pheromone, pheromone)
            
            # Ensure max_pheromone is not zero to avoid division by zero
            max_pheromone = max(max_pheromone, 0.001)
            
            # Draw non-path nodes first (in the background) with 20% opacity
            non_path_nodes = [node for node in all_nodes if node not in path_nodes]
            non_path_xs = [self.graph.pos[node][0] for node in non_path_nodes if node in self.graph.pos]
            non_path_ys = [self.graph.pos[node][1] for node in non_path_nodes if node in self.graph.pos]
            plt.scatter(non_path_xs, non_path_ys, s=700, c='skyblue', edgecolors='black', alpha=0.2, zorder=1)
            
            # Draw path nodes on top (highlighted)
            path_xs = [self.graph.pos[node][0] for node in path_nodes if node in self.graph.pos]
            path_ys = [self.graph.pos[node][1] for node in path_nodes if node in self.graph.pos]
            plt.scatter(path_xs, path_ys, s=900, c='lightcoral', edgecolors='black', zorder=2)
            
            # Draw all edges in the graph with pheromone-based coloring
            for u, v in self.graph.get_edges():
                # Skip if either node position is missing
                if u not in self.graph.pos or v not in self.graph.pos:
                    continue
                    
                x1, y1 = self.graph.pos[u]
                x2, y2 = self.graph.pos[v]
                
                # Get pheromone value for this edge
                pheromone = self.graph.edges.get((u, v), {}).get("pheromones", 0)
                
                # Calculate color based on pheromone level - red for high, green for low
                # Normalize pheromone to [0, 1] range
                normalized_pheromone = pheromone / max_pheromone
                
                # Create color: interpolate from green (low) to red (high)
                # RGB interpolation from green (0.0, 0.8, 0.0) to red (1.0, 0.0, 0.0)
                r = normalized_pheromone
                g = 0.8 * (1 - normalized_pheromone)
                b = 0.0
                
                # Set different opacity for path vs non-path edges
                if (u, v) in path_edges:
                    # For path edges: high opacity (0.7-1.0) based on pheromone strength
                    alpha = 0.7 + 0.3 * normalized_pheromone
                    line_width = 3
                else:
                    # For non-path edges: fixed low opacity (20%)
                    alpha = 0.2
                    line_width = 1
                    
                edge_color = (r, g, b)
                
                # Draw edge with pheromone-based color and opacity
                plt.plot([x1, x2], [y1, y2], color=edge_color, linewidth=line_width, alpha=alpha, zorder=0)
                
                # Add arrow to show direction
                dx = x2 - x1
                dy = y2 - y1
                plt.arrow(
                    x1 + 0.8*dx, y1 + 0.8*dy, 
                    0.1*dx, 0.1*dy, 
                    head_width=0.05, 
                    head_length=0.1, 
                    fc=edge_color, ec=edge_color,
                    alpha=alpha,
                    zorder=0
                )
                
                # Only add labels to path edges
                if (u, v) in path_edges:
                    # Add edge label for cost and pheromone for path edges
                    midx, midy = (x1 + x2) / 2, (y1 + y2) / 2
                    edge_data = None
                    edge_pairs = [(u, v)]
                    
                    for edge_pair in edge_pairs:
                        if edge_pair in self.graph.edges:
                            edge_data = self.graph.edges[edge_pair]
                            break
                        
                    if edge_data:
                        cost = edge_data.get("cost", 0)
                        pheromone = edge_data.get("pheromones", 0)
                        label = f"cost: {cost}\nphero: {round(pheromone, 3)}"
                        plt.text(midx, midy, label, fontsize=8, fontweight='bold',
                                bbox=dict(facecolor='mistyrose', alpha=0.8, edgecolor='gray'),
                                zorder=5, ha='center', va='center')
            
            # Add node labels for ALL nodes
            for node in all_nodes:
                if node not in self.graph.pos:
                    continue
                    
                x, y = self.graph.pos[node]
                # Highlight path node labels
                if node in path_nodes:
                    plt.text(x, y, node, fontsize=14, fontweight='bold', ha='center', va='center', zorder=6)
                else:
                    # Use 20% opacity for unused node labels
                    plt.text(x, y, node, fontsize=12, fontweight='normal', alpha=0.2, 
                            ha='center', va='center', zorder=6)
            
            # Add grid
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add title
            path_str = " → ".join(shortest_path)
            cost_str = f", Cost: {shortest_path_cost}" if shortest_path_cost is not None else ""
            plt.title(f"Shortest Path: {path_str}{cost_str}", fontsize=14)
            
            # Create gradient legend
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient))
            
            # Create a custom colormap
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'pheromone_cmap', [(0.0, 0.8, 0.0), (1.0, 0.0, 0.0)]
            )
            
            # Legend elements
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                    markersize=15, label='Path node'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', alpha=0.2,
                    markersize=15, label='Unused node (20% opacity)'),
                Patch(facecolor='white', edgecolor='black', label='Pheromone Level:')
            ]
            
            # Position the legend
            legend = plt.legend(handles=legend_elements, loc='upper right')
            
            # Add the colorbar for pheromone levels
            ax = plt.gca()
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_pheromone))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.02, shrink=0.5)
            cbar.set_label('Pheromone Level (Green: Low, Red: High, Unused paths at 20% opacity)')
            
            # Adjust layout
            plt.tight_layout()
            
            # Show the plot
            plt.show()
        
        except Exception as e:
            import traceback
            print(f"Detailed visualization error: {e}")
            traceback.print_exc()
