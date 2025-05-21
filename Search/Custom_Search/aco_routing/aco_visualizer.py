import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import numpy as np
from typing import List, Dict

class ACOVisualizer:
    def __init__(self, graph_api, fig_size=(12, 10), update_interval=500):
        """Initialize the ACO Visualizer.
        
        Args:
            graph_api: The GraphApi instance
            fig_size: Size of the figure (width, height)
            update_interval: Animation update interval in milliseconds
        """
        self.graph_api = graph_api
        self.fig_size = fig_size
        self.update_interval = update_interval
        
        # Setup interactive plotting
        plt.ion()
        self.fig = plt.figure(figsize=self.fig_size)
        self.ax = self.fig.add_subplot(111)
        
        # Store animation data
        self.frames = []
        self.current_iteration = 0
        self.best_path = []
        self.best_path_cost = float('inf')
        self.animation = None
        
        
    def update_state(self, iteration, best_path, best_path_cost):
        """Update the visualization state with the current algorithm progress.
        
        Args:
            iteration: Current iteration number
            best_path: Current best path
            best_path_cost: Current best path cost
        """
        self.current_iteration = iteration
        self.best_path = best_path
        self.best_path_cost = best_path_cost
            
        # Create a frame of the current state
        self.visualize_current_state()
    
    def visualize_current_state(self):
        """Create a visualization of the current state."""
        # Clear previous plot and colorbar
        self.ax.clear()
        self.fig.clf()  # Clear the entire figure to remove old colorbars
        self.ax = self.fig.add_subplot(111)  # Re-create the axis
        
        # Create path edges for highlighting
        path_edges = set()
        if len(self.best_path) > 1:
            for i in range(len(self.best_path) - 1):
                path_edges.add((self.best_path[i], self.best_path[i + 1]))
        
        # Get nodes and set attributes
        path_nodes = set(self.best_path)
        all_nodes = list(self.graph_api.get_all_nodes())
        
        # Get max pheromone value for normalization
        max_pheromone = 0.0001  # Small non-zero value to avoid division by zero
        for u, v in self.graph_api.graph.get_edges():
            pheromone = self.graph_api.graph.edges.get((u, v), {}).get("pheromones", 0)
            max_pheromone = max(max_pheromone, pheromone)
        
        # Draw non-path nodes
        non_path_nodes = [node for node in all_nodes if node not in path_nodes]
        non_path_xs = [self.graph_api.graph.pos[node][0] for node in non_path_nodes if node in self.graph_api.graph.pos]
        non_path_ys = [self.graph_api.graph.pos[node][1] for node in non_path_nodes if node in self.graph_api.graph.pos]
        self.ax.scatter(non_path_xs, non_path_ys, s=700, c='skyblue', edgecolors='black', alpha=0.2, zorder=1)
        
        # Draw path nodes
        path_xs = [self.graph_api.graph.pos[node][0] for node in path_nodes if node in self.graph_api.graph.pos]
        path_ys = [self.graph_api.graph.pos[node][1] for node in path_nodes if node in self.graph_api.graph.pos]
        self.ax.scatter(path_xs, path_ys, s=900, c='lightcoral', edgecolors='black', zorder=2)
        
        # Draw all edges with pheromone visualization
        for u, v in self.graph_api.graph.get_edges():
            # Skip if node positions are missing
            if u not in self.graph_api.graph.pos or v not in self.graph_api.graph.pos:
                continue
            
            x1, y1 = self.graph_api.graph.pos[u]
            x2, y2 = self.graph_api.graph.pos[v]
            
            # Get edge pheromone level
            pheromone = self.graph_api.graph.edges.get((u, v), {}).get("pheromones", 0)
            normalized_pheromone = pheromone / max_pheromone
            
            # Create color: green (low) to red (high)
            r = normalized_pheromone
            g = 0.8 * (1 - normalized_pheromone)
            b = 0.0
            
            # Set opacity based on path membership
            if (u, v) in path_edges:
                alpha = 0.7 + 0.3 * normalized_pheromone
                line_width = 3
            else:
                alpha = 0.05
                line_width = 1
            
            edge_color = (r, g, b)
            
            # Draw the edge
            self.ax.plot([x1, x2], [y1, y2], color=edge_color, linewidth=line_width, alpha=alpha, zorder=0)
            
            # Add direction arrow
            dx = x2 - x1
            dy = y2 - y1
            self.ax.arrow(
                x1 + 0.8*dx, y1 + 0.8*dy,
                0.1*dx, 0.1*dy,
                head_width=0.05,
                head_length=0.1,
                fc=edge_color, ec=edge_color,
                alpha=alpha,
                zorder=0
            )
            
            # Add edge labels for path edges
            # if (u, v) in path_edges:
            #     midx, midy = (x1 + x2) / 2, (y1 + y2) / 2
            #     cost = self.graph_api.graph.edges.get((u, v), {}).get("cost", 0)
            #     pheromone_value = self.graph_api.graph.edges.get((u, v), {}).get("pheromones", 0)
            #     label = f"cost: {cost}\nphero: {round(pheromone_value, 3)}"
            #     self.ax.text(midx, midy, label, fontsize=8, fontweight='bold',
            #             bbox=dict(facecolor='mistyrose', alpha=0.8, edgecolor='gray'),
            #             zorder=5, ha='center', va='center')
        
        # Add node labels
        for node in all_nodes:
            if node not in self.graph_api.graph.pos:
                continue
                
            x, y = self.graph_api.graph.pos[node]
            if node in path_nodes:
                self.ax.text(x, y, node, fontsize=14, fontweight='bold', ha='center', va='center', zorder=6)
            else:
                self.ax.text(x, y, node, fontsize=12, fontweight='normal', alpha=0.2,
                        ha='center', va='center', zorder=6)
        
        # Add grid
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add title and information
        path_str = " → ".join(self.best_path) if self.best_path else "No path found yet"
        cost_str = f", Cost: {self.best_path_cost:.2f}" if self.best_path_cost < float('inf') else ""
        self.ax.set_title(f"ACO Progress: Iteration {self.current_iteration}\nBest Path: {path_str}{cost_str}", fontsize=14)
        
        # Create a custom colormap
        cmap = mcolors.LinearSegmentedColormap.from_list(
            'pheromone_cmap', [(0.0, 0.8, 0.0), (1.0, 0.0, 0.0)]
        )
        
        # Add color bar for pheromone levels
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_pheromone))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=self.ax, orientation='horizontal', pad=0.02, shrink=0.5)
        cbar.set_label('Pheromone Level (Green: Low, Red: High)')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.5)  # Short pause to allow GUI update
        
        # Save the frame for future animation
        self.frames.append(np.array(self.fig.canvas.renderer.buffer_rgba()))