"""
Utility functions for predicting traffic flow and integrating with path finding.
This module serves as a bridge between the traffic flow prediction model and the path finding algorithms.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import importlib.util

# Add parent directory to path to import from modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the math utilities for flow to velocity and velocity to time conversions
from Utils.maths import flow_to_velocity, velocity_to_time

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
SAMPLE_DATASET_PATH = os.path.join(BASE_DIR, "ML", "Data", "Transformed", "_sample_final_time_series.csv")

class TrafficPredictionIntegrator:
    """
    Class to integrate traffic flow prediction with path finding algorithms.
    """
    
    def __init__(self, model_path=None, dataset_path=None):
        """
        Initialize the traffic prediction integrator.
        
        Args:
            model_path: Path to the model file (default: use the default model)
            dataset_path: Path to the dataset file (default: use _sample_final_time_series.csv)
        """
        self.model_path = model_path
        self.dataset_path = SAMPLE_DATASET_PATH if dataset_path is None else dataset_path
        
        # Initialize the predictor
        self._init_predictor()
        
    def _init_predictor(self):
        """Initialize the traffic flow predictor."""
        try:
            # Import the TrafficFlowPredictor from the init module
            from ML.Transformer.init import TrafficFlowPredictor
            
            # Create an instance of the predictor
            self.predictor = TrafficFlowPredictor(model_path=self.model_path)
            print("Traffic flow predictor initialized successfully.")
        except Exception as e:
            print(f"Error initializing traffic flow predictor: {e}")
            self.predictor = None
    
    def predict_and_convert_to_costs(self, site_ids, start_time, graph_file_path):
        """
        Predict traffic flow for a list of sites and convert to edge costs.
        
        Args:
            site_ids: List of SCATS site IDs to predict for
            start_time: Datetime to predict for (can be string or datetime object)
            graph_file_path: Path to the graph file containing edge information
            
        Returns:
            Dictionary mapping edge tuples (node1, node2) to new costs (travel time in seconds)
        """
        # Import parser for reading the graph file
        spec = importlib.util.spec_from_file_location(
            "parser", 
            os.path.join(BASE_DIR, 'Search', 'data_reader', 'parser.py')
        )
        parser_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parser_module)
        parse_graph_file = parser_module.parse_graph_file
        
        # Parse the graph file to get nodes and edges
        nodes, edges, _, _ = parse_graph_file(graph_file_path)
        
        # Create a mapping from node ID to SCATS site ID
        # In this application, they are the same
        node_to_site = {node_id: node_id for node_id in nodes.keys()}
        
        # Initialize dictionary to store new edge costs
        new_edge_costs = {}
        
        # For each site, predict traffic flow and calculate new edge costs
        site_predictions = {}
        for site_id in site_ids:
            if self.predictor is None:
                # If predictor failed to initialize, use default flow
                site_predictions[site_id] = 500  # Default moderate flow
                continue
                
            try:
                # Predict traffic flow for the site
                prediction = self.predictor.predict_flow(site_id, start_time, self.dataset_path)
                
                if prediction['predicted_flows'] is not None:
                    # Use the first prediction (current time step)
                    flow = prediction['predicted_flows'][0]
                    site_predictions[site_id] = flow
                else:
                    # Use default flow if prediction failed
                    site_predictions[site_id] = 500  # Default moderate flow
            except Exception as e:
                print(f"Error predicting flow for site {site_id}: {e}")
                site_predictions[site_id] = 500  # Default moderate flow
        
        # Calculate new edge costs based on predicted flows
        for edge, distance in edges.items():
            node1, node2 = edge
            
            # Get site IDs for both nodes
            site1 = node_to_site.get(node1)
            site2 = node_to_site.get(node2)
            
            # Use average flow if both nodes have predictions, otherwise use available prediction or default
            if site1 in site_predictions and site2 in site_predictions:
                flow = (site_predictions[site1] + site_predictions[site2]) / 2
            elif site1 in site_predictions:
                flow = site_predictions[site1]
            elif site2 in site_predictions:
                flow = site_predictions[site2]
            else:
                flow = 500  # Default moderate flow
            
            # Convert flow to velocity (km/h)
            try:
                velocity = flow_to_velocity(flow)
                
                # Convert velocity to travel time (seconds)
                travel_time = velocity_to_time(velocity, distance)
                
                # Store the new cost (travel time)
                new_edge_costs[edge] = travel_time
            except ValueError as e:
                print(f"Error calculating cost for edge {edge}: {e}")
                # Use a fallback calculation based on distance
                # Assume 30 km/h if calculation failed
                velocity = 30
                travel_time = velocity_to_time(velocity, distance)
                new_edge_costs[edge] = travel_time
        
        return new_edge_costs
    
    def update_graph_with_predictions(self, chunked_graph_path, start_time, output_path):
        """
        Update a chunked graph file with predicted traffic flow costs.
        
        Args:
            chunked_graph_path: Path to the chunked graph file
            start_time: Start time for prediction
            output_path: Path to write the updated graph file
            
        Returns:
            Boolean indicating success or failure
        """
        try:
            # Read the chunked graph file
            with open(chunked_graph_path, 'r') as f:
                lines = f.readlines()
            
            # Extract node IDs from the graph
            site_ids = []
            nodes_section = False
            edges_section = False
            
            for line in lines:
                line = line.strip()
                if line == "Nodes:":
                    nodes_section = True
                    edges_section = False
                    continue
                elif line == "Edges:":
                    nodes_section = False
                    edges_section = True
                    continue
                elif line == "Origin:" or line == "Destinations:":
                    nodes_section = False
                    edges_section = False
                    continue
                
                if nodes_section and line:
                    # Extract node ID from line like "2015: (144.9631,-37.8136)"
                    node_id = line.split(':')[0].strip()
                    site_ids.append(node_id)
            
            # Predict and convert to costs
            new_edge_costs = self.predict_and_convert_to_costs(site_ids, start_time, chunked_graph_path)
            
            # Create updated graph file
            with open(output_path, 'w') as f:
                # Write nodes section
                nodes_section = False
                edges_section = False
                for line in lines:
                    line = line.strip()
                    if line == "Nodes:":
                        nodes_section = True
                        edges_section = False
                        f.write("Nodes:\n")
                        continue
                    elif line == "Edges:":
                        nodes_section = False
                        edges_section = True
                        f.write("Edges:\n")
                        continue
                    elif line == "Origin:" or line == "Destinations:":
                        nodes_section = False
                        edges_section = False
                        f.write(line + "\n")
                        continue
                    
                    if nodes_section:
                        f.write(line + "\n")
                    elif edges_section:
                        if line:
                            # Parse edge line like "(2015,3629): 500"
                            edge_part, cost_part = line.split(':')
                            edge_str = edge_part.strip()[1:-1]  # Remove parentheses
                            node1, node2 = edge_str.split(',')
                            
                            # Get new cost if available
                            edge = (node1, node2)
                            if edge in new_edge_costs:
                                f.write(f"({node1},{node2}): {new_edge_costs[edge]}\n")
                            else:
                                f.write(line + "\n")
                    else:
                        f.write(line + "\n")
            
            return True
        except Exception as e:
            print(f"Error updating graph with predictions: {e}")
            return False


def predict_traffic_flows(site_ids, start_time, dataset_path=None):
    """
    Predict traffic flows for a list of sites.
    
    Args:
        site_ids: List of SCATS site IDs to predict for
        start_time: Datetime to predict for (can be string or datetime object)
        dataset_path: Path to the dataset file (default: use _sample_final_time_series.csv)
        
    Returns:
        Dictionary mapping site IDs to predicted flow values
    """
    # Create an instance of the integrator
    integrator = TrafficPredictionIntegrator(dataset_path=dataset_path)
    
    # Initialize dictionary to store predictions
    predictions = {}
    
    # For each site, predict traffic flow
    for site_id in site_ids:
        if integrator.predictor is None:
            # If predictor failed to initialize, use default flow
            predictions[site_id] = 500  # Default moderate flow
            continue
            
        try:
            # Predict traffic flow for the site
            # prediction = integrator.predictor.predict_flow(site_id, start_time, dataset_path)
            
            # Fix the site id to 100 since the prepared sample data only contains this site
            prediction = integrator.predictor.predict_flow(100, start_time, dataset_path)
            
            if prediction['predicted_flows'] is not None:
                # Use the first prediction (current time step)
                flow = prediction['predicted_flows'][0]
                predictions[site_id] = flow
            else:
                # Use default flow if prediction failed
                predictions[site_id] = 500  # Default moderate flow
        except Exception as e:
            print(f"Error predicting flow for site {site_id}: {e}")
            predictions[site_id] = 500  # Default moderate flow
    
    return predictions


def prepare_traffic_based_search(origin, destination, start_time, chunked_graph_path=None, output_path=None):
    """
    Prepare a graph for traffic-based path search.
    
    Args:
        origin: Origin node ID
        destination: Destination node ID
        start_time: Start time for prediction
        chunked_graph_path: Path to the chunked graph file (default: Data/temp_chunked_graph.txt)
        output_path: Path to write the updated graph file (default: Data/traffic_graph.txt)
        
    Returns:
        Path to the updated graph file or None if preparation failed
    """
    # Set default paths if not provided
    if chunked_graph_path is None:
        chunked_graph_path = os.path.join(DATA_DIR, 'temp_chunked_graph.txt')
    
    if output_path is None:
        output_path = os.path.join(DATA_DIR, 'temp_chunked_graph.txt')
    
    # Import the chunking utility if not already imported
    try:
        from Utils.filter_chunk import create_chunked_graph, write_chunked_graph_to_file
    except ImportError:
        spec = importlib.util.spec_from_file_location(
            "filter_chunk", 
            os.path.join(BASE_DIR, 'Utils', 'filter_chunk.py')
        )
        filter_chunk_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(filter_chunk_module)
        create_chunked_graph = filter_chunk_module.create_chunked_graph
        write_chunked_graph_to_file = filter_chunk_module.write_chunked_graph_to_file
    
    try:
        # Create chunked graph
        graph_file_path = os.path.join(DATA_DIR, 'graph.txt')
        filtered_nodes, filtered_edges, origin_id, dest_ids = create_chunked_graph(
            graph_file_path, origin, destination, margin_factor=0.2
        )
        
        # Write chunked graph to file
        write_chunked_graph_to_file(filtered_nodes, filtered_edges, origin_id, dest_ids, chunked_graph_path)
        
        # Create an instance of the integrator
        integrator = TrafficPredictionIntegrator()
        
        # Update graph with predictions
        success = integrator.update_graph_with_predictions(chunked_graph_path, start_time, output_path)
        
        if success:
            return output_path
        else:
            return chunked_graph_path  # Fall back to original chunked graph if update failed
    except Exception as e:
        print(f"Error preparing traffic-based search: {e}")
        return None
