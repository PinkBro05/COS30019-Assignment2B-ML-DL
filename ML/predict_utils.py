"""
Utility functions for predicting traffic flow and integrating with path finding.
This module serves as a bridge between the traffic flow prediction model and the path finding algorithms.
"""

import os
import sys
import importlib.util
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

# Add parent directory to path to import from modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the math utilities for flow to velocity and velocity to time conversions
from Utils.maths import flow_to_velocity, velocity_to_time

# Constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'Data')
SAMPLE_DATASET_PATH = os.path.join(BASE_DIR, "ML", "Data", "Transformed", "_sample_final_time_series.csv")
DEFAULT_FLOW = 500
SCATS_DELAY = 30  # seconds
DEFAULT_VELOCITY = 30  # km/h
GRAPH_SECTIONS = {"NODES": "Nodes:", "EDGES": "Edges:", "ORIGIN": "Origin:", "DESTINATIONS": "Destinations:"}

def _import_module_from_path(module_name: str, file_path: str):
    """Helper function to dynamically import a module from a file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class TrafficPredictionIntegrator:
    """Integrates traffic flow prediction with path finding algorithms."""
    
    def __init__(self, model_path: Optional[str] = None, dataset_path: Optional[str] = None):
        """Initialize the traffic prediction integrator."""
        self.model_path = model_path
        self.dataset_path = dataset_path or SAMPLE_DATASET_PATH
        self.predictor = self._init_predictor()
        
    def _init_predictor(self):
        """Initialize the traffic flow predictor."""
        try:
            from ML.Transformer.init import TrafficFlowPredictor
            predictor = TrafficFlowPredictor(model_path=self.model_path)
            print("Traffic flow predictor initialized successfully.")
            return predictor
        except Exception as e:
            print(f"Error initializing traffic flow predictor: {e}")
            return None

    def _get_site_predictions(self, site_ids: List[str], start_time: Union[str, datetime]) -> Dict[str, float]:
        """Get traffic flow predictions for all sites."""
        if not self.predictor:
            return {site_id: DEFAULT_FLOW for site_id in site_ids}
        
        predictions = {}
        for site_id in site_ids:
            try:
                prediction = self.predictor.predict_flow(site_id, start_time, self.dataset_path)
                flow = (prediction['predicted_flows'][0] 
                       if prediction['predicted_flows'] is not None 
                       else DEFAULT_FLOW)
                predictions[site_id] = flow
            except Exception as e:
                print(f"Error predicting flow for site {site_id}: {e}")
                predictions[site_id] = DEFAULT_FLOW
        return predictions

    def _calculate_edge_cost(self, flow: float, distance: float) -> float:
        """Calculate travel time for an edge based on flow and distance."""
        try:
            velocity = flow_to_velocity(flow)
            travel_time = velocity_to_time(velocity, distance)
        except ValueError as e:
            print(f"Error calculating velocity/time: {e}")
            travel_time = velocity_to_time(DEFAULT_VELOCITY, distance)
        
        return travel_time + SCATS_DELAY

    def _get_edge_flow(self, edge: Tuple[str, str], site_predictions: Dict[str, float]) -> float:
        """Get flow for an edge, prioritizing destination node."""
        node1, node2 = edge
        return site_predictions.get(node2, site_predictions.get(node1, DEFAULT_FLOW))

    def predict_and_convert_to_costs(self, site_ids: List[str], start_time: Union[str, datetime], 
                                    graph_file_path: str) -> Dict[Tuple[str, str], float]:
        """Predict traffic flow for sites and convert to edge costs."""
        # Parse graph file
        parser_module = _import_module_from_path("parser", 
                                               os.path.join(BASE_DIR, 'Search', 'data_reader', 'parser.py'))
        nodes, edges, _, _ = parser_module.parse_graph_file(graph_file_path)
          # Get predictions for all sites
        site_predictions = self._get_site_predictions(site_ids, start_time)
        
        # Calculate edge costs
        return {edge: self._calculate_edge_cost(self._get_edge_flow(edge, site_predictions), distance)
                for edge, distance in edges.items()}

    def _extract_site_ids_from_graph(self, lines: List[str]) -> List[str]:
        """Extract site IDs from graph file lines."""
        site_ids = []
        nodes_section = False
        
        for line in lines:
            line = line.strip()
            if line == GRAPH_SECTIONS["NODES"]:
                nodes_section = True
            elif line in [GRAPH_SECTIONS["EDGES"], GRAPH_SECTIONS["ORIGIN"], GRAPH_SECTIONS["DESTINATIONS"]]:
                nodes_section = False
            elif nodes_section and line:
                site_ids.append(line.split(':')[0].strip())
        return site_ids

    def _update_graph_lines(self, lines: List[str], new_edge_costs: Dict[Tuple[str, str], float]) -> List[str]:
        """Update graph file lines with new edge costs."""
        updated_lines = []
        section = None
        
        for line in lines:
            stripped = line.strip()
            
            # Determine current section
            if stripped in GRAPH_SECTIONS.values():
                section = stripped
                updated_lines.append(line)
            elif section == GRAPH_SECTIONS["EDGES"] and stripped:
                # Parse and update edge costs
                try:
                    edge_part, _ = stripped.split(':')
                    edge_str = edge_part.strip()[1:-1]  # Remove parentheses
                    node1, node2 = edge_str.split(',')
                    edge = (node1, node2)
                    
                    if edge in new_edge_costs:
                        updated_lines.append(f"({node1},{node2}): {new_edge_costs[edge]}\n")
                    else:
                        updated_lines.append(line)
                except ValueError:
                    updated_lines.append(line)
            else:
                updated_lines.append(line)
        
        return updated_lines

    def update_graph_with_predictions(self, chunked_graph_path: str, start_time: Union[str, datetime], 
                                    output_path: str) -> bool:
        """Update a chunked graph file with predicted traffic flow costs."""
        try:
            with open(chunked_graph_path, 'r') as f:
                lines = f.readlines()
            
            site_ids = self._extract_site_ids_from_graph(lines)
            new_edge_costs = self.predict_and_convert_to_costs(site_ids, start_time, chunked_graph_path)
            updated_lines = self._update_graph_lines(lines, new_edge_costs)
            
            with open(output_path, 'w') as f:
                f.writelines(updated_lines)
            
            return True
        except Exception as e:
            print(f"Error updating graph with predictions: {e}")
            return False

def _get_filter_chunk_functions():
    """Import chunking utilities dynamically."""
    try:
        from Utils.filter_chunk import create_chunked_graph, write_chunked_graph_to_file
        return create_chunked_graph, write_chunked_graph_to_file
    except ImportError:
        filter_chunk_module = _import_module_from_path("filter_chunk", 
                                                     os.path.join(BASE_DIR, 'Utils', 'filter_chunk.py'))
        return filter_chunk_module.create_chunked_graph, filter_chunk_module.write_chunked_graph_to_file

def prepare_traffic_based_search(origin: str, destination: str, start_time: Union[str, datetime], 
                                chunked_graph_path: Optional[str] = None, 
                                output_path: Optional[str] = None) -> Optional[str]:
    """Prepare a graph for traffic-based path search."""
    # Set default paths
    chunked_graph_path = chunked_graph_path or os.path.join(DATA_DIR, 'temp_chunked_graph.txt')
    output_path = output_path or os.path.join(DATA_DIR, 'temp_chunked_graph.txt')
    
    try:
        # Get chunking functions
        create_chunked_graph, write_chunked_graph_to_file = _get_filter_chunk_functions()
        
        # Create and write chunked graph
        graph_file_path = os.path.join(DATA_DIR, 'graph.txt')
        filtered_nodes, filtered_edges, origin_id, dest_ids = create_chunked_graph(
            graph_file_path, origin, destination, margin_factor=0.2
        )
        write_chunked_graph_to_file(filtered_nodes, filtered_edges, origin_id, dest_ids, chunked_graph_path)
        
        # Update graph with traffic predictions
        integrator = TrafficPredictionIntegrator()
        success = integrator.update_graph_with_predictions(chunked_graph_path, start_time, output_path)
        
        return output_path if success else chunked_graph_path
    except Exception as e:
        print(f"Error preparing traffic-based search: {e}")
        return None
