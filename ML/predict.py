"""
Traffic Flow Prediction Module for Integration with Main Application
This module wraps the Transformer inference functionality and provides
time-based cost calculations for pathfinding algorithms.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
transformer_dir = os.path.join(current_dir, "Transformer")
utils_dir = os.path.join(os.path.dirname(current_dir), "Utils")

# Add to path at the beginning
sys.path.insert(0, transformer_dir)
sys.path.insert(0, utils_dir)

# Add to path for Utils module
sys.path.append(os.path.join(os.path.dirname(current_dir), "Utils"))

# Import necessary modules
from ML.Transformer.utils.traffic_data_collector import TrafficDataCollector
from ML.Transformer.models.model import TransformerModel
from ML.Transformer.inference import (
    get_embedding_dimensions_from_model_file, 
    load_model,
    prepare_data_for_inference,
    predict_next_steps
)

# Import math functions
import importlib.util
maths_path = os.path.join(os.path.dirname(current_dir), "Utils", "maths.py")
spec = importlib.util.spec_from_file_location("maths", maths_path)
maths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maths)
flow_to_velocity = maths.flow_to_velocity
velocity_to_time = maths.velocity_to_time


class TrafficFlowPredictor:
    """
    A wrapper class for the Transformer model to predict traffic flow
    and calculate time-based costs for pathfinding.
    """
    
    def __init__(self, 
                 model_path: str = None,
                 data_path: str = None):
        """
        Initialize the TrafficFlowPredictor.
        
        Args:
            model_path: Path to the trained Transformer model
            data_path: Path to the traffic data CSV file
        """
        self.current_dir = current_dir
        
        # Set default paths if not provided
        if model_path is None:
            self.model_path = os.path.join(
                transformer_dir, 
                "save_models", 
                "2024_4_steps_5_epochs_transformer_traffic_model.pth"
            )
        else:
            self.model_path = model_path
            
        if data_path is None:
            self.data_path = os.path.join(
                current_dir,
                "Data",
                "Transformed",
                "_sample_final_time_series.csv"
            )
        else:
            self.data_path = data_path
        
        # Model parameters (must match training configuration)
        self.model_params = {
            'input_dim': 16,  # Will be updated based on data
            'd_model': 64,
            'num_heads': 8,
            'num_layers': 3,
            'd_ff': 256,
            'dropout': 0.1        }
        
        # Initialize components
        self.model = None
        self.data_collector = None
        self.df = None
        self.categorical_metadata = None
        self.categorical_indices = None
        
        # Load data first, then get metadata from model, then load model
        self._load_data()
        self._get_model_metadata()
        self._load_model()
    
    def _load_data(self):
        """Load and prepare the traffic data."""
        try:
            print(f"Loading traffic data from: {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            
            # Initialize data collector with embedding dimension that matches model
            self.data_collector = TrafficDataCollector(
                data_path=os.path.dirname(self.data_path),
                embedding_dim=16
            )
            
            # Fit the encoders and scalers on the data (similar to inference.py)
            self.data_collector.site_encoder.fit(self.df['NB_SCATS_SITE'])
            self.data_collector.scat_type_encoder.fit(self.df['scat_type'])
            self.data_collector.day_type_encoder.fit(self.df['day_type'])
            self.data_collector.school_count_scaler.fit(self.df[['school_count']])
            self.data_collector.flow_scaler.fit(self.df[['Flow']])
            
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def _get_model_metadata(self):
        """Get categorical metadata from the model file."""
        try:
            print(f"Getting model metadata from: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            # Get categorical metadata from the model file
            self.categorical_metadata, self.categorical_indices = get_embedding_dimensions_from_model_file(
                self.model_path
            )
            
            print("Model metadata loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model metadata: {e}")
            raise
    
    def _load_model(self):
        """Load the trained Transformer model."""
        try:
            print(f"Loading model from: {self.model_path}")
            
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Create a dummy args object for prepare_data_for_inference
            args = type('Args', (), {
                'num_steps': 4,
                'embedding_dim': 16,
                'd_model': self.model_params['d_model'],
                'num_heads': self.model_params['num_heads'],
                'num_layers': self.model_params['num_layers'],
                'd_ff': self.model_params['d_ff'],
                'dropout': self.model_params['dropout']
            })()
            
            # Get a valid index from the data to prepare inference data
            valid_indices = [idx for idx in self.df.index if idx >= 24]
            if not valid_indices:
                raise ValueError("Not enough data for model initialization (need at least 24 rows)")
            
            # Prepare data for inference to get the correct input dimension
            model_metadata = {
                'categorical_metadata': self.categorical_metadata,
                'categorical_indices': self.categorical_indices
            }
            
            inference_data = prepare_data_for_inference(
                self.data_collector,
                self.df,
                valid_indices[0],  # Use first valid index
                args,
                model_metadata
            )
            
            # Update input_dim based on the processed data shape
            self.model_params['input_dim'] = inference_data['X'].shape[2]
            print(f"Model input dimension: {self.model_params['input_dim']}")
            
            # Now load the model with correct parameters
            self.model = load_model(
                self.model_path,
                self.model_params,
                self.categorical_metadata,
                self.categorical_indices
            )
            
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_traffic_flow(self, 
                           start_time: str,
                           scats_sites: List[str],
                           num_steps: int = 4) -> Dict[str, List[float]]:
        """
        Predict traffic flow for specified SCATS sites starting from a given time.
        
        Args:
            start_time: Start time in format 'YYYY-MM-DD HH:MM:SS'
            scats_sites: List of SCATS site numbers to predict for
            num_steps: Number of 15-minute time steps to predict (default: 4 for 1 hour)
        
        Returns:
            Dictionary mapping SCATS site to list of predicted flow values
        """
        try:
            # Parse start time
            start_datetime = pd.to_datetime(start_time)
            
            predictions = {}
            
            for site in scats_sites:
                site_predictions = self._predict_for_site(
                    site, start_datetime, num_steps
                )
                predictions[site] = site_predictions
            
            return predictions
            
        except Exception as e:
            print(f"Error predicting traffic flow: {e}")
            return {}
    
    def _predict_for_site(self, 
                         site: str, 
                         start_datetime: pd.Timestamp,
                         num_steps: int) -> List[float]:
        """
        Predict traffic flow for a specific SCATS site.
        
        Args:
            site: SCATS site number
            start_datetime: Start datetime for prediction
            num_steps: Number of time steps to predict
        
        Returns:
            List of predicted flow values
        """
        try:
            # Find a suitable starting point in the data for this site
            site_data = self.df[self.df['NB_SCATS_SITE'] == int(site)]
            
            if len(site_data) == 0:
                print(f"No data found for site {site}")
                return [0.0] * num_steps
            
            # Use a random valid index for this site as the starting point
            # In practice, you might want to find the closest historical point
            import random
            valid_indices = site_data.index.tolist()
            
            # Ensure we have enough historical data (need 24 steps for input)
            valid_indices = [idx for idx in valid_indices if idx >= 24]
            
            if not valid_indices:
                print(f"Not enough historical data for site {site}")
                return [0.0] * num_steps
            
            index = random.choice(valid_indices)
            
            # Prepare data for inference
            model_metadata = {
                'categorical_metadata': self.categorical_metadata,
                'categorical_indices': self.categorical_indices
            }
            
            prepared_data = prepare_data_for_inference(
                self.data_collector,
                self.df,
                index,
                type('Args', (), {
                    'num_steps': num_steps,
                    'embedding_dim': 16,
                    'd_model': self.model_params['d_model'],
                    'num_heads': self.model_params['num_heads'],
                    'num_layers': self.model_params['num_layers'],
                    'd_ff': self.model_params['d_ff'],
                    'dropout': self.model_params['dropout']
                })(),
                model_metadata
            )
            
            # Make predictions
            predictions = predict_next_steps(
                self.model,
                prepared_data['X'],
                num_steps=num_steps,
                future_features=prepared_data['future_features'],
                encoders_scalers=prepared_data['encoders_scalers']
            )
            
            return predictions.tolist() if isinstance(predictions, np.ndarray) else predictions
            
        except Exception as e:
            print(f"Error predicting for site {site}: {e}")
            return [0.0] * num_steps
    
    def calculate_time_based_costs(self,
                                 scats_predictions: Dict[str, List[float]],
                                 graph_edges: Dict[Tuple[str, str], Dict],
                                 distance_key: str = 'distance') -> Dict[Tuple[str, str], float]:
        """
        Calculate time-based costs for graph edges using traffic flow predictions.
        
        Args:
            scats_predictions: Dictionary of SCATS site predictions
            graph_edges: Dictionary of graph edges with their properties
            distance_key: Key for distance in edge properties
        
        Returns:
            Dictionary mapping edge tuples to time-based costs
        """
        time_costs = {}
        
        for edge, properties in graph_edges.items():
            try:
                # Get edge properties
                distance = properties.get(distance_key, 100)  # Default 100m if no distance
                
                # For simplicity, use the first node's SCATS site if available
                # In practice, you might want to interpolate between nearby sites
                node1, node2 = edge
                
                # Try to find predictions for nodes (assuming node names might match SCATS sites)
                flow_rate = self._get_flow_for_edge(edge, scats_predictions)
                
                # Convert flow to velocity (assuming under capacity)
                try:
                    velocity = flow_to_velocity(flow_rate, status=True)
                except ValueError:
                    # If flow rate is invalid, use a default velocity
                    velocity = 30  # Default 30 km/h
                
                # Convert velocity to time
                try:
                    time_cost = velocity_to_time(velocity, distance)
                except ValueError:
                    # If velocity is invalid, use distance as fallback
                    time_cost = distance / 10  # Rough approximation
                
                time_costs[edge] = time_cost
                
            except Exception as e:
                print(f"Error calculating cost for edge {edge}: {e}")
                # Fallback to distance-based cost
                time_costs[edge] = properties.get(distance_key, 100)
        
        return time_costs
    
    def _get_flow_for_edge(self, 
                          edge: Tuple[str, str], 
                          scats_predictions: Dict[str, List[float]]) -> float:
        """
        Get predicted flow rate for an edge.
        
        Args:
            edge: Tuple of (node1, node2)
            scats_predictions: Dictionary of SCATS predictions
        
        Returns:
            Average hourly flow rate for the edge
        """
        node1, node2 = edge
        
        # Try to find predictions for either node
        # This is a simplified approach - in practice you might need
        # more sophisticated mapping between graph nodes and SCATS sites
        for node in [node1, node2]:
            if node in scats_predictions:
                # Sum 4x15-minute predictions to get 1-hour total
                hourly_flow = sum(scats_predictions[node])
                return max(hourly_flow, 1)  # Ensure minimum flow of 1
        
        # If no direct match, try to find nearby SCATS sites
        # For now, use a default flow rate
        return 500  # Default flow rate (vehicles/hour)
    
    def get_scats_in_bounds(self, 
                          min_lat: float, 
                          max_lat: float,
                          min_lon: float, 
                          max_lon: float) -> List[str]:
        """
        Get SCATS sites within geographical bounds.
        
        Args:
            min_lat: Minimum latitude
            max_lat: Maximum latitude  
            min_lon: Minimum longitude
            max_lon: Maximum longitude
        
        Returns:
            List of SCATS site numbers within the bounds
        """
        try:
            # Load SCATS location data
            geojson_path = os.path.join(
                os.path.dirname(current_dir),
                "Data",
                "Traffic_Lights.geojson"
            )
            
            if os.path.exists(geojson_path):
                import geopandas as gpd
                gdf = gpd.read_file(geojson_path)
                
                # Filter by bounds
                filtered = gdf[
                    (gdf.geometry.y >= min_lat) & 
                    (gdf.geometry.y <= max_lat) &
                    (gdf.geometry.x >= min_lon) & 
                    (gdf.geometry.x <= max_lon)
                ]
                
                return filtered['SITE_NO'].astype(str).tolist()
            else:
                print(f"GeoJSON file not found: {geojson_path}")
                return []
                
        except Exception as e:
            print(f"Error getting SCATS in bounds: {e}")
            return []


def integrate_with_pathfinding(start_time: str,
                             origin: str,
                             destination: str,
                             min_lat: float = -37.9,
                             max_lat: float = -37.7,
                             min_lon: float = 144.8,
                             max_lon: float = 145.1) -> Dict[Tuple[str, str], float]:
    """
    Main integration function to get time-based costs for pathfinding.
    
    Args:
        start_time: Start time for prediction in format 'YYYY-MM-DD HH:MM:SS'
        origin: Origin node/SCATS site
        destination: Destination node/SCATS site
        min_lat: Minimum latitude for filtering
        max_lat: Maximum latitude for filtering
        min_lon: Minimum longitude for filtering  
        max_lon: Maximum longitude for filtering
    
    Returns:
        Dictionary mapping edge tuples to time-based costs
    """
    try:
        # Initialize predictor
        predictor = TrafficFlowPredictor()
        
        # Get SCATS sites in the filtered area
        scats_sites = predictor.get_scats_in_bounds(min_lat, max_lat, min_lon, max_lon)
        
        if not scats_sites:
            print("No SCATS sites found in the specified bounds")
            return {}
        
        print(f"Found {len(scats_sites)} SCATS sites in bounds")
        
        # Get traffic flow predictions for all SCATS sites
        predictions = predictor.predict_traffic_flow(start_time, scats_sites)
        
        # For demonstration, create a simple graph structure
        # In practice, this would come from your actual graph data
        graph_edges = _create_sample_graph_edges(scats_sites)
        
        # Calculate time-based costs
        time_costs = predictor.calculate_time_based_costs(predictions, graph_edges)
        
        print(f"Calculated time-based costs for {len(time_costs)} edges")
        return time_costs
        
    except Exception as e:
        print(f"Error in pathfinding integration: {e}")
        return {}


def _create_sample_graph_edges(scats_sites: List[str]) -> Dict[Tuple[str, str], Dict]:
    """
    Create sample graph edges for demonstration.
    In practice, this would load from your actual graph file.
    
    Args:
        scats_sites: List of SCATS site numbers
    
    Returns:
        Dictionary of graph edges with properties
    """
    edges = {}
    
    # Create edges between consecutive sites (simplified)
    for i in range(len(scats_sites) - 1):
        site1 = scats_sites[i]
        site2 = scats_sites[i + 1]
        
        # Add bidirectional edges
        edges[(site1, site2)] = {
            'distance': 200,  # Default 200m
            'cost': 200,      # Original distance-based cost
            'weight': 200
        }
        edges[(site2, site1)] = {
            'distance': 200,
            'cost': 200,
            'weight': 200
        }
    
    return edges


# Convenience functions for use in main.py
def get_traffic_predictions(start_time: str, 
                          origin: str, 
                          destination: str) -> Dict[str, any]:
    """
    Get traffic predictions and time-based costs for pathfinding.
    
    Args:
        start_time: Start time in format 'YYYY-MM-DD HH:MM:SS'
        origin: Origin SCATS site
        destination: Destination SCATS site
    
    Returns:
        Dictionary containing predictions and time costs
    """
    try:
        # Melbourne CBD approximate bounds
        time_costs = integrate_with_pathfinding(
            start_time=start_time,
            origin=origin,
            destination=destination,
            min_lat=-37.9,
            max_lat=-37.7,
            min_lon=144.8,
            max_lon=145.1
        )
        
        return {
            'time_costs': time_costs,
            'start_time': start_time,
            'status': 'success'
        }
        
    except Exception as e:
        print(f"Error getting traffic predictions: {e}")
        return {
            'time_costs': {},
            'start_time': start_time,
            'status': 'error',
            'error': str(e)
        }


if __name__ == "__main__":
    # Test the prediction functionality
    print("Testing Traffic Flow Predictor...")
    
    # Test with current time
    start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    result = get_traffic_predictions(
        start_time=start_time,
        origin="1001",
        destination="1002"
    )
    
    print(f"Result: {result}")
