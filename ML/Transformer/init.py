"""
Initialization module for the Transformer-based traffic flow prediction model.
This module provides functions to load the model and make predictions.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ML.Transformer.utils.traffic_data_collector import TrafficDataCollector
from ML.Transformer.models.model import TransformerModel

class TrafficFlowPredictor:
    def __init__(self, model_path=None, data_path=None, embedding_dim=16):
        """
        Initialize the traffic flow predictor.
        
        Args:
            model_path: Path to the model file (default: use the latest model in save_models)
            data_path: Path to the data directory (default: use the default path)
            embedding_dim: Dimension for categorical embeddings (default: 16)
        """
        # Set default model path if not provided
        if model_path is None:
            model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'save_models',
                '2024_4_steps_5_epochs_transformer_traffic_model.pth'
            )
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        self.model_path = model_path
        self.embedding_dim = embedding_dim
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create data collector
        self.data_collector = TrafficDataCollector(data_path, embedding_dim)
        
        # Load model metadata
        self.categorical_metadata, self.categorical_indices = self._get_embedding_dimensions_from_model()
        
        # Prepare model parameters
        self.model_params = {
            'input_dim': 13,  # Default value, will be updated when loading data
            'd_model': 64,
            'num_heads': 8,
            'num_layers': 3,
            'd_ff': 256,
            'dropout': 0.1,
            'output_size': 1
        }
        
        # Initialize model (will be loaded on first prediction)
        self.model = None
    
    def _get_embedding_dimensions_from_model(self):
        """Extract embedding dimensions from the saved model file."""
        # Load the model state dict
        state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        
        # Extract embedding dimensions
        categorical_metadata = {}
        categorical_indices = {}
        
        # Find all embedding layers in the state dict
        for key in state_dict.keys():
            if 'embedding_layers' in key and 'weight' in key:
                # Extract feature name from key, format: embedding_layers.{feature_name}.weight
                feature_name = key.split('.')[1]
                
                # Get weight shape
                weight_shape = state_dict[key].shape
                num_classes, embedding_dim = weight_shape
                
                # Store metadata
                categorical_metadata[feature_name] = {
                    'num_classes': num_classes,
                    'embedding_dim': embedding_dim
                }
        
        # Define feature order
        feature_cols = [
            'day_of_week_sin', 'day_of_week_cos',  # Date features
            'month_sin', 'month_cos',
            'hour_sin', 'hour_cos',  # Time features
            'minute_sin', 'minute_cos',
            'NB_SCATS_SITE_encoded',  # Site ID
            'scat_type_encoded',  # Scat type
            'day_type_encoded',  # Day type
            'school_count_scaled',  # Standardized school count
            'Flow_scaled'  # Standardized flow (target)
        ]
        
        # Define categorical feature indices
        categorical_indices = {
            'NB_SCATS_SITE': feature_cols.index('NB_SCATS_SITE_encoded'),
            'day_type': feature_cols.index('day_type_encoded')
        }
        
        return categorical_metadata, categorical_indices
    
    def load_model(self):
        """Load the trained model."""
        # Create model with dimensions from the saved model
        self.model = TransformerModel(
            input_dim=self.model_params['input_dim'],
            d_model=self.model_params['d_model'],
            num_heads=self.model_params['num_heads'],
            d_ff=self.model_params['d_ff'],
            output_size=self.model_params['output_size'],
            num_layers=self.model_params['num_layers'],
            dropout=self.model_params['dropout'],
            categorical_metadata=self.categorical_metadata,
            categorical_indices=self.categorical_indices
        )
        
        # Load the saved state dict
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        
        return self.model
    
    def prepare_data_for_inference(self, df, site_id, timestamp, seq_len=24):
        """
        Prepare data for inference.
        
        Args:
            df: DataFrame with historical traffic data
            site_id: SCATS site ID to predict for
            timestamp: Datetime to predict for
            seq_len: Number of time steps to use as input (default: 24)
            
        Returns:
            Dictionary with processed data for inference
        """
        # Filter data for the specific site
        site_df = df[df['NB_SCATS_SITE'] == site_id].copy()
        
        if len(site_df) == 0:
            raise ValueError(f"No data found for site {site_id}")
        
        # Sort by datetime to ensure correct sequence
        site_df['datetime'] = pd.to_datetime(site_df['date'] + ' ' + site_df['time'])
        site_df = site_df.sort_values('datetime')
        
        # Get the timestamp as datetime
        if isinstance(timestamp, str):
            pred_datetime = pd.to_datetime(timestamp)
        else:
            pred_datetime = timestamp
            
        # Find data before the prediction time
        historical_df = site_df[site_df['datetime'] < pred_datetime].copy()
        
        if len(historical_df) < seq_len:
            raise ValueError(f"Not enough historical data for site {site_id}. Need at least {seq_len} time steps.")
        
        # Take the most recent seq_len time steps
        historical_df = historical_df.iloc[-seq_len:].reset_index(drop=True)
        
        # Create cyclical features
        df_processed = self.data_collector._create_cyclical_features(historical_df)
        
        # Process categorical features
        df_processed['NB_SCATS_SITE_encoded'] = self.data_collector.site_encoder.transform(df_processed['NB_SCATS_SITE'])
        df_processed['scat_type_encoded'] = self.data_collector.scat_type_encoder.transform(df_processed['scat_type'])
        df_processed['day_type_encoded'] = self.data_collector.day_type_encoder.transform(df_processed['day_type'])
        
        # Standardize numerical features
        df_processed['school_count_scaled'] = self.data_collector.school_count_scaler.transform(df_processed[['school_count']])
        df_processed['Flow_scaled'] = self.data_collector.flow_scaler.transform(df_processed[['Flow']])
        
        # Create feature columns list (must match the order in training)
        feature_cols = [
            'day_of_week_sin', 'day_of_week_cos',  # Date features
            'month_sin', 'month_cos',
            'hour_sin', 'hour_cos',  # Time features
            'minute_sin', 'minute_cos',
            'NB_SCATS_SITE_encoded',  # Site ID
            'scat_type_encoded',  # Scat type
            'day_type_encoded',  # Day type
            'school_count_scaled',  # Standardized school count
            'Flow_scaled'  # Standardized flow (target)
        ]
        
        # Extract features
        features = df_processed[feature_cols].values
        
        # Add batch dimension for model input
        X = np.expand_dims(features, axis=0)
        
        return {
            'X': X,
            'site_id': site_id,
            'features': features,
            'flow_scaler': self.data_collector.flow_scaler
        }
    
    def predict_flow(self, site_id, timestamp, dataset_path=None, num_steps=1):
        """
        Predict traffic flow for a specific site and time.
        
        Args:
            site_id: SCATS site ID to predict for
            timestamp: Datetime to predict for (can be string or datetime object)
            dataset_path: Path to the dataset file (default: use _sample_final_time_series.csv)
            num_steps: Number of steps to predict (default: 1)
            
        Returns:
            Dictionary with predicted flow values and metadata
        """
        # Load model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Set default dataset path if not provided
        if dataset_path is None:
            dataset_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'ML', 'Data', 'Transformed', '_sample_final_time_series.csv'
            )
        
        # Load dataset
        df = pd.read_csv(dataset_path)
        
        # Fit encoders with the data if not already fitted
        if not hasattr(self.data_collector.site_encoder, 'classes_'):
            self.data_collector.site_encoder.fit(df['NB_SCATS_SITE'])
            self.data_collector.scat_type_encoder.fit(df['scat_type'])
            self.data_collector.day_type_encoder.fit(df['day_type'])
            self.data_collector.school_count_scaler.fit(df[['school_count']])
            self.data_collector.flow_scaler.fit(df[['Flow']])
        
        try:
            # Prepare data for inference
            # inference_data = self.prepare_data_for_inference(df, site_id, timestamp)
            
            # Fix site_id to 100 since the sample data only contains this site
            site_id = 100
            inference_data = self.prepare_data_for_inference(df, site_id, timestamp)
            
            # Make prediction
            X_tensor = torch.FloatTensor(inference_data['X']).to(self.device)
            
            with torch.no_grad():
                predictions = self.model(X_tensor, pred_len=num_steps)
                predictions = predictions.cpu().numpy()
            
            # Denormalize predictions
            predictions_reshaped = predictions.reshape(-1, 1)
            predicted_flows = self.data_collector.flow_scaler.inverse_transform(predictions_reshaped).flatten()
            
            return {
                'site_id': site_id,
                'timestamp': timestamp,
                'predicted_flows': predicted_flows.tolist()
            }
            
        except Exception as e:
            print(f"Error predicting flow for site {site_id}: {e}")
            return {
                'site_id': site_id,
                'timestamp': timestamp,
                'predicted_flows': None,
                'error': str(e)
            }