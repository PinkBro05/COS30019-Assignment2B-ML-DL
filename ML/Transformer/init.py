"""
Initialization module for the Transformer-based traffic flow prediction model.
This module provides functions to load the model and make predictions.
"""

import os
import sys
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
from datetime import datetime

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ML.Transformer.utils.traffic_data_collector import TrafficDataCollector
from ML.Transformer.models.model import TransformerModel

# Constants
DEFAULT_MODEL_NAME = '2024_4_steps_5_epochs_transformer_traffic_model.pth'
DEFAULT_DATASET_NAME = '_sample_final_time_series.csv'
DEFAULT_EMBEDDING_DIM = 16
DEFAULT_SEQ_LEN = 24
FALLBACK_SITE_ID = 100  # Site ID used when sample data only contains one site

# Model configuration
MODEL_CONFIG = {
    'input_dim': 13,
    'd_model': 64,
    'num_heads': 8,
    'num_layers': 3,
    'd_ff': 256,
    'dropout': 0.1,
    'output_size': 1
}

# Feature columns in order (must match training order)
FEATURE_COLUMNS = [
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

# Categorical feature indices
CATEGORICAL_INDICES = {
    'NB_SCATS_SITE': FEATURE_COLUMNS.index('NB_SCATS_SITE_encoded'),
    'day_type': FEATURE_COLUMNS.index('day_type_encoded')
}

class TrafficFlowPredictor:
    """Transformer-based traffic flow predictor."""
    
    def __init__(self, model_path: Optional[str] = None, data_path: Optional[str] = None, 
                 embedding_dim: int = DEFAULT_EMBEDDING_DIM):
        """Initialize the traffic flow predictor."""
        self.model_path = model_path or self._get_default_model_path()
        self.embedding_dim = embedding_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Validate model file exists
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        
        # Initialize components
        self.data_collector = TrafficDataCollector(data_path, embedding_dim)
        self.categorical_metadata, self.categorical_indices = self._extract_model_metadata()
        self.model = None  # Lazy loading
        
    def _get_default_model_path(self) -> str:
        """Get the default model path."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_models', DEFAULT_MODEL_NAME)
    
    def _get_default_dataset_path(self) -> str:
        """Get the default dataset path."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base_dir, 'ML', 'Data', 'Transformed', DEFAULT_DATASET_NAME)
    
    def _extract_model_metadata(self) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
        """Extract embedding dimensions and categorical indices from saved model."""
        state_dict = torch.load(self.model_path, map_location=torch.device('cpu'))
        
        categorical_metadata = {}
        for key in state_dict.keys():
            if 'embedding_layers' in key and 'weight' in key:
                feature_name = key.split('.')[1]
                num_classes, embedding_dim = state_dict[key].shape
                categorical_metadata[feature_name] = {
                    'num_classes': num_classes,
                    'embedding_dim': embedding_dim
                }
        return categorical_metadata, CATEGORICAL_INDICES

    def load_model(self) -> TransformerModel:
        """Load the trained model."""
        if self.model is not None:
            return self.model
            
        self.model = TransformerModel(
            categorical_metadata=self.categorical_metadata,
            categorical_indices=self.categorical_indices,
            **MODEL_CONFIG
        )
        
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        return self.model

    def _process_site_data(self, df: pd.DataFrame, site_id: Union[str, int], 
                          timestamp: Union[str, datetime], seq_len: int) -> pd.DataFrame:
        """Process and filter site data for the given timestamp."""
        site_df = df[df['NB_SCATS_SITE'] == site_id].copy()
        if len(site_df) == 0:
            raise ValueError(f"No data found for site {site_id}")
        
        # Prepare datetime and filter historical data
        site_df['datetime'] = pd.to_datetime(site_df['date'] + ' ' + site_df['time'])
        site_df = site_df.sort_values('datetime')
        
        pred_datetime = pd.to_datetime(timestamp) if isinstance(timestamp, str) else timestamp
        historical_df = site_df[site_df['datetime'] < pred_datetime].copy()
        
        if len(historical_df) < seq_len:
            raise ValueError(f"Not enough historical data for site {site_id}. Need at least {seq_len} time steps.")
        
        return historical_df.iloc[-seq_len:].reset_index(drop=True)

    def _apply_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all data transformations (cyclical features, encoding, scaling)."""
        df_processed = self.data_collector._create_cyclical_features(df)
        
        # Apply encodings and scaling
        transformations = [
            ('NB_SCATS_SITE_encoded', self.data_collector.site_encoder, 'NB_SCATS_SITE'),
            ('scat_type_encoded', self.data_collector.scat_type_encoder, 'scat_type'),
            ('day_type_encoded', self.data_collector.day_type_encoder, 'day_type'),
            ('school_count_scaled', self.data_collector.school_count_scaler, ['school_count']),
            ('Flow_scaled', self.data_collector.flow_scaler, ['Flow'])
        ]
        
        for target_col, transformer, source_col in transformations:
            df_processed[target_col] = transformer.transform(df_processed[source_col])
        
        return df_processed

    def prepare_data_for_inference(self, df: pd.DataFrame, site_id: Union[str, int], 
                                 timestamp: Union[str, datetime], seq_len: int = DEFAULT_SEQ_LEN) -> Dict[str, Any]:
        """Prepare data for inference."""
        historical_df = self._process_site_data(df, site_id, timestamp, seq_len)
        df_processed = self._apply_transformations(historical_df)
        
        # Extract features and add batch dimension
        features = df_processed[FEATURE_COLUMNS].values
        X = np.expand_dims(features, axis=0)
        
        return {
            'X': X,
            'site_id': site_id,
            'features': features,            
            'flow_scaler': self.data_collector.flow_scaler
        }

    def _fit_encoders_if_needed(self, df: pd.DataFrame) -> None:
        """Fit encoders and scalers if not already fitted."""
        if hasattr(self.data_collector.site_encoder, 'classes_'):
            return
            
        encoders_and_columns = [
            (self.data_collector.site_encoder, 'NB_SCATS_SITE'),
            (self.data_collector.scat_type_encoder, 'scat_type'),
            (self.data_collector.day_type_encoder, 'day_type'),
            (self.data_collector.school_count_scaler, ['school_count']),
            (self.data_collector.flow_scaler, ['Flow'])
        ]
        
        for encoder, column in encoders_and_columns:
            encoder.fit(df[column])

    def _make_prediction(self, inference_data: Dict[str, Any], num_steps: int) -> np.ndarray:
        """Make prediction using the loaded model."""
        X_tensor = torch.FloatTensor(inference_data['X']).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor, pred_len=num_steps)
            return predictions.cpu().numpy()

    def predict_flow(self, site_id: Union[str, int], timestamp: Union[str, datetime], 
                    dataset_path: Optional[str] = None, num_steps: int = 1) -> Dict[str, Any]:
        """Predict traffic flow for a specific site and time."""
        # Load model and dataset
        if self.model is None:
            self.load_model()
            
        dataset_path = dataset_path or self._get_default_dataset_path()
        df = pd.read_csv(dataset_path)
        
        # Fit encoders if needed
        self._fit_encoders_if_needed(df)
        
        try:
            # Use fallback site ID if needed (for sample data compatibility)
            actual_site_id = FALLBACK_SITE_ID if site_id not in df['NB_SCATS_SITE'].values else site_id
            
            # Prepare data and make prediction
            inference_data = self.prepare_data_for_inference(df, actual_site_id, timestamp)
            predictions = self._make_prediction(inference_data, num_steps)
            
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