import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import math

class TrafficDataset(Dataset):
    """Traffic Flow Dataset for Transformer model."""
    
    def __init__(self, X, y, categorical_indices=None):
        """Initialize the dataset.
        
        Args:
            X: Input sequences (shape: [n_samples, seq_len, n_features])
            y: Target sequences (shape: [n_samples, pred_len])
            categorical_indices: Dictionary mapping categorical feature names to their indices in X
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.categorical_indices = categorical_indices
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TrafficDataCollector:
    """Collects and prepares traffic flow data for Transformer model."""
    
    def __init__(self, data_path=None, embedding_dim=16):
        """Initialize the data collector.
        
        Args:
            data_path: Path to the transformed data directory
            embedding_dim: Dimension for categorical embeddings (8, 16, or 32)
        """
        if data_path is None:
            # Default path relative to the Transformer directory
            self.data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                         'Data', 'Transformed')
        else:
            self.data_path = data_path
            
        # Set embedding dimension
        if embedding_dim not in [8, 16, 32]:
            print(f"Warning: embedding_dim {embedding_dim} not in [8, 16, 32]. Using default value 16.")
            embedding_dim = 16
        self.embedding_dim = embedding_dim
        
        # Initialize encoders and scalers
        self.site_encoder = LabelEncoder()
        self.scat_type_encoder = LabelEncoder()
        self.day_type_encoder = LabelEncoder()
        self.school_count_scaler = StandardScaler()
        self.flow_scaler = StandardScaler()
    
    def _create_cyclical_features(self, df):
        """Create cyclical features for time and date.
        
        Args:
            df: DataFrame with date and time columns
            
        Returns:
            DataFrame with cyclical features added
        """
        # Convert date to datetime
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        
        # Extract features for day of week (0-6, Monday=0)
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Extract features for month (1-12)
        df['month'] = df['datetime'].dt.month
        df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
        
        # Extract features for hour and minute
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        
        # Normalize hour to 0-23 and minute to 0-59
        df['hour_norm'] = df['hour'] / 23
        df['minute_norm'] = df['minute'] / 59
        
        # Create cyclical features for hour and minute
        df['hour_sin'] = np.sin(2 * np.pi * df['hour_norm'])
        df['hour_cos'] = np.cos(2 * np.pi * df['hour_norm'])
        df['minute_sin'] = np.sin(2 * np.pi * df['minute_norm'])
        df['minute_cos'] = np.cos(2 * np.pi * df['minute_norm'])
        
        # Drop temporary columns
        df = df.drop(['datetime', 'day_of_week', 'month', 'hour', 'minute', 'hour_norm', 'minute_norm'], axis=1)
        
        return df
    
    def prepare_data(self, input_file, seq_len=24, pred_len=8, step_size=1, 
                    save=True, output_file='transformer_processed_data.pkl'):
        """Prepare data from long-format traffic data.
        
        Args:
            input_file: Name of the input CSV file in long format
            seq_len: Length of input sequence (in time steps)
            pred_len: Length of prediction sequence (in time steps)
            step_size: Step size for sliding window
            save: Whether to save the processed data
            output_file: Name of the output file
            
        Returns:
            Dictionary with processed data
        """
        # Load the data
        file_path = os.path.join(self.data_path, input_file)
        df = pd.read_csv(file_path)
        
        print(f"Loaded data with shape: {df.shape}")
        
        # Create cyclical features for date and time
        df = self._create_cyclical_features(df)
        
        # Process categorical features
        
        # 1. NB_SCATS_SITE: encode as indices for embedding
        df['NB_SCATS_SITE_encoded'] = self.site_encoder.fit_transform(df['NB_SCATS_SITE'])
        site_classes = len(self.site_encoder.classes_)
        
        # 4. scat_type: categorical encoding
        df['scat_type_encoded'] = self.scat_type_encoder.fit_transform(df['scat_type'])
        scat_type_classes = len(self.scat_type_encoder.classes_)
        
        # 5. day_type: encode as indices for embedding
        df['day_type_encoded'] = self.day_type_encoder.fit_transform(df['day_type'])
        day_type_classes = len(self.day_type_encoder.classes_)
        
        # 6 & 7. Standardize numerical features
        df['school_count_scaled'] = self.school_count_scaler.fit_transform(df[['school_count']])
        df['Flow_scaled'] = self.flow_scaler.fit_transform(df[['Flow']])
        
        # Create feature columns list
        feature_cols = [
            'day_of_week_sin', 'day_of_week_cos',  # Date features
            'month_sin', 'month_cos',
            'hour_sin', 'hour_cos',  # Time features
            'minute_sin', 'minute_cos',
            'NB_SCATS_SITE_encoded',  # Site ID (will be embedded in the model)
            'scat_type_encoded',  # Scat type
            'day_type_encoded',  # Day type (will be embedded in the model)
            'school_count_scaled',  # Standardized school count
            'Flow_scaled'  # Standardized flow (target)
        ]
        
        # Define categorical feature indices (for the model to know which features need embedding)
        categorical_indices = {
            'NB_SCATS_SITE': feature_cols.index('NB_SCATS_SITE_encoded'),
            'day_type': feature_cols.index('day_type_encoded')
        }
        
        # Define categorical feature metadata
        categorical_metadata = {
            'NB_SCATS_SITE': {
                'num_classes': site_classes,
                'embedding_dim': self.embedding_dim
            },
            'day_type': {
                'num_classes': day_type_classes,
                'embedding_dim': self.embedding_dim
            }
        }
        
        # Select features for modeling
        df_features = df[feature_cols].copy()
        
        # Get unique sites and sort chronologically
        unique_sites = df['NB_SCATS_SITE'].unique()
        df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        
        # Create sequences for each site
        X_list = []
        y_list = []
        sites_list = []
        dates_list = []
        
        for site in unique_sites:
            site_data = df[df['NB_SCATS_SITE'] == site].sort_values('date_time')
            site_features = df_features[df['NB_SCATS_SITE'] == site].values
            
            # Create sequences
            for i in range(0, len(site_features) - seq_len - pred_len + 1, step_size):
                X_list.append(site_features[i:i+seq_len])
                y_list.append(site_features[i+seq_len:i+seq_len+pred_len, -1])  # Only predict Flow
                sites_list.append(site)
                dates_list.append(site_data.iloc[i+seq_len]['date_time'])
        
        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)
        sites = np.array(sites_list)
        dates = np.array(dates_list)
        
        print(f"Created sequences: X shape={X.shape}, y shape={y.shape}")
        
        # Calculate additional metadata for the model
        input_dim = X.shape[2]  # Number of features
        output_dim = 1  # Predicting Flow only
        
        # Store the mappings, scalers, and metadata for later use
        encoders_scalers = {
            'site_encoder': self.site_encoder,
            'scat_type_encoder': self.scat_type_encoder,
            'day_type_encoder': self.day_type_encoder,
            'school_count_scaler': self.school_count_scaler,
            'flow_scaler': self.flow_scaler
        }
        
        # Prepare data dictionary
        data = {
            'X': X,
            'y': y,
            'sites': sites,
            'dates': dates,
            'encoders_scalers': encoders_scalers,
            'feature_cols': feature_cols,
            'categorical_indices': categorical_indices,
            'categorical_metadata': categorical_metadata,
            'input_dim': input_dim,
            'output_dim': output_dim
        }
        
        # Save processed data
        if save:
            output_path = os.path.join(self.data_path, output_file)
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            print(f"Saved processed data to {output_path}")
        
        return data
            
    def load_data(self, filename='transformer_processed_data.pkl'):
        """Load the prepared data for the Transformer model.
        
        Args:
            filename: Name of the file containing the prepared data
            
        Returns:
            Dictionary with processed data
        """
        file_path = os.path.join(self.data_path, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}. Run prepare_data method first.")
        
        # Load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        return data
    
    def get_data_loaders(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                         batch_size=32, shuffle=True, num_workers=0, 
                         random_state=42, filename='transformer_processed_data.pkl'):
        """Prepare data loaders for training, validation, and testing.
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            test_ratio: Ratio of test data
            batch_size: Batch size for dataloaders
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for dataloaders
            random_state: Random seed for reproducibility
            filename: Name of the file containing the prepared data
            
        Returns:
            Dictionary with data loaders and metadata
        """
        # Load the data
        data = self.load_data(filename)
        X, y = data['X'], data['y']
        categorical_indices = data.get('categorical_indices', None)
        categorical_metadata = data.get('categorical_metadata', None)
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        # Get dataset size
        n_samples = len(X)
        
        # Create indices for train/val/test split
        indices = np.random.permutation(n_samples)
        train_size = int(train_ratio * n_samples)
        val_size = int(val_ratio * n_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Extract train, validation, and test sets
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        
        # Create datasets
        train_dataset = TrafficDataset(X_train, y_train, categorical_indices)
        val_dataset = TrafficDataset(X_val, y_val, categorical_indices)
        test_dataset = TrafficDataset(X_test, y_test, categorical_indices)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'encoders_scalers': data['encoders_scalers'],
            'categorical_indices': categorical_indices,
            'categorical_metadata': categorical_metadata,
            'embedding_dim': self.embedding_dim,
            'input_dim': data.get('input_dim', X.shape[2]),
            'output_dim': data.get('output_dim', 1)
        }
    
    def inverse_transform_flow(self, normalized_data, encoders_scalers):
        """Inverse transform normalized flow data to original scale.
        
        Args:
            normalized_data: Normalized flow data (torch tensor or numpy array)
            encoders_scalers: Dictionary with encoders and scalers
            
        Returns:
            Flow data in original scale
        """
        # Convert to numpy if it's a torch tensor
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.detach().cpu().numpy()
        
        # Get flow scaler
        flow_scaler = encoders_scalers['flow_scaler']
        
        # Reshape for inverse transform if needed
        original_shape = normalized_data.shape
        reshaped_data = normalized_data.reshape(-1, 1)
        
        # Inverse transform
        original_data = flow_scaler.inverse_transform(reshaped_data)
        
        # Restore original shape
        return original_data.reshape(original_shape)

# Example usage
if __name__ == "__main__":
    # Create a data collector with embedding dimension 16
    data_collector = TrafficDataCollector(embedding_dim=16)
    
    try:
        # Prepare data from long-format traffic data
        data_collector.prepare_data(
            input_file='sample_long_format_revised.csv',
            seq_len=24,  # 6 hours (15-minute intervals)
            pred_len=8,  # 2 hours prediction horizon
            step_size=4  # Create a new sequence every hour
        )
        
        # Get data loaders
        data = data_collector.get_data_loaders(batch_size=32)
        
        # Print some information about the data
        train_loader = data['train_loader']
        val_loader = data['val_loader']
        test_loader = data['test_loader']
        categorical_metadata = data['categorical_metadata']
        
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Number of testing batches: {len(test_loader)}")
        
        # Get a batch from the training loader
        X_batch, y_batch = next(iter(train_loader))
        print(f"Input batch shape: {X_batch.shape}")
        print(f"Target batch shape: {y_batch.shape}")
        
        # Print embedding information
        print("\nCategorical Feature Metadata:")
        for feature, metadata in categorical_metadata.items():
            print(f"  {feature}: {metadata['num_classes']} classes, embedding dim={metadata['embedding_dim']}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the long-format traffic data file exists in the Data/Transformed directory.")