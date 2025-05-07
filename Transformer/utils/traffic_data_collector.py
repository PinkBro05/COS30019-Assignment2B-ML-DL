import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
import math

class TrafficDataset(Dataset):
    """Traffic Flow Dataset for Transformer model."""
    
    def __init__(self, X, y, categorical_indices=None, device=None):
        """Initialize the dataset.
        
        Args:
            X: Input sequences (shape: [n_samples, seq_len, n_features])
            y: Target sequences (shape: [n_samples, pred_len])
            categorical_indices: Dictionary mapping categorical feature names to their indices in X
            device: The device to put tensors on ('cuda' or 'cpu')
        """
        # Convert to tensors if they're numpy arrays
        if not isinstance(X, torch.Tensor):
            X = torch.FloatTensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.FloatTensor(y)
            
        # Store the tensors - do NOT move to device here for compatibility with DataLoader's pin_memory
        self.X = X
        self.y = y
            
        self.categorical_indices = categorical_indices
        self.device = device
        
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
        
        # Store metadata for inverse transforms
        self.metadata = {}
    
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
    
    def process_data(self, input_file, seq_len=24, pred_len=4, step_size=4):
        """Process data from a CSV file.
        
        Args:
            input_file: Path to the CSV file
            seq_len: Length of input sequence (in time steps)
            pred_len: Length of prediction sequence (in time steps)
            step_size: Step size for sliding window
            
        Returns:
            Dictionary with processed data
        """
        # Check if file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"CSV file not found at {input_file}")
        
        # Load the CSV file
        df = pd.read_csv(input_file)
        print(f"Loaded data with shape: {df.shape}")
        
        # Create cyclical features for date and time
        df = self._create_cyclical_features(df)
        
        # Process categorical features
        # 1. NB_SCATS_SITE: encode as indices for embedding
        df['NB_SCATS_SITE_encoded'] = self.site_encoder.fit_transform(df['NB_SCATS_SITE'])
        site_classes = len(self.site_encoder.classes_)
        
        # 2. scat_type: categorical encoding
        df['scat_type_encoded'] = self.scat_type_encoder.fit_transform(df['scat_type'])
        scat_type_classes = len(self.scat_type_encoder.classes_)
        
        # 3. day_type: encode as indices for embedding
        df['day_type_encoded'] = self.day_type_encoder.fit_transform(df['day_type'])
        day_type_classes = len(self.day_type_encoder.classes_)
        
        # 4. Standardize numerical features
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
        
        # Store the encoders and scalers for later use
        encoders_scalers = {
            'site_encoder': self.site_encoder,
            'scat_type_encoder': self.scat_type_encoder,
            'day_type_encoder': self.day_type_encoder,
            'school_count_scaler': self.school_count_scaler,
            'flow_scaler': self.flow_scaler
        }
        
        # Store metadata for later use
        self.metadata = {
            'sites': sites,
            'dates': dates,
            'feature_cols': feature_cols
        }
        
        # Prepare data dictionary
        data = {
            'X': X,
            'y': y,
            'encoders_scalers': encoders_scalers,
            'categorical_indices': categorical_indices,
            'categorical_metadata': categorical_metadata,
            'input_dim': input_dim,
            'output_dim': output_dim
        }
        
        return data
    
    def process_data_in_chunks(self, input_file, chunk_size=100000, seq_len=24, pred_len=4, step_size=4, initialize_encoders=True):
        """Process data from a CSV file in chunks to handle large datasets.
        
        Args:
            input_file: Path to the CSV file
            chunk_size: Number of rows to process in each chunk
            seq_len: Length of input sequence (in time steps)
            pred_len: Length of prediction sequence (in time steps)
            step_size: Step size for sliding window
            initialize_encoders: Whether to initialize encoders by scanning the full dataset first
            
        Returns:
            Generator that yields processed data chunks
        """
        # Check if file exists
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"CSV file not found at {input_file}")
        
        # Initialize encoders for consistent transformations across chunks
        # This should only be done once before processing any chunks
        if initialize_encoders:
            self._initialize_encoders_on_full_dataset(input_file)
        
        # Create chunks using pandas
        chunk_reader = pd.read_csv(input_file, chunksize=chunk_size)
        chunk_count = 0
        
        # Process each chunk
        for chunk_df in chunk_reader:
            chunk_count += 1
            print(f"Processing chunk {chunk_count}, size: {len(chunk_df)} rows")
            
            try:
                # Create cyclical features for date and time
                chunk_df = self._create_cyclical_features(chunk_df)
                
                # Process categorical features using pre-fitted encoders
                # 1. NB_SCATS_SITE encoding
                chunk_df['NB_SCATS_SITE_encoded'] = self.site_encoder.transform(chunk_df['NB_SCATS_SITE'])
                site_classes = len(self.site_encoder.classes_)
                
                # 2. scat_type encoding
                chunk_df['scat_type_encoded'] = self.scat_type_encoder.transform(chunk_df['scat_type'])
                scat_type_classes = len(self.scat_type_encoder.classes_)
                
                # 3. day_type encoding
                chunk_df['day_type_encoded'] = self.day_type_encoder.transform(chunk_df['day_type'])
                day_type_classes = len(self.day_type_encoder.classes_)
                
                # 4. Standardize numerical features using pre-fitted scalers
                chunk_df['school_count_scaled'] = self.school_count_scaler.transform(chunk_df[['school_count']])
                chunk_df['Flow_scaled'] = self.flow_scaler.transform(chunk_df[['Flow']])
                
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
                
                # Define categorical feature indices and metadata (same for all chunks)
                categorical_indices = {
                    'NB_SCATS_SITE': feature_cols.index('NB_SCATS_SITE_encoded'),
                    'day_type': feature_cols.index('day_type_encoded')
                }
                
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
                df_features = chunk_df[feature_cols].copy()
                
                # Get unique sites in this chunk and sort chronologically
                unique_sites = chunk_df['NB_SCATS_SITE'].unique()
                chunk_df['date_time'] = pd.to_datetime(chunk_df['date'] + ' ' + chunk_df['time'])
                
                # Create sequences for each site
                X_list = []
                y_list = []
                sites_list = []
                dates_list = []
                
                for site in unique_sites:
                    site_data = chunk_df[chunk_df['NB_SCATS_SITE'] == site].sort_values('date_time')
                    site_features = df_features[chunk_df['NB_SCATS_SITE'] == site].values
                    
                    # Create sequences only if we have enough data points
                    if len(site_features) >= seq_len + pred_len:
                        for i in range(0, len(site_features) - seq_len - pred_len + 1, step_size):
                            X_list.append(site_features[i:i+seq_len])
                            y_list.append(site_features[i+seq_len:i+seq_len+pred_len, -1])  # Only predict Flow
                            sites_list.append(site)
                            dates_list.append(site_data.iloc[i+seq_len]['date_time'])
                
                # Skip empty chunks
                if not X_list:
                    print(f"Chunk {chunk_count} generated no sequences, skipping")
                    continue
                
                # Convert to numpy arrays
                X = np.array(X_list)
                y = np.array(y_list)
                sites = np.array(sites_list)
                dates = np.array(dates_list)
                
                print(f"Chunk {chunk_count} created {len(X)} sequences: X shape={X.shape}, y shape={y.shape}")
                
                # Calculate additional metadata for the model
                input_dim = X.shape[2]  # Number of features
                output_dim = 1  # Predicting Flow only
                
                # Store the encoders and scalers for later use
                encoders_scalers = {
                    'site_encoder': self.site_encoder,
                    'scat_type_encoder': self.scat_type_encoder,
                    'day_type_encoder': self.day_type_encoder,
                    'school_count_scaler': self.school_count_scaler,
                    'flow_scaler': self.flow_scaler
                }
                
                # Store metadata for current chunk
                chunk_metadata = {
                    'sites': sites,
                    'dates': dates,
                    'feature_cols': feature_cols,
                    'chunk_id': chunk_count
                }
                
                # Update the metadata with current chunk info
                self.metadata.update(chunk_metadata)
                
                # Prepare data dictionary for this chunk
                chunk_data = {
                    'X': X,
                    'y': y,
                    'encoders_scalers': encoders_scalers,
                    'categorical_indices': categorical_indices,
                    'categorical_metadata': categorical_metadata,
                    'input_dim': input_dim,
                    'output_dim': output_dim,
                    'chunk_id': chunk_count
                }
                
                yield chunk_data
                
            except Exception as e:
                print(f"Error processing chunk {chunk_count}: {e}")
                continue
    
    def _initialize_encoders_on_full_dataset(self, input_file):
        """Initialize and fit encoders and scalers on the full dataset.
        
        This ensures consistent encoding across all chunks.
        
        Args:
            input_file: Path to the CSV file
        """
        
        # Get unique categorical values and statistics for continuous variables
        # Read only necessary columns to save memory
        columns_for_encoding = ['NB_SCATS_SITE', 'scat_type', 'day_type', 'school_count', 'Flow']
        
        # Use pandas to read specific columns
        categorical_data = pd.read_csv(input_file, usecols=columns_for_encoding)
        
        # Fit encoders on the entire dataset
        self.site_encoder.fit(categorical_data['NB_SCATS_SITE'])
        self.scat_type_encoder.fit(categorical_data['scat_type'])
        self.day_type_encoder.fit(categorical_data['day_type'])
        
        # Fit scalers on the entire dataset
        self.school_count_scaler.fit(categorical_data[['school_count']])
        self.flow_scaler.fit(categorical_data[['Flow']])
        
        print("Encoders and scalers fitted on the full dataset.")
        print(f"Number of unique sites: {len(self.site_encoder.classes_)}")
        print(f"Number of scat types: {len(self.scat_type_encoder.classes_)}")
        print(f"Number of day types: {len(self.day_type_encoder.classes_)}")
    
    def get_data_loaders_from_chunk(self, chunk_data, batch_size=32, shuffle=True, num_workers=0,
                                   train_ratio=0.7, val_ratio=0.3, random_state=42, device=None):
        """Create data loaders from a processed data chunk.
        
        Args:
            chunk_data: Dictionary with processed chunk data
            batch_size: Batch size for dataloaders
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for dataloaders
            train_ratio: Ratio of training data (default: 0.7)
            val_ratio: Ratio of validation data (default: 0.3)
            random_state: Random seed for reproducibility
            device: Device to move tensors to ('cuda' or 'cpu')
            
        Returns:
            Dictionary with data loaders and metadata
        """
        X = chunk_data['X']
        y = chunk_data['y']
        categorical_indices = chunk_data['categorical_indices']
        categorical_metadata = chunk_data['categorical_metadata']
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        # Get dataset size
        n_samples = len(X)
        
        # Create indices for train/val split
        indices = np.random.permutation(n_samples)
        train_size = int(train_ratio * n_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Extract train and validation sets
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Create datasets with device specification
        train_dataset = TrafficDataset(X_train, y_train, categorical_indices, device)
        val_dataset = TrafficDataset(X_val, y_val, categorical_indices, device)
        
        # Use pin_memory for efficient CPU to GPU transfers when device is not None and not CPU
        pin_memory = device is not None and 'cuda' in str(device)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'encoders_scalers': chunk_data['encoders_scalers'],
            'categorical_indices': categorical_indices,
            'categorical_metadata': categorical_metadata,
            'embedding_dim': self.embedding_dim,
            'input_dim': chunk_data.get('input_dim', X.shape[2]),
            'output_dim': chunk_data.get('output_dim', 1),
            'chunk_id': chunk_data.get('chunk_id', 0)
        }
    
    def get_data_loaders(self, data_file, batch_size=32, shuffle=True, num_workers=0, 
                         train_ratio=0.7, val_ratio=0.3, random_state=42, device=None):
        """Prepare data loaders for training, validation, and testing.
        
        Args:
            data_file: Path to the CSV data file
            batch_size: Batch size for dataloaders
            shuffle: Whether to shuffle the data
            num_workers: Number of workers for dataloaders
            train_ratio: Ratio of training data (default: 0.7)
            val_ratio: Ratio of validation data (default: 0.3) 
            random_state: Random seed for reproducibility
            device: Device to move tensors to ('cuda' or 'cpu')
            
        Returns:
            Dictionary with data loaders and metadata
        """
        # Process the data
        data = self.process_data(data_file)
        X, y = data['X'], data['y']
        categorical_indices = data['categorical_indices']
        categorical_metadata = data['categorical_metadata']
        
        # Set random seed for reproducibility
        np.random.seed(random_state)
        
        # Get dataset size
        n_samples = len(X)
        
        # Create indices for train/val split
        indices = np.random.permutation(n_samples)
        train_size = int(train_ratio * n_samples)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]
        
        # Extract train and validation sets
        X_train, y_train = X[train_indices], y[train_indices]
        X_val, y_val = X[val_indices], y[val_indices]
        
        # Create datasets with device specification
        train_dataset = TrafficDataset(X_train, y_train, categorical_indices, device)
        val_dataset = TrafficDataset(X_val, y_val, categorical_indices, device)
        
        # Use pin_memory for efficient CPU to GPU transfers when device is not None and not CPU
        pin_memory = device is not None and 'cuda' in str(device)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'encoders_scalers': data['encoders_scalers'],
            'categorical_indices': categorical_indices,
            'categorical_metadata': categorical_metadata,
            'embedding_dim': self.embedding_dim,
            'input_dim': data.get('input_dim', X.shape[2]),
            'output_dim': data.get('output_dim', 1)
        }
    
    def get_metadata(self):
        """Get metadata stored during data processing.
        
        Returns:
            Dictionary with metadata
        """
        return self.metadata
    
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
        
    def show_sample_data(self, data_loader):
        """Show a sample of data from the data loader.
        
        Args:
            data_loader: DataLoader object
            
        Returns:
            None
        """
        # Get the first batch
        X_batch, y_batch = next(iter(data_loader))
        
        # Convert tensors to numpy for easier handling
        X_np = X_batch.detach().cpu().numpy()
        y_np = y_batch.detach().cpu().numpy()
        
        # Get encoders and scalers
        encoders_scalers = data_loader.dataset.categorical_indices
        
        # Display only first 3 samples
        num_samples = min(3, len(X_np))
        
        print("\n===== Sample Data Visualization =====")
        
        for i in range(num_samples):
            print(f"\nSample {i+1}:")
            
            # Get feature data for the current sample
            features = X_np[i]
            target = y_np[i]
            
            # Get metadata for better interpretation
            if hasattr(self, 'metadata') and self.metadata:
                feature_cols = self.metadata.get('feature_cols', [])
            else:
                feature_cols = [f"Feature_{j}" for j in range(features.shape[1])]
            
            # Display sequence information
            print(f"  Sequence length: {features.shape[0]} time steps")
            print(f"  Features per time step: {features.shape[1]}")
            
            # Display processed features from the first time step
            print("\n  First time step (processed features):")
            for j, col in enumerate(feature_cols):
                if j < features.shape[1]:  # Make sure we don't go out of bounds
                    print(f"    {col}: {features[0, j]:.4f}")
            
            # Display last time step
            print("\n  Last time step (processed features):")
            for j, col in enumerate(feature_cols):
                if j < features.shape[1]:  # Make sure we don't go out of bounds
                    print(f"    {col}: {features[-1, j]:.4f}")
            
            # Display target values (processed)
            print("\n  Target sequence (processed):")
            for j in range(len(target)):
                print(f"    t+{j+1}: {target[j]:.4f}")
            
            # Display original flow values if possible
            if hasattr(self, 'flow_scaler') and self.flow_scaler is not None:
                try:
                    # Original flow value for the last input time step
                    last_flow_processed = features[-1, feature_cols.index('Flow_scaled')]
                    last_flow_original = self.inverse_transform_flow(
                        np.array([last_flow_processed]), 
                        {'flow_scaler': self.flow_scaler}
                    )[0]
                    
                    # Original flow values for targets
                    target_original = self.inverse_transform_flow(
                        target, 
                        {'flow_scaler': self.flow_scaler}
                    )
                    
                    print("\n  Original values (inverse transformed):")
                    print(f"    Last input flow: {last_flow_original:.2f} vehicles")
                    print("    Target flows:")
                    for j in range(len(target_original)):
                        print(f"      t+{j+1}: {target_original[j]:.2f} vehicles")
                        
                except (ValueError, IndexError) as e:
                    print(f"\n  Could not display original values: {e}")
            
            print("\n" + "-" * 40)

# Example usage
if __name__ == "__main__":
    # Create a data collector
    data_collector = TrafficDataCollector(embedding_dim=16)
    
    try:
        # Get data loaders directly from CSV file
        data_file = os.path.join(data_collector.data_path, '_sample_final_time_series.csv')
        data = data_collector.get_data_loaders(data_file, batch_size=32)
        
        # Print some information about the data
        train_loader = data['train_loader']
        val_loader = data['val_loader']
        categorical_metadata = data['categorical_metadata']
        
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        
        # Get a batch from the training loader
        X_batch, y_batch = next(iter(train_loader))
        print(f"Input batch shape: {X_batch.shape}")
        print(f"Target batch shape: {y_batch.shape}")
        
        # Print embedding information
        print("\nCategorical Feature Metadata:")
        for feature, metadata in categorical_metadata.items():
            print(f"  {feature}: {metadata['num_classes']} classes, embedding dim={metadata['embedding_dim']}")

        # Show a sample of data
        data_collector.show_sample_data(train_loader)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the CSV data file exists.")