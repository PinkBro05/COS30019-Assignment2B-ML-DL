import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class TrafficDataset(Dataset):
    """Traffic Flow Dataset for Transformer model."""
    
    def __init__(self, X, y):
        """Initialize the dataset.
        
        Args:
            X: Input sequences (shape: [n_samples, seq_len, n_features])
            y: Target sequences (shape: [n_samples, pred_len, n_features])
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TrafficDataCollector:
    """Collects and prepares traffic flow data for Transformer model."""
    
    def __init__(self, data_path=None):
        """Initialize the data collector.
        
        Args:
            data_path: Path to the transformed data directory
        """
        if data_path is None:
            # Default path relative to the Transformer directory
            self.data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                                         'Data', 'Transformed')
        else:
            self.data_path = data_path
            
    def load_data(self, filename='transformer_data.pkl'):
        """Load the prepared data for the Transformer model.
        
        Args:
            filename: Name of the file containing the prepared data
            
        Returns:
            Dictionary with X, y, sites, dates arrays
        """
        file_path = os.path.join(self.data_path, filename)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}. Run the data transformation script first.")
        
        # Load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        return data
    
    def get_data_loaders(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, 
                         batch_size=32, shuffle=True, num_workers=0, 
                         random_state=42, filename='transformer_data.pkl'):
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
            Dictionary with train_loader, val_loader, test_loader, and scaler
        """
        # Load the data
        data = self.load_data(filename)
        X, y = data['X'], data['y']
        
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
        
        # Normalize the data (fit on training data only)
        X_mean = X_train.mean(axis=(0, 1))
        X_std = X_train.std(axis=(0, 1))
        y_mean = y_train.mean(axis=(0, 1))
        y_std = y_train.std(axis=(0, 1))
        
        # Avoid division by zero
        X_std = np.where(X_std == 0, 1, X_std)
        y_std = np.where(y_std == 0, 1, y_std)
        
        # Normalize the data
        X_train = (X_train - X_mean) / X_std
        X_val = (X_val - X_mean) / X_std
        X_test = (X_test - X_mean) / X_std
        
        y_train = (y_train - y_mean) / y_std
        y_val = (y_val - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std
        
        # Create datasets
        train_dataset = TrafficDataset(X_train, y_train)
        val_dataset = TrafficDataset(X_val, y_val)
        test_dataset = TrafficDataset(X_test, y_test)
        
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
        
        # Create scaler for inverse transformation
        scaler = {
            'X_mean': X_mean,
            'X_std': X_std,
            'y_mean': y_mean,
            'y_std': y_std
        }
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'scaler': scaler
        }
    
    def inverse_transform(self, normalized_data, scaler, is_target=True):
        """Inverse transform normalized data to original scale.
        
        Args:
            normalized_data: Normalized data (torch tensor or numpy array)
            scaler: Dictionary with mean and std values
            is_target: Whether the data is target (y) or input (X)
            
        Returns:
            Data in original scale
        """
        # Convert to numpy if it's a torch tensor
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.detach().cpu().numpy()
        
        # Get appropriate scaler values
        if is_target:
            mean = scaler['y_mean']
            std = scaler['y_std']
        else:
            mean = scaler['X_mean']
            std = scaler['X_std']
        
        # Inverse transform
        return normalized_data * std + mean

# Example usage
if __name__ == "__main__":
    # Create a data collector
    data_collector = TrafficDataCollector()
    
    try:
        # Get data loaders
        data = data_collector.get_data_loaders(batch_size=32)
        
        # Print some information about the data
        train_loader = data['train_loader']
        val_loader = data['val_loader']
        test_loader = data['test_loader']
        
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")
        print(f"Number of testing batches: {len(test_loader)}")
        
        # Get a batch from the training loader
        X_batch, y_batch = next(iter(train_loader))
        print(f"Input batch shape: {X_batch.shape}")
        print(f"Target batch shape: {y_batch.shape}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the transform_traffic_data.py script first to prepare the data.")