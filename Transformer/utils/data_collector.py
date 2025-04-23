import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class TimeSeriesTrafficFlow:
    """
    Class for handling the transformed time series traffic flow data.
    This class works with data in the format where each row represents a single 15-minute interval.
    
    The expected format is:
    Date SCATS Number Location CD_MELWAY HF VicRoads Internal VR Internal Stat Flow
    10/1/2006 00:00 0970 WARRIGAL_RD N of HIGH STREET_RD 060 G10 249 182 16
    10/1/2006 00:15 0970 WARRIGAL_RD N of HIGH STREET_RD 060 G10 249 182 24
    ...
    """
    def __init__(self, data_path):
        print(f"Loading time series traffic flow data from {data_path}...")
        self.data = pd.read_csv(data_path)
        print(f"Loaded data with shape: {self.data.shape}")
        print(f"Columns: {self.data.columns.tolist()}")
        
        # Process the Date column to extract date and time components
        if 'Date' in self.data.columns:
            self._process_date_column()
    
    def _process_date_column(self):
        """Process the Date column to extract date and time components"""
        print("Processing date and time information...")
        
        # Check if the Date column exists and contains datetime strings
        if 'Date' in self.data.columns:
            # Convert to datetime
            try:
                self.data['DateTime'] = pd.to_datetime(self.data['Date'])
                
                # Extract components
                self.data['Date_Only'] = self.data['DateTime'].dt.date
                self.data['Time_Only'] = self.data['DateTime'].dt.time
                self.data['Hour'] = self.data['DateTime'].dt.hour
                self.data['Minute'] = self.data['DateTime'].dt.minute
                self.data['DayOfWeek'] = self.data['DateTime'].dt.dayofweek
                self.data['Day'] = self.data['DateTime'].dt.day
                self.data['Month'] = self.data['DateTime'].dt.month
                self.data['Year'] = self.data['DateTime'].dt.year
                
                print("Successfully processed datetime information.")
            except Exception as e:
                print(f"Error processing datetime: {e}")
                print("Date column format may not be compatible.")
    
    def prepare_data_for_training(self, target_col='Flow', test_size=0.2, val_size=0.1, 
                                  random_state=42, scale_method='standard',
                                  sequence_length=4, prediction_horizon=1):
        """
        Prepare time series data for training.
        
        Args:
            target_col: Column name for the target variable (Flow)
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
            scale_method: Method to scale data ('standard', 'minmax', or None)
            sequence_length: Number of previous time steps to use as input
            prediction_horizon: Number of time steps ahead to predict
            
        Returns:
            Dictionary containing training, validation, and test data
        """
        print("Preparing time series data for training...")
        
        # Check if the target column exists
        if target_col not in self.data.columns:
            raise ValueError(f"Target column '{target_col}' not found in the data.")
        
        # Features to use for training, using DayofWeek since the traffic flow offten cycles weekly (common sense)
        feature_columns = [
            'SCATS Number', 'Location', 'Hour', 'Minute', 'DayOfWeek', 'Flow'
        ]
        
        # Filter to only include columns that exist
        feature_columns = [col for col in feature_columns if col in self.data.columns]
        
        # Ensure we have at least some features
        if not feature_columns:
            raise ValueError("No valid feature columns found in the data.")
        
        # Convert categorical features to numeric
        data_processed = self.data.copy()
        categorical_columns = ['SCATS Number', 'Location', 'CD_MELWAY']
        for col in categorical_columns:
            if col in data_processed.columns:
                data_processed[col] = pd.Categorical(data_processed[col]).codes
        
        # Create sequences of data
        sequences = self._create_sequences(
            data_processed, 
            feature_columns=feature_columns,
            target_col=target_col,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon
        )
        
        if len(sequences) == 0:
            raise ValueError("No valid sequences could be created from the data.")
        
        # Convert to numpy arrays
        X = np.array([seq['features'] for seq in sequences])
        y = np.array([seq['target'] for seq in sequences])
        
        print(f"Created {len(sequences)} sequences with shape: X={X.shape}, y={y.shape}")
        
        # Split data into train, validation, and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        # Scale features if requested
        if scale_method == 'standard':
            # Reshape to 2D for scaling
            shape_X_train = X_train.shape
            shape_X_val = X_val.shape
            shape_X_test = X_test.shape
            
            X_train_2d = X_train.reshape(-1, X_train.shape[-1])
            X_val_2d = X_val.reshape(-1, X_val.shape[-1])
            X_test_2d = X_test.reshape(-1, X_test.shape[-1])
            
            scaler_X = StandardScaler()
            X_train_2d = scaler_X.fit_transform(X_train_2d)
            X_val_2d = scaler_X.transform(X_val_2d)
            X_test_2d = scaler_X.transform(X_test_2d)
            
            # Reshape back
            X_train = X_train_2d.reshape(shape_X_train)
            X_val = X_val_2d.reshape(shape_X_val)
            X_test = X_test_2d.reshape(shape_X_test)
            
            # Scale targets
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
            y_val = scaler_y.transform(y_val.reshape(-1, 1)).reshape(-1)
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)
            
            scalers = {'X': scaler_X, 'y': scaler_y}
            
        elif scale_method == 'minmax':
            # Reshape to 2D for scaling
            shape_X_train = X_train.shape
            shape_X_val = X_val.shape
            shape_X_test = X_test.shape
            
            X_train_2d = X_train.reshape(-1, X_train.shape[-1])
            X_val_2d = X_val.reshape(-1, X_val.shape[-1])
            X_test_2d = X_test.reshape(-1, X_test.shape[-1])
            
            scaler_X = MinMaxScaler()
            X_train_2d = scaler_X.fit_transform(X_train_2d)
            X_val_2d = scaler_X.transform(X_val_2d)
            X_test_2d = scaler_X.transform(X_test_2d)
            
            # Reshape back
            X_train = X_train_2d.reshape(shape_X_train)
            X_val = X_val_2d.reshape(shape_X_val)
            X_test = X_test_2d.reshape(shape_X_test)
            
            # Scale targets
            scaler_y = MinMaxScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
            y_val = scaler_y.transform(y_val.reshape(-1, 1)).reshape(-1)
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).reshape(-1)
            
            scalers = {'X': scaler_X, 'y': scaler_y}
            
        else:
            scalers = None
        
        # Create data dictionary
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'feature_columns': feature_columns,
            'scalers': scalers
        }
        
        return data
    
    def _create_sequences(self, data, feature_columns, target_col, sequence_length=4, prediction_horizon=1):
        """
        Create sequences from time series data.
        
        Args:
            data: DataFrame with the data
            feature_columns: Columns to use as features
            target_col: Target column name
            sequence_length: Number of time steps in each sequence
            prediction_horizon: Steps ahead to predict
        
        Returns:
            List of sequence dictionaries with 'features' and 'target'
        """
        sequences = []
        
        # Group data by location and date to ensure we're working with continuous data
        if 'SCATS Number' in data.columns and 'Date_Only' in data.columns:
            groups = data.groupby(['SCATS Number', 'Date_Only'])
        elif 'SCATS Number' in data.columns:
            groups = data.groupby(['SCATS Number'])
        else:
            # If we don't have grouping columns, treat all data as one group
            groups = [(None, data)]
        
        for _, group_data in groups:
            # Ensure the group has enough data for a sequence
            if len(group_data) <= sequence_length + prediction_horizon:
                continue
            
            # Sort by date/time if available
            if 'DateTime' in group_data.columns:
                group_data = group_data.sort_values('DateTime')
            
            # Get features and targets
            features = group_data[feature_columns].values
            targets = group_data[target_col].values
            
            # Create sequences
            for i in range(len(group_data) - sequence_length - prediction_horizon + 1):
                feature_seq = features[i:i+sequence_length]
                target_value = targets[i+sequence_length+prediction_horizon-1]
                
                sequences.append({
                    'features': feature_seq,
                    'target': target_value
                })
        
        return sequences

class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data"""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_time_series_dataloaders(data_dict, batch_size=32):
    """Create PyTorch DataLoaders from the data dictionary"""
    train_dataset = TimeSeriesDataset(data_dict['X_train'], data_dict['y_train'])
    val_dataset = TimeSeriesDataset(data_dict['X_val'], data_dict['y_val'])
    test_dataset = TimeSeriesDataset(data_dict['X_test'], data_dict['y_test'])
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return {
        'train': train_dataloader,
        'val': val_dataloader,
        'test': test_dataloader
    }

def create_dataloaders(data, batch_size=32):
    """Create PyTorch DataLoaders from data for the original TrafficFlow class"""
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(data['X_train'], dtype=torch.float32)
    y_train_tensor = torch.tensor(data['y_train'], dtype=torch.float32)
    X_val_tensor = torch.tensor(data['X_val'], dtype=torch.float32)
    y_val_tensor = torch.tensor(data['y_val'], dtype=torch.float32)
    X_test_tensor = torch.tensor(data['X_test'], dtype=torch.float32)
    y_test_tensor = torch.tensor(data['y_test'], dtype=torch.float32)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def load_time_series_data():
    """
    Load the transformed time series data.
    
    Returns:
        TimeSeriesTrafficFlow object
    """
    # Try different paths to find the transformed data
    data_path = '../Data/Transformed/transformed_scats_data.csv'
    if not os.path.exists(data_path):
        data_path = 'Data/Transformed/transformed_scats_data.csv'
    
    # Check if the file exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Could not find transformed data at {data_path}")
    
    # Load the data
    traffic_flow = TimeSeriesTrafficFlow(data_path)
    return traffic_flow

def show_sample_data(data_dict, num_samples=5):
    """
    Print a sample of the data being fed to the model
    
    Args:
        data_dict: Data dictionary returned by prepare_data_for_training
        num_samples: Number of samples to print
    """
    print("\n===== SAMPLE DATA BEING FED TO THE MODEL =====")
    print(f"Feature columns: {data_dict['feature_columns']}")
    
    # Get the first few samples from the training set
    X_samples = data_dict['X_train'][:num_samples]
    y_samples = data_dict['y_train'][:num_samples]
    
    # Print raw sample values (possibly scaled)
    print("\nRaw sample input values (after scaling):")
    for i, (x, y) in enumerate(zip(X_samples, y_samples)):
        print(f"\nSample {i+1}:")
        
        # If we have sequence data (3D tensor)
        if len(x.shape) > 1:
            for j, time_step in enumerate(x):
                print(f"  Time step {j+1}: {time_step}")
        else:
            print(f"  Features: {x}")
        
        print(f"  Target: {y}")
    
    # If scalers are available, try to inverse transform to show original values
    if data_dict['scalers'] is not None:
        print("\nOriginal sample input values (before scaling):")
        
        try:
            # Get scalers
            scaler_X = data_dict['scalers']['X']
            scaler_y = data_dict['scalers']['y']
            
            # Inverse transform X samples
            if len(X_samples.shape) > 2:  # Sequence data (3D)
                # Reshape to 2D for inverse transform
                orig_shape = X_samples.shape
                X_samples_2d = X_samples.reshape(-1, X_samples.shape[-1])
                X_samples_original = scaler_X.inverse_transform(X_samples_2d)
                X_samples_original = X_samples_original.reshape(orig_shape)
            else:  # Non-sequence data (2D)
                X_samples_original = scaler_X.inverse_transform(X_samples)
            
            # Inverse transform y samples
            y_samples_original = scaler_y.inverse_transform(y_samples.reshape(-1, 1)).reshape(-1)
            
            # Print original values
            for i, (x, y) in enumerate(zip(X_samples_original, y_samples_original)):
                print(f"\nSample {i+1} (original values):")
                
                # If we have sequence data (3D tensor)
                if len(x.shape) > 1:
                    for j, time_step in enumerate(x):
                        print(f"  Time step {j+1}: {time_step}")
                else:
                    print(f"  Features: {x}")
                
                print(f"  Target: {y}")
                
        except Exception as e:
            print(f"Could not inverse transform scaled data: {e}")

def time_series_data_example():
    """Example usage of the TimeSeriesTrafficFlow class"""
    try:
        # Load the transformed time series data
        traffic_flow = load_time_series_data()
        
        # Prepare data for training
        data = traffic_flow.prepare_data_for_training(
            sequence_length=4,  # Use 4 time steps (1 hour) as input
            prediction_horizon=1,  # Predict 1 time step ahead (15 minutes)
            scale_method='standard'
        )
        
        # Create dataloaders
        dataloaders = create_time_series_dataloaders(data, batch_size=32)
        
        print("\n===== TIME SERIES DATA PREPARATION SUMMARY =====")
        print(f"Training data: X={data['X_train'].shape}, y={data['y_train'].shape}")
        print(f"Validation data: X={data['X_val'].shape}, y={data['y_val'].shape}")
        print(f"Test data: X={data['X_test'].shape}, y={data['y_test'].shape}")
        print(f"Feature columns used: {data['feature_columns']}")
        print(f"Number of training batches: {len(dataloaders['train'])}")
        
        # Show a sample of the data
        show_sample_data(data)
        
        return data, dataloaders
    
    except Exception as e:
        print(f"Error preparing time series data: {e}")
        return None, None

def main():
    # Example usage of the time series data preparation
    data, dataloaders = time_series_data_example()
        
if __name__ == "__main__":
    main()
