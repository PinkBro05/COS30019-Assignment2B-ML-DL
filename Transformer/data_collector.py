import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

class TrafficFlow():
    def __init__(self, data_path):
        # Read CSV without using the first column as index and skip the first row
        # The first row contains time information corresponding to Vxx columns
        self.data = pd.read_csv(data_path, skiprows=1)
        print("Columns in CSV file:", self.data.columns.tolist())
        print(f"Data shape after skipping first row: {self.data.shape}")
        # Create time mapping dictionary
        self.time_mapping = self._create_time_mapping()
    
    def _create_time_mapping(self):
        """Create a mapping between V-columns and their corresponding times"""
        time_mapping = {}
        for i in range(96):  # 96 time slots in a day (24 hours * 4 quarters)
            column_name = f'V{i:02d}'  # Format as V00, V01, V02, etc.
            hour = i // 4  # Integer division to get the hour
            minute = (i % 4) * 15  # Get minutes (0, 15, 30, 45)
            time_str = f'{hour}:{minute:02d}'  # Format as 0:00, 0:15, etc.
            time_mapping[column_name] = time_str
        return time_mapping
    
    def get_time_for_column(self, input_value):
        """
        Bidirectional mapping between V-columns and times.
        If input is a V-column (e.g., 'V00'), returns the corresponding time (e.g., '0:00').
        If input is a time (e.g., '00:12'), returns the corresponding V-column (e.g., 'V00').
        """
        # Check if input is a V-column
        if input_value in self.time_mapping:
            return self.time_mapping[input_value]
        
        # Check if input is a time string
        try:
            if ':' in input_value:
                # Parse the time
                hours, minutes = map(int, input_value.split(':'))
                
                # Convert to minutes since midnight
                total_minutes = hours * 60 + minutes
                
                # Find the corresponding 15-minute interval
                interval_index = total_minutes // 15
                
                # Return the V-column
                return f'V{interval_index:02d}'
        except:
            pass
            
        return None
    
    def display_time_mapping(self):
        """Display the mapping of V-columns to times"""
        for column, time in self.time_mapping.items():
            print(f"{column} = {time}")

    def transform_data_for_time_series(self, save_path=None):
        """
        Transform the SCATS data into time series format with one row per 15-minute interval.
        
        The resulting format will have:
        Date SCATS Number Location CD_MELWAY HF VicRoads Internal VR Internal Stat Flow
        10/1/2006 00:00 0970 WARRIGAL_RD N of HIGH STREET_RD 060 G10 249 182 16
        10/1/2006 00:15 0970 WARRIGAL_RD N of HIGH STREET_RD 060 G10 249 182 24
        ...
        
        This format has one row per location per time interval, making it suitable
        for time series analysis.
        """
        print("Transforming SCATS data into time series format...")
        
        # Get the time series columns (V00-V95)
        time_columns = [f'V{i:02d}' for i in range(96)]
        
        # Get the metadata columns (everything except time series values)
        metadata_columns = [col for col in self.data.columns if col not in time_columns]
        
        # Create a new dataframe for the transformed data
        transformed_rows = []
        
        # For each row in the original data
        for _, row in self.data.iterrows():
            location_metadata = {col: row[col] for col in metadata_columns}
            date = location_metadata.get('Date', '')
            
            # For each time interval
            for time_col in time_columns:
                if time_col in self.data.columns:
                    time_str = self.time_mapping[time_col]
                    flow_value = row[time_col]
                    
                    # Create a new row with date+time, metadata, and flow value
                    new_row = {
                        'Date': f"{date} {time_str}",
                        'SCATS Number': location_metadata.get('SCATS Number', ''),
                        'Location': location_metadata.get('Location', ''),
                        'CD_MELWAY': location_metadata.get('CD_MELWAY', ''),
                        'HF VicRoads Internal': location_metadata.get('HF VicRoads Internal', ''),
                        'VR Internal Stat': location_metadata.get('VR Internal Stat', ''),
                        'Flow': flow_value
                    }
                    transformed_rows.append(new_row)
        
        # Convert to DataFrame
        columns_order = ['Date', 'SCATS Number', 'Location', 'CD_MELWAY', 
                         'HF VicRoads Internal', 'VR Internal Stat', 'Flow']
        transformed_df = pd.DataFrame(transformed_rows)
        transformed_df = transformed_df[columns_order]
        
        # Save the transformed data
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            transformed_df.to_csv(save_path, index=False)
            print(f"Transformed time series data saved to {save_path}")
            print(f"Transformed data shape: {transformed_df.shape}")
            print(f"Sample of transformed data:\n{transformed_df.head(10)}")
        
        return transformed_df

    def transform_data_for_transformer(self, save_path=None):
        """
        Transform the SCATS data for use with a Transformer model.
        - Use metadata columns as features
        - Use time series values (V00-V95) as targets
        """
        print("Transforming SCATS data for Transformer model...")

        # Get the time series columns (V00-V95)
        time_columns = [f'V{i:02d}' for i in range(96)]
        
        # Print available columns for debugging
        print("Available columns:", self.data.columns.tolist())
        
        # Create a new dataframe with the features and all time series data
        transformed_data = []
        
        for _, row in self.data.iterrows():
            # Create metadata dict with available columns
            metadata = {}
            for col in self.data.columns:
                if col not in time_columns:  # If it's not a time series column
                    metadata[col] = row[col]
            
            # Get all time series values for this row
            time_series_values = []
            for col in time_columns:
                if col in self.data.columns:
                    time_series_values.append(row[col])
                else:
                    time_series_values.append(np.nan)
            
            time_series_values = np.array(time_series_values)
            
            # For CSV format, we'll expand the time series values into individual columns
            sample = {**metadata}
            for i, val in enumerate(time_series_values):
                sample[f'V{i:02d}'] = val
            
            transformed_data.append(sample)
        
        # Convert to DataFrame
        transformed_df = pd.DataFrame(transformed_data)
        
        # Save the transformed data
        if save_path:
            # If the save_path has .pkl extension, change it to .csv
            if save_path.endswith('.pkl'):
                save_path = save_path.replace('.pkl', '.csv')
            elif not save_path.endswith('.csv'):
                save_path = f"{save_path}.csv"
                
            transformed_df.to_csv(save_path, index=False)
            print(f"Transformed data saved to {save_path}")
        
        # Create a DataFrame with the TimeSeries as a separate column for internal use
        internal_df = pd.DataFrame(transformed_data)
        internal_df['TimeSeries'] = [np.array([row[f'V{i:02d}'] for i in range(96)]) for _, row in transformed_df.iterrows()]
        
        return internal_df

    def prepare_data_for_training(self, test_size=0.2, val_size=0.1, random_state=42, scale_method='standard'):
        """
        Prepare data for training a transformer model.
        
        Args:
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
            scale_method: Method to scale data ('standard', 'minmax', or None)
            
        Returns:
            Dictionary containing training, validation, and test data
        """
        # Get the transformed data
        transformed_df = self.transform_data_for_transformer()
        
        # Print available columns for debugging
        print("Available columns in transformed data:", transformed_df.columns.tolist())
        
        # Use only available columns for features
        all_feature_columns = [
            'SCATS Number', 'Location', 'CD_MELWAY', 
            'NB_LATITUDE', 'NB_LONGITUDE', 
            'HF VicRoads Internal', 'VR Internal Stat', 'VR Internal Loc', 
            'NB_TYPE_SURVEY', 'Date'
        ]
        
        # Filter to only include columns that exist in the dataframe
        feature_columns = [col for col in all_feature_columns if col in transformed_df.columns]
        print("Using feature columns:", feature_columns)
        
        # Extract features and targets
        X = transformed_df[feature_columns].copy()
        y = np.stack(transformed_df['TimeSeries'].values)
        
        # Convert categorical features to numeric
        categorical_columns = ['SCATS Number', 'Location', 'CD_MELWAY']
        for col in categorical_columns:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes
        
        # For NB_TYPE_SURVEY, we'll keep it as is if it's already numeric
        if 'NB_TYPE_SURVEY' in X.columns and not pd.api.types.is_numeric_dtype(X['NB_TYPE_SURVEY']):
            X['NB_TYPE_SURVEY'] = pd.Categorical(X['NB_TYPE_SURVEY']).codes
        
        # Handle date column if present
        if 'Date' in X.columns:
            X['Date'] = pd.to_datetime(X['Date'])
            X['Day'] = X['Date'].dt.day
            X['Month'] = X['Date'].dt.month
            X['Year'] = X['Date'].dt.year
            X['DayOfWeek'] = X['Date'].dt.dayofweek
            X = X.drop('Date', axis=1)
        
        # Fill any missing values in features
        X = X.fillna(X.median())
        
        # Split data into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Split train into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size/(1-test_size), random_state=random_state
        )
        
        # Scale features if requested
        if scale_method == 'standard':
            scaler_X = StandardScaler()
            X_train = scaler_X.fit_transform(X_train)
            X_val = scaler_X.transform(X_val)
            X_test = scaler_X.transform(X_test)
            
            # Scale time series data (along each time point)
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train)
            y_val = scaler_y.transform(y_val)
            y_test = scaler_y.transform(y_test)
            
            scalers = {'X': scaler_X, 'y': scaler_y}
            
        elif scale_method == 'minmax':
            scaler_X = MinMaxScaler()
            X_train = scaler_X.fit_transform(X_train)
            X_val = scaler_X.transform(X_val)
            X_test = scaler_X.transform(X_test)
            
            # Scale time series data (along each time point)
            scaler_y = MinMaxScaler()
            y_train = scaler_y.fit_transform(y_train)
            y_val = scaler_y.transform(y_val)
            y_test = scaler_y.transform(y_test)
            
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
            'scalers': scalers
        }
        
        return data

class TrafficTimeSeriesDataset(Dataset):
    """Dataset class for loading and processing transformed SCATS traffic data"""
    
    def __init__(self, csv_path, feature_cols=None, time_cols=None, seq_length=96):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to the transformed_scats_data.csv file
            feature_cols: List of columns to use as features (default: SCATS Number, NB_TYPE_SURVEY)
            time_cols: List of columns to use as time series values (default: V00-V95)
            seq_length: Length of the time series sequence
        """
        self.data = pd.read_csv(csv_path)
        self.seq_length = seq_length
        
        # Set default feature columns if not provided
        if feature_cols is None:
            self.feature_cols = ['SCATS Number', 'NB_TYPE_SURVEY']
        else:
            self.feature_cols = feature_cols
            
        # Set default time series columns if not provided
        if time_cols is None:
            self.time_cols = [f'V{i:02d}' for i in range(seq_length)]
        else:
            self.time_cols = time_cols
            
        # Verify that the required columns exist
        missing_features = [col for col in self.feature_cols if col not in self.data.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns: {missing_features}")
            
        missing_time_cols = [col for col in self.time_cols if col not in self.data.columns]
        if missing_time_cols:
            raise ValueError(f"Missing time series columns: {missing_time_cols}")
        
        # Process features (convert categorical to numerical)
        self.processed_features = self._process_features()
        
        # Process time series data
        self.time_series = self._process_time_series()
        
    def _process_features(self):
        """Convert categorical features to numerical values"""
        features_df = self.data[self.feature_cols].copy()
        
        # Convert categorical features to numerical codes
        for col in self.feature_cols:
            if pd.api.types.is_object_dtype(features_df[col]):
                features_df[col] = pd.Categorical(features_df[col]).codes
                
        # Fill missing values
        features_df = features_df.fillna(features_df.median())
        
        return features_df.values
    
    def _process_time_series(self):
        """Extract and process time series data"""
        time_series_df = self.data[self.time_cols]
        
        # Convert to float and handle missing values
        time_series_data = time_series_df.values.astype(np.float32)
        
        # Replace NaN with 0 or mean
        if np.any(np.isnan(time_series_data)):
            col_means = np.nanmean(time_series_data, axis=0)
            for i, col_mean in enumerate(col_means):
                mask = np.isnan(time_series_data[:, i])
                time_series_data[mask, i] = col_mean
        
        return time_series_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        features = torch.tensor(self.processed_features[idx], dtype=torch.float32)
        time_series = torch.tensor(self.time_series[idx], dtype=torch.float32)
        
        return features, time_series
                
def main():
    # Path to the data file
    data_path = '../Data/Raw/Scat_Data.csv'
    
    # Check if the file exists
    if not os.path.exists(data_path):
        # Try another path
        data_path = 'Data/Raw/Scat_Data.csv'
        if not os.path.exists(data_path):
            print(f"Error: Could not find the data file at {data_path}")
            return
    
    print(f"Reading data from: {data_path}")
    
    # Create TrafficFlow object
    traffic_flow = TrafficFlow(data_path)
    
    # Transform data for transformer model and save it
    transformed_data = traffic_flow.transform_data_for_transformer(save_path='Data/Transformed/transformed_scats_data.csv')
    print(f"Transformed data saved with shape: {transformed_data.shape}")
    
    # Print information about the transformed data
    print("\n===== TRANSFORMED DATA SUMMARY =====")
    print(f"Number of samples: {len(transformed_data)}")
    print(f"All columns: {transformed_data.columns.tolist()}")
    print(f"Metadata columns: {[col for col in transformed_data.columns if col != 'TimeSeries']}")
    
    # Example shape of time series data
    if 'TimeSeries' in transformed_data.columns and len(transformed_data) > 0:
        first_time_series = transformed_data['TimeSeries'].iloc[0]
        print(f"Time series shape: {first_time_series.shape} (96 15-minute intervals)")
    
    # Example: Prepare data for training
    prepared_data = traffic_flow.prepare_data_for_training()
    print("\n===== PREPARED DATA FOR TRAINING =====")
    print(f"Training data: {prepared_data['X_train'].shape}, {prepared_data['y_train'].shape}")
    print(f"Validation data: {prepared_data['X_val'].shape}, {prepared_data['y_val'].shape}")
    print(f"Test data: {prepared_data['X_test'].shape}, {prepared_data['y_test'].shape}")
    
    # Example: Create dataloaders
    dataloaders = create_dataloaders(prepared_data, batch_size=32)
    print(f"Number of training batches: {len(dataloaders['train'])}")
    print(f"Number of validation batches: {len(dataloaders['val'])}")
    print(f"Number of test batches: {len(dataloaders['test'])}")
    
    print("\nData preparation complete! You can now use the transformed data for your Transformer model.")

if __name__ == "__main__":
    main()