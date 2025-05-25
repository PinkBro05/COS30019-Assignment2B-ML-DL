import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class DataProcessor:
    def __init__(self, data_path, sequence_length=10):
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None

    def load_data(self):
        df = pd.read_csv(self.data_path)
        
        # Convert Date to datetime
        df['Date'] = pd.to_datetime(df['Date'])
            
        if 'scat_type' in df.columns:
            scat_dummies = pd.get_dummies(df['scat_type'], prefix='scat')
            df = pd.concat([df, scat_dummies], axis=1)
        
        return df

    def create_sequences(self, data, sequence_length=None):
        """Convert to supervised learning format"""
        if sequence_length is None:
            sequence_length = self.sequence_length
        
        xs, ys = [], []
        for i in range(len(data) - sequence_length):
            x = data[i:(i + sequence_length)]
            y = data[i + sequence_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def scale_and_split_data(self, train_ratio=0.8, sequence_length=10):
        """
        Process SCATS traffic data, scale features, create sequences and split into train/test sets
        
        Args:
            train_ratio: Ratio of data to use for training (default 0.8)
            sequence_length: Length of input sequences for LSTM (default 10)
        
        Returns:
            X_train, X_test, y_train, y_test: Processed data ready for LSTM model
        """
        # Load the data
        df = self.load_data()
        
        # Handle Date column if needed (commented in basic_new_data.ipynb)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        
        # Select features based on available columns
        if 'Flow' in df.columns:
            # Use Flow column directly as in basic_new_data.ipynb
            feature_data = df[['Flow']].values
        else:
            # Use traffic volume columns if available (V00_0 to V95_0)
            feature_cols = [col for col in df.columns if col.startswith('V') and '_' in col]
            
            # Include categorical features if needed
            categorical_cols = [col for col in df.columns if col.startswith('day_') or col.startswith('scat_')]
            if categorical_cols:
                feature_cols.extend(categorical_cols)
                
            # If no specific features found, use all numeric columns
            if not feature_cols:
                feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
            feature_data = df[feature_cols].values
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Create sequences for LSTM
        X, y = self.create_sequences(scaled_data, self.sequence_length)
        
        # Split into train/test sets
        split_idx = int(train_ratio * len(X))
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        return X_train, X_test, y_train, y_test

    def inverse_transform(self, data):
        """
        Transform scaled data back to original scale
        
        Args:
            data: Scaled data to transform back
        
        Returns:
            Data in original scale
        """
        # Reshape if needed
        if len(data.shape) == 2 and data.shape[1] == 1:
            return self.scaler.inverse_transform(data)
        else:
            return self.scaler.inverse_transform(data.reshape(-1, 1))