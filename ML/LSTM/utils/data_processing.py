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
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # One-hot encode categorical columns
        if 'day_type' in df.columns:
            day_dummies = pd.get_dummies(df['day_type'], prefix='day')
            df = pd.concat([df, day_dummies], axis=1)
            
        if 'scat_type' in df.columns:
            scat_dummies = pd.get_dummies(df['scat_type'], prefix='scat')
            df = pd.concat([df, scat_dummies], axis=1)
        
        return df

    def create_sequences(self, data):
        """Convert to supervised learning format"""
        xs, ys = [], []
        for i in range(len(data) - self.sequence_length):
            x = data[i:(i + self.sequence_length)]
            y = data[i + self.sequence_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def scale_and_split_data(self):
        df = self.load_data()
        
        # Extract traffic volume columns (V00_0 to V95_0)
        feature_cols = [col for col in df.columns if col.startswith('V') and '_' in col]
        
        # Include categorical features (one-hot encoded columns)
        categorical_cols = [col for col in df.columns if col.startswith('day_') or col.startswith('scat_')]
        feature_cols.extend(categorical_cols)