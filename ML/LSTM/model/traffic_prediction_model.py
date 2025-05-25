import torch
import torch.nn as nn
from .lstm import LSTM

class TrafficPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1):
        super(TrafficPredictionModel, self).__init__()
        # First LSTM layer (128 units with return_sequences=True)
        self.lstm1 = LSTM(input_size, hidden_size, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(0.3)
        
        # Second LSTM layer (64 units with return_sequences=False)
        self.lstm2 = LSTM(hidden_size, 64, batch_first=True)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        # Dense layers matching notebook architecture
        self.fc1 = nn.Linear(64, 64)
        self.dropout3 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First LSTM layer (return sequences)
        out, (h, c) = self.lstm1(x)
        # No need to extract last timestep as we want all sequences
        batch_size, seq_len, hidden_size = out.size()
        
        # Second LSTM layer (no return sequences)
        out, (h, c) = self.lstm2(out)
        # Extract last timestep
        out = out[:, -1, :]
        out = self.bn2(out)
        out = self.dropout2(out)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout3(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        
        return out