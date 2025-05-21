import torch
import torch.nn as nn
from .lstm import LSTM

class TrafficPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TrafficPredictionModel, self).__init__()
        self.lstm1 = LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.lstm2 = LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        
        self.lstm3 = LSTM(hidden_size, hidden_size//2, batch_first=True)
        self.dropout3 = nn.Dropout(0.3)
        self.bn3 = nn.BatchNorm1d(hidden_size//2)
        
        self.fc1 = nn.Linear(hidden_size//2, 128)
        self.dropout4 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.dropout5 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, output_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # First LSTM layer
        out, (h, c) = self.lstm1(x)
        batch_size, seq_len, hidden_size = out.size()
        out = out[:, -1, :]  # Take the last sequence output
        out = self.dropout1(out)
        out = self.bn1(out)
        
        # Second LSTM layer - reshape for sequence input
        out = out.unsqueeze(1).repeat(1, seq_len, 1)
        out, (h, c) = self.lstm2(out)
        out = out[:, -1, :]
        out = self.dropout2(out)
        out = self.bn2(out)
        
        # Third LSTM layer - reshape for sequence input
        out = out.unsqueeze(1).repeat(1, seq_len, 1)
        out, (h, c) = self.lstm3(out)
        out = out[:, -1, :]
        out = self.dropout3(out)
        out = self.bn3(out)
        
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout4(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout5(out)
        out = self.fc3(out)
        
        return out
