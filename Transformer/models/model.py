"""
Transformer model definition for traffic flow prediction
"""
from torch import nn
import torch
from vanila_encoder import TransformerEncoderLayer

# Sinusoid positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, output_size=1, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        # Feature projection layer (instead of embedding)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Project input features to d_model dimensions
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        # Global average pooling across sequence dimension
        x = x.mean(dim=1)
        # Final projection to output dimension
        x = self.fc(self.dropout(x))
        return x
    
