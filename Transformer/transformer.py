import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from data_collector import TrafficTimeSeriesDataset

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.num_heads = num_heads # Number of attention heads
        self.d_model = d_model # Input dimension for the model
        self.depth = d_model // num_heads # Depth of each attention head

        # Linear layers for Q, K, V matrices
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        # Output linear transformation
        self.dense = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization and dropout
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x, batch_size):
        """ Split the input into multiple heads.
        Args:
            x (seq_len_q, d_model): input for each query, key, value
            batch_size (int): batch size

        Returns:
            x_splited (batch_size, num_heads, seq_len_q, depth): x split into multiple heads
        """
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """ Calculate the attention weights, and new query

        Args:
            q (batch_size, num_heads, seq_len_q, depth): query with heads
            k (_type_batch_size, num_heads, seq_len_q, depth): key with heads
            v (batch_size, num_heads, seq_len_q, depth): value with heads
            mask (boolean, optional): to control if causual attention or not. Defaults to None.

        Returns:
            ouput (batch_size, num_heads, seq_len_q, depth): new query with heads
            attention_weights (batch_size, num_heads, seq_len_q, seq_len_k): attention weights
        """
        
        matmul_qk = torch.matmul(q, k.transpose(-2, -1)) 
        
        dk = torch.tensor(k.size(-1), dtype=torch.float32) # scalar
        
        scaled_attention_logits = matmul_qk / torch.sqrt(dk) # QK/sqrt(dk)

        if mask is not None:
            scaled_attention_logits = scaled_attention_logits.masked_fill(mask == 0, -1e9)

        attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
        output = torch.matmul(attention_weights, v) # softmax(QK/sqrt(dk))V

        return output, attention_weights

    def forward(self, x, mask=None):
        """ Encoder layer forward pass

        Args:
            x (input): input values (batch_size, seq_len, d_model)
            mask (boolean, optional): to control if causual attention or not. Defaults to None.

        Returns:
            new x: x after attention and feed forward network
        """
        batch_size = x.size(0)

        q = self.split_heads(self.wq(x), batch_size)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)

        scaled_attention, _ = self.scaled_dot_product_attention(q, k, v, mask) # use mask (causal attention)

        scaled_attention = scaled_attention.transpose(1, 2).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        attn_output = self.dense(concat_attention)

        x = self.layernorm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)

        x = self.layernorm2(x + self.dropout(ff_output))

        return x

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
    def __init__(self, vocab_size, d_model, num_heads, d_ff, output_size, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)  # Ensure embed_size matches d_model
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        x = x.mean(dim=1)
        x = self.fc(self.dropout(x))
        return x
    
def main():
    # Create dataset
    dataset = TrafficTimeSeriesDataset(
        csv_path="Data/Transformed/transformed_scats_data.csv",
        feature_cols=['SCATS Number', 'Location', 'Date']  # Customize features as needed
    )

    print (dataset[0])  # Check the first data point
    print (dataset[0][1])  # Check the features of the first data point
    # Create dataloaders
    batch_size = 32
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

if __name__ == "__main__":
    main()