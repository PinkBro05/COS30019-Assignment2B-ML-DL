"""
Transformer model definition for traffic flow prediction
"""
from torch import nn
import torch
import math
from Transformer.models.vanila_encoder import TransformerEncoderLayer
from Transformer.models.vanila_decoder import TransformerDecoderLayer

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
        # Feature projection layers (instead of embedding)
        self.encoder_input_projection = nn.Linear(input_dim, d_model)
        self.decoder_input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final output layer
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def create_masks(self, src_seq_len, tgt_seq_len):
        # Create causal mask for decoder self-attention
        look_ahead_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).bool()
        # Create mask for encoder
        src_mask = torch.triu(torch.ones(src_seq_len, src_seq_len), diagonal=1).bool()
        
        return src_mask, look_ahead_mask

    def encode(self, src):
        """Encoder forward pass"""
        # Project input features to d_model dimensions
        src = self.encoder_input_projection(src)
        src = self.positional_encoding(src)
        
        # Get sequence length for masks
        src_seq_len = src.size(1)
        
        # Create causal mask for encoder (for time-series data)
        src_mask = torch.triu(torch.ones(src_seq_len, src_seq_len), diagonal=1).bool()
        src_mask = src_mask.to(src.device)
        
        # Pass through encoder layers
        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
            
        return enc_output

    def decode(self, tgt, enc_output):
        """Decoder forward pass"""
        # Project input features to d_model dimensions
        tgt = self.decoder_input_projection(tgt)
        tgt = self.positional_encoding(tgt)
        
        # Get sequence lengths for masks
        tgt_seq_len = tgt.size(1)
        
        # Create causal mask for decoder self-attention
        look_ahead_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len), diagonal=1).bool()
        look_ahead_mask = look_ahead_mask.to(tgt.device)
        
        # No need for padding mask in this implementation
        padding_mask = None
        
        # Pass through decoder layers
        dec_output = tgt
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, look_ahead_mask, padding_mask)
            
        # Global average pooling across sequence dimension
        dec_output = dec_output.mean(dim=1)
        
        # Final projection to output dimension
        output = self.fc(self.dropout(dec_output))
        
        return output

    def forward(self, src, tgt=None):
        """
        Forward pass of the transformer model
        
        Args:
            src: Source sequence (input data) - (batch_size, seq_len, input_dim)
            tgt: Target sequence for decoder - optional, if None, src will be used as tgt
            
        Returns:
            output: Predicted values - (batch_size, output_size)
        """
        # If no target is provided, use source as target
        if tgt is None:
            tgt = src
            
        # Pass through encoder
        enc_output = self.encode(src)
        
        # Pass through decoder and get final output
        output = self.decode(tgt, enc_output)
        
        return output
