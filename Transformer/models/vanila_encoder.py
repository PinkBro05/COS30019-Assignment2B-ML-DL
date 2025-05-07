"""
Encoder part of the transformer model.
Contains attention and feed forward modules.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from Transformer.models.math import MultiHeadAttention, PositionWiseFeedForward

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        # Define multi-head attention
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        
        # Define feed-forward network (mini version of the Transformer)
        self.feed_forward = PositionWiseFeedForward(d_model, dim_feedforward, dropout=dropout)
        
        # Define normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Define dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass for the encoder layer.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            mask: Attention mask (optional)
            
        Returns:
            Tensor after attention and feed-forward (batch_size, seq_len, d_model)
        """
        device = x.device
        
        # Self attention (Pre-LayerNorm variant)
        residual = x
        x = self.norm1(x)
        
        # Move mask to the same device as x if it exists
        if mask is not None:
            mask = mask.to(device)
            
        x = self.self_attn(x, x, x, mask)
        x = self.dropout(x)
        x = residual + x
        
        # Feed forward (Pre-LayerNorm variant)
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x
