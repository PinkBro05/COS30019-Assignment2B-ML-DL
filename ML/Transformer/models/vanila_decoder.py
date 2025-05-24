"""
Decoder part of the transformer model.
Contains self-attention, encoder-decoder attention, and feed forward modules.
"""
import torch.nn as nn
from ML.Transformer.models.math import MultiHeadAttention, PositionWiseFeedForward

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        # Define multi-head attention modules
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.encoder_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        
        # Define feed-forward network (mini version of the Transformer)
        self.feed_forward = PositionWiseFeedForward(d_model, dim_feedforward, dropout=dropout)
        
        # Define normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Define dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, memory, look_ahead_mask=None, memory_mask=None):
        """
        Forward pass for the decoder layer.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            memory: Output from the encoder (batch_size, src_seq_len, d_model)
            look_ahead_mask: Mask for self-attention (optional)
            memory_mask: Mask for encoder-decoder attention (optional)
            
        Returns:
            Tensor after attention and feed-forward (batch_size, seq_len, d_model)
        """
        device = x.device
        
        # Self attention (Pre-LayerNorm variant)
        residual = x
        x = self.norm1(x)
        
        # Ensure masks are on the same device
        if look_ahead_mask is not None:
            look_ahead_mask = look_ahead_mask.to(device)
            
        x = self.self_attn(x, x, x, look_ahead_mask)
        x = self.dropout(x)
        x = residual + x
        
        # Encoder-decoder attention (Pre-LayerNorm variant)
        residual = x
        x = self.norm2(x)
        
        # Ensure memory and masks are on the same device
        memory = memory.to(device)
        if memory_mask is not None:
            memory_mask = memory_mask.to(device)
            
        x = self.encoder_attn(x, memory, memory, memory_mask)
        x = self.dropout(x)
        x = residual + x
        
        # Feed forward (Pre-LayerNorm variant)
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x
