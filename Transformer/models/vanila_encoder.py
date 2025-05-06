"""
Encoder layer of Transformer model
"""
import torch
import torch.nn as nn

from Transformer.models.math import split_heads, scaled_dot_product_attention

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

    def forward(self, x, mask=None):
        """ Encoder layer forward pass

        Args:
            x (input): input values (batch_size, seq_len, d_model)
            mask (boolean, optional): to control if causual attention or not. Defaults to None.

        Returns:
            new x: x after attention and feed forward network
        """
        batch_size = x.size(0)

        q = split_heads(self.num_heads, self.depth, self.wq(x), batch_size)
        k = split_heads(self.num_heads, self.depth, self.wk(x), batch_size)
        v = split_heads(self.num_heads, self.depth, self.wv(x), batch_size)

        scaled_attention, _ = scaled_dot_product_attention(q, k, v, mask) # use mask (causal attention)

        scaled_attention = scaled_attention.transpose(1, 2).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)

        attn_output = self.dense(concat_attention)

        x = self.layernorm1(x + self.dropout(attn_output))

        ff_output = self.feed_forward(x)

        x = self.layernorm2(x + self.dropout(ff_output))

        return x
