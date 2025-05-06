"""
Decoder for the Transformer model.
"""
import torch
import torch.nn as nn

from Transformer.models.math import split_heads, scaled_dot_product_attention

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.num_heads = num_heads # Number of attention heads
        self.d_model = d_model # Input dimension for the model
        self.depth = d_model // num_heads # Depth of each attention head

        # Self Attention
        self.self_attn_wq = nn.Linear(d_model, d_model)
        self.self_attn_wk = nn.Linear(d_model, d_model)
        self.self_attn_wv = nn.Linear(d_model, d_model)
        self.self_attn_dense = nn.Linear(d_model, d_model)
        
        # Cross Attention (encoder-decoder attention)
        self.cross_attn_wq = nn.Linear(d_model, d_model)
        self.cross_attn_wk = nn.Linear(d_model, d_model)
        self.cross_attn_wv = nn.Linear(d_model, d_model)
        self.cross_attn_dense = nn.Linear(d_model, d_model)

        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization and dropout
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.layernorm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, look_ahead_mask=None, padding_mask=None):
        """ Decoder layer forward pass

        Args:
            x (input): input values (batch_size, seq_len, d_model)
            enc_output: output from encoder (batch_size, enc_seq_len, d_model)
            look_ahead_mask: mask for self-attention (causal attention)
            padding_mask: mask for encoder-decoder attention

        Returns:
            new x: x after attention and feed forward network
        """
        batch_size = x.size(0)
        
        # Self attention
        q = split_heads(self.num_heads, self.depth, self.self_attn_wq(x), batch_size)
        k = split_heads(self.num_heads, self.depth, self.self_attn_wk(x), batch_size)
        v = split_heads(self.num_heads, self.depth, self.self_attn_wv(x), batch_size)

        self_attn_output, _ = scaled_dot_product_attention(q, k, v, look_ahead_mask)
        
        self_attn_output = self_attn_output.transpose(1, 2).contiguous()
        self_attn_output = self_attn_output.view(batch_size, -1, self.d_model)
        self_attn_output = self.self_attn_dense(self_attn_output)
        
        # Add & Norm (first sublayer)
        attn1 = self.layernorm1(x + self.dropout(self_attn_output))

        # Cross attention (encoder-decoder attention)
        q = split_heads(self.num_heads, self.depth, self.cross_attn_wq(attn1), batch_size)
        k = split_heads(self.num_heads, self.depth, self.cross_attn_wk(enc_output), batch_size)
        v = split_heads(self.num_heads, self.depth, self.cross_attn_wv(enc_output), batch_size)

        cross_attn_output, _ = scaled_dot_product_attention(q, k, v, padding_mask)
        
        cross_attn_output = cross_attn_output.transpose(1, 2).contiguous()
        cross_attn_output = cross_attn_output.view(batch_size, -1, self.d_model)
        cross_attn_output = self.cross_attn_dense(cross_attn_output)
        
        # Add & Norm (second sublayer)
        attn2 = self.layernorm2(attn1 + self.dropout(cross_attn_output))
        
        # Feed Forward Network
        ffn_output = self.feed_forward(attn2)
        
        # Add & Norm (third sublayer)
        out = self.layernorm3(attn2 + self.dropout(ffn_output))
        
        return out
