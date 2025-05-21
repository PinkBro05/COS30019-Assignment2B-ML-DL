"""
Math operations for transformer model.
Contains attention and feed forward modules.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Define linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Define dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights for better training stability"""
        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """
        Compute scaled dot-product attention.
        
        Args:
            Q: Query matrix (batch_size, nhead, seq_len, d_k)
            K: Key matrix (batch_size, nhead, seq_len, d_k)
            V: Value matrix (batch_size, nhead, seq_len, d_k)
            mask: Attention mask (optional)
            
        Returns:
            Attention output and attention weights
        """
        # Get device from input tensors
        device = Q.device
        
        # Compute attention scores
        # Q: (batch_size, nhead, seq_len, d_k)
        # K: (batch_size, nhead, seq_len, d_k)
        # K.transpose: (batch_size, nhead, d_k, seq_len)
        # QK: (batch_size, nhead, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.d_k) + 1e-8)
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask is on the correct device
            mask = mask.to(device)
            
            # Handle different mask shapes
            if mask.dim() == 2:
                # Same mask for all heads: (seq_len, seq_len)
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), -1e9)
            elif mask.dim() == 3:
                # Different mask for each batch: (batch_size, seq_len, seq_len)
                scores = scores.masked_fill(mask.unsqueeze(1), -1e9)
                
        # IMPROVED: Apply numerical stability by subtracting max value before softmax
        scores_max, _ = scores.max(dim=-1, keepdim=True)
        scores = scores - scores_max
                
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        attn_weights = self.dropout(attn_weights)
        
        # Compute attention output
        # attn_weights: (batch_size, nhead, seq_len, seq_len)
        # V: (batch_size, nhead, seq_len, d_k)
        # output: (batch_size, nhead, seq_len, d_k)
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor (batch_size, seq_len, d_model)
            key: Key tensor (batch_size, seq_len, d_model)
            value: Value tensor (batch_size, seq_len, d_model)
            mask: Attention mask (optional)
            
        Returns:
            Tensor after attention (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)
        device = query.device
        
        # Linear projections and reshape for multi-head attention
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model) -> (batch_size, seq_len, nhead, d_k) -> (batch_size, nhead, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.nhead, self.d_k).transpose(1, 2)
        
        # Apply scaled dot-product attention
        output, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Reshape and apply final linear projection
        # (batch_size, nhead, seq_len, d_k) -> (batch_size, seq_len, nhead, d_k) -> (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        
        # Define feed-forward network (mini version of the Transformer)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """
        Initialize the weights for better training stability
        Using Kaiming (He) uniform initialization for linear layers.
        
        Source: https://paperswithcode.com/method/he-initialization
        """
        for m in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(m.bias, -bound, bound)
                
    def forward(self, x):
        """
        Forward pass for position-wise feed-forward network.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Tensor after feed-forward (batch_size, seq_len, d_model)
        """
        # Get device
        device = x.device
        
        # Apply first linear layer and ReLU activation
        x = F.relu(self.fc1(x))
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply second linear layer
        x = self.fc2(x)
        
        return x