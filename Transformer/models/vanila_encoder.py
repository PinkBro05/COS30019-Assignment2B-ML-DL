"""
Encoder layer of Transformer model
"""
import torch
import torch.nn as nn

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

    