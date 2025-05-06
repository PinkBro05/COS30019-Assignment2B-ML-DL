"""
    Math and mechanics functions for the Transformer model
"""
import torch
import torch.nn as nn


def split_heads(num_heads, depth, x, batch_size):
    """ Split the input into multiple heads.
    Args:
        x (seq_len_q, d_model): input for each query, key, value
        batch_size (int): batch size

    Returns:
        x_splited (batch_size, num_heads, seq_len_q, depth): x split into multiple heads
    """
    x = x.view(batch_size, -1, num_heads, depth)
    return x.transpose(1, 2)

def scaled_dot_product_attention(q, k, v, mask=None):
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
        # Apply the mask to the scaled attention logits

        scaled_attention_logits = scaled_attention_logits.masked_fill(mask, -1e9)

    attention_weights = torch.nn.functional.softmax(scaled_attention_logits, dim=-1)
    output = torch.matmul(attention_weights, v) # softmax(QK/sqrt(dk))V

    return output, attention_weights