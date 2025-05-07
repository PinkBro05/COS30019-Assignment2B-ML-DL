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
    def __init__(self, input_dim, d_model, num_heads, d_ff, output_size=1, num_layers=2, dropout=0.1,
                 categorical_metadata=None, categorical_indices=None):
        super(TransformerModel, self).__init__()
        
        self.categorical_metadata = categorical_metadata
        self.categorical_indices = categorical_indices
        
        # Create embedding layers for categorical features
        self.embedding_layers = nn.ModuleDict()
        self.embedding_dims = {}
        
        # Initialize embedding layers if categorical metadata is provided
        if categorical_metadata and categorical_indices:
            for feature_name, metadata in categorical_metadata.items():
                num_classes = metadata['num_classes']
                embedding_dim = metadata['embedding_dim']
                self.embedding_layers[feature_name] = nn.Embedding(num_classes, embedding_dim)
                self.embedding_dims[feature_name] = embedding_dim
        
        # Calculate the adjusted input dimension (after replacing categorical indices with embeddings)
        self.input_dim = input_dim
        if categorical_metadata and categorical_indices:
            # Remove original categorical feature dimensions and add embedding dimensions
            for feature_name, metadata in categorical_metadata.items():
                # Subtract 1 for the categorical index that will be replaced
                self.input_dim = self.input_dim - 1
                # Add the embedding dimension
                self.input_dim = self.input_dim + metadata['embedding_dim']
        
        # Feature projection layers
        self.encoder_input_projection = nn.Linear(self.input_dim, d_model)
        self.decoder_input_projection = nn.Linear(self.input_dim, d_model)
        
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

    def _apply_embeddings(self, x):
        """
        Apply embeddings to categorical features in the input tensor
        
        Args:
            x: Input tensor with categorical indices [batch_size, seq_len, input_dim]
            
        Returns:
            Tensor with categorical indices replaced by embeddings
        """
        if not (self.categorical_metadata and self.categorical_indices):
            return x
        
        batch_size, seq_len, _ = x.shape
        device = x.device  # Get the device of the input tensor
        
        # Create a list to store processed features
        features_list = []
        
        # Process each feature in the input
        for i in range(x.shape[2]):
            if i in [self.categorical_indices[name] for name in self.categorical_indices]:
                # Get feature name based on index
                feature_name = None
                for name, idx in self.categorical_indices.items():
                    if idx == i:
                        feature_name = name
                        break
                
                if feature_name:
                    # Extract categorical indices and ensure they're on the correct device
                    cat_indices = x[:, :, i].long().to(device)
                    
                    # Apply embedding
                    embedding = self.embedding_layers[feature_name](cat_indices)
                    
                    # Add the embedding to the features list
                    features_list.append(embedding)
            else:
                # For non-categorical features, keep them as is
                features_list.append(x[:, :, i].unsqueeze(2))
        
        # Concatenate all features along the last dimension
        embedded_x = torch.cat(features_list, dim=2)
        
        return embedded_x

    def create_masks(self, src_seq_len, tgt_seq_len, device):
        # Create causal mask for decoder self-attention (directly on the device)
        look_ahead_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=device), diagonal=1).bool()
        # Create mask for encoder (directly on the device)
        src_mask = torch.triu(torch.ones(src_seq_len, src_seq_len, device=device), diagonal=1).bool()
        
        return src_mask, look_ahead_mask

    def encode(self, src):
        """Encoder forward pass"""
        # Apply embeddings to categorical features
        src = self._apply_embeddings(src)
        
        # Project input features to d_model dimensions
        src = self.encoder_input_projection(src)
        src = self.positional_encoding(src)
        
        # Get sequence length for masks
        src_seq_len = src.size(1)
        device = src.device
        
        # Create causal mask for encoder (for time-series data) directly on the device
        src_mask = torch.triu(torch.ones(src_seq_len, src_seq_len, device=device), diagonal=1).bool()
        
        # Pass through encoder layers
        enc_output = src
        for layer in self.encoder_layers:
            enc_output = layer(enc_output, src_mask)
            
        return enc_output

    def decode(self, tgt, enc_output):
        """Decoder forward pass"""
        # Apply embeddings to categorical features
        tgt = self._apply_embeddings(tgt)
        
        # Project input features to d_model dimensions
        tgt = self.decoder_input_projection(tgt)
        tgt = self.positional_encoding(tgt)
        
        # Get sequence lengths for masks
        tgt_seq_len = tgt.size(1)
        device = tgt.device
        
        # Create causal mask for decoder self-attention directly on the device
        look_ahead_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=device), diagonal=1).bool()
        
        # Pass through decoder layers
        dec_output = tgt
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_output, look_ahead_mask)
            
        # Global average pooling across sequence dimension
        dec_output = dec_output.mean(dim=1)
        
        # Final projection to output dimension
        output = self.fc(self.dropout(dec_output))
        
        return output

    def forward(self, src, pred_len=1):
        """
        Forward pass of the transformer model with autoregressive prediction
        
        Args:
            src: Source sequence (input data) - (batch_size, seq_len, input_dim)
            pred_len: Number of time steps to predict
            
        Returns:
            output: Predicted values - (batch_size, pred_len)
        """
        # Pass through encoder
        enc_output = self.encode(src)
        
        # Autoregressive prediction
        batch_size = src.size(0)
        device = src.device
        
        # Initialize predictions tensor directly on the device
        predictions = torch.zeros(batch_size, pred_len, device=device)
        
        # Start with the last time step from source as initial decoder input
        curr_input = src[:, -1:, :].clone()
        
        # Create all predictions at once for better GPU utilization
        for t in range(pred_len):
            # Get prediction for current time step
            output = self.decode(curr_input, enc_output)  # Shape: [batch_size, 1]
            
            # Store the prediction
            predictions[:, t] = output.squeeze()
            
            # If we're not at the last time step, prepare input for next prediction
            if t < pred_len - 1:
                # Create a copy of the current input
                next_input = curr_input.clone()
                
                # Find the index of the Flow_scaled column (last column in our features)
                flow_idx = next_input.size(2) - 1
                
                # Update the flow value with our prediction for the next time step
                next_input[:, 0, flow_idx] = output.squeeze()
                
                # Use this as input for the next time step
                curr_input = next_input
        
        return predictions
