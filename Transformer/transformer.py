import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

from data_collector import load_time_series_data, create_time_series_dataloaders

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
    def __init__(self, input_dim, d_model, num_heads, d_ff, output_size=1, num_layers=2, dropout=0.1):
        super(TransformerModel, self).__init__()
        # Feature projection layer (instead of embedding)
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.fc = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Project input features to d_model dimensions
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        # Global average pooling across sequence dimension
        x = x.mean(dim=1)
        # Final projection to output dimension
        x = self.fc(self.dropout(x))
        return x
    
def train_model(model, train_loader, val_loader, optimizer, device, 
                num_epochs=50, patience=10, use_mse_and_mae=True):
    """
    Train the transformer model using both MSE and MAE loss functions
    
    Args:
        model: TransformerModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: PyTorch optimizer
        device: Device to train on (cpu or cuda)
        num_epochs: Maximum number of training epochs
        patience: Early stopping patience
        use_mse_and_mae: Whether to use both MSE and MAE loss functions
    
    Returns:
        Dictionary with training history
    """
    model.to(device)
    
    # Loss functions
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    
    # For early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'val_mse': [],
        'train_mae': [],
        'val_mae': []
    }
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mse_total = 0.0
        train_mae_total = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate losses
            mse = mse_loss(outputs.squeeze(), targets)
            mae = mae_loss(outputs.squeeze(), targets)
            
            # Combine losses if using both
            if use_mse_and_mae:
                loss = mse + mae  # Equal weighting
            else:
                loss = mse  # Use only MSE
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * inputs.size(0)
            train_mse_total += mse.item() * inputs.size(0)
            train_mae_total += mae.item() * inputs.size(0)
        
        # Calculate average training losses
        train_loss = train_loss / len(train_loader.dataset)
        train_mse = train_mse_total / len(train_loader.dataset)
        train_mae = train_mae_total / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mse_total = 0.0
        val_mae_total = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate losses
                mse = mse_loss(outputs.squeeze(), targets)
                mae = mae_loss(outputs.squeeze(), targets)
                
                # Combine losses if using both
                if use_mse_and_mae:
                    loss = mse + mae
                else:
                    loss = mse
                
                # Track metrics
                val_loss += loss.item() * inputs.size(0)
                val_mse_total += mse.item() * inputs.size(0)
                val_mae_total += mae.item() * inputs.size(0)
        
        # Calculate average validation losses
        val_loss = val_loss / len(val_loader.dataset)
        val_mse = val_mse_total / len(val_loader.dataset)
        val_mae = val_mae_total / len(val_loader.dataset)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        history['train_mae'].append(train_mae)
        history['val_mae'].append(val_mae)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss: {train_loss:.4f} (MSE: {train_mse:.4f}, MAE: {train_mae:.4f}) | '
              f'Val Loss: {val_loss:.4f} (MSE: {val_mse:.4f}, MAE: {val_mae:.4f})')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Could save best model here
            best_model_state = model.state_dict()
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                # Restore best model
                model.load_state_dict(best_model_state)
                break
    
    return history, model

def evaluate_model(model, test_loader, device):
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained TransformerModel
        test_loader: DataLoader for test data
        device: Device to evaluate on
    
    Returns:
        Dictionary with evaluation metrics
    """
    model.to(device)
    model.eval()
    
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    
    test_mse = 0.0
    test_mae = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Save predictions and targets
            all_predictions.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Calculate losses
            mse = mse_loss(outputs.squeeze(), targets)
            mae = mae_loss(outputs.squeeze(), targets)
            
            # Track metrics
            test_mse += mse.item() * inputs.size(0)
            test_mae += mae.item() * inputs.size(0)
    
    # Calculate average test losses
    test_mse = test_mse / len(test_loader.dataset)
    test_mae = test_mae / len(test_loader.dataset)
    
    print(f'Test Results | MSE: {test_mse:.4f}, MAE: {test_mae:.4f}')
    
    return {
        'mse': test_mse,
        'mae': test_mae,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets)
    }
    
def plot_training_history(history):
    """ Plot training history for loss and metrics

    Args:
        history: Dictionary with training history
    """
    plt.figure(figsize=(12, 5))
    
    # Losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Metrics
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mse'], label='Train MSE')
    plt.plot(history['val_mse'], label='Validation MSE')
    plt.plot(history['train_mae'], label='Train MAE')
    plt.plot(history['val_mae'], label='Validation MAE')
    plt.title('Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Metric Value')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
def main():
    argparser = argparse.ArgumentParser(description="Train a Transformer model for traffic flow prediction")
    argparser.add_argument('--mode', type=str, required= True, default='train', choices=['train', 'inference'],
                           help='Mode to run the script in: train or inference')
    argparser.add_argument('--save_path', type=str, default='Transformer/models/transformer_model_1.pth')
    argparser.add_argument('--load_path', type=str, default='Transformer/models/transformer_model_1.pth')
    
    args = argparser.parse_args()
    
    try:
        if args.mode == 'train':
                
            # Load data
            traffic_flow = load_time_series_data()
            
            # Prepare data for supervised learning (1 step input, predict next step)
            data = traffic_flow.prepare_data_for_training(
                sequence_length=1,  # Use 1 time step (1 hour) as input
                prediction_horizon=1,  # Predict 1 time step ahead
                scale_method='standard'  # Standardize data
            )
            
            # Create DataLoaders
            batch_size = 64
            dataloaders = create_time_series_dataloaders(data, batch_size=batch_size)
            
            # Get input dimensionality from data
            input_dim = data['X_train'].shape[-1]  # Number of features
            
            # Initialize model
            d_model = 64  # Hidden dimension
            num_heads = 8  # Number of attention heads
            d_ff = 256  # Feed-forward layer dimension
            num_layers = 2  # Number of transformer layers
            output_size = 1  # Predicting a single flow value
            dropout = 0.1
            
            model = TransformerModel(
                input_dim=input_dim,
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                output_size=output_size,
                num_layers=num_layers,
                dropout=dropout
            )
            
            # Setup training
            learning_rate = 0.001
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # Train model
            print(f"Training model on {device}...")
            history, trained_model = train_model(
                model=model,
                train_loader=dataloaders['train'],
                val_loader=dataloaders['val'],
                optimizer=optimizer,
                device=device,
                num_epochs=50,
                patience=10,
                use_mse_and_mae=True  # Use both MSE and MAE losses
            )
            
            # Evaluate model
            results = evaluate_model(
                model=trained_model,
                test_loader=dataloaders['test'],
                device=device
            )
            
            print("\nTraining complete!")
            print(f"Test MSE: {results['mse']:.4f}")
            print(f"Test MAE: {results['mae']:.4f}")
            
            # Plot training history
            plot_training_history(history)
            
            # Save model
            torch.save(trained_model.state_dict(), args.save_path)
            print(f"Model saved to {args.save_path}")
            
        else:
            pass
        
    except Exception as e:
        import traceback
        print(f"Error in main: {e}")
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    main()