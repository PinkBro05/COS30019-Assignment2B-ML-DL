import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

from utils.data_collector import load_time_series_data, create_time_series_dataloaders, show_sample_data
from models.vanila_encoder import TransformerModel
    
def train_model(model, train_loader, val_loader, optimizer, device, 
                num_epochs=50, patience=10):
    """
    Args:
        model: TransformerModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: PyTorch optimizer
        device: Device to train on (cpu or cuda)
        num_epochs: Maximum number of training epochs
        patience: Early stopping patience
    
    Returns:
        Dictionary with training history and trained model
    """
    model.to(device)
    
    # Loss function
    mse_loss = nn.MSELoss()
    
    # For early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Ensure outputs and targets have the right shape for multi-step prediction
            # outputs shape should be [batch_size, output_size] where output_size=4
            # targets shape should be [batch_size, output_size] where output_size=4
            loss = mse_loss(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Average loss over batches
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = mse_loss(outputs, targets)
                val_loss += loss.item()
        
        # Average loss over batches
        val_loss /= len(val_loader)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print progress
        print(f'Epoch {epoch+1}/{num_epochs} | '
              f'Train Loss (MSE): {train_loss:.4f} | '
              f'Val Loss (MSE): {val_loss:.4f}')
        
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
    Evaluate model on test data
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    mse_loss = nn.MSELoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # MSE loss calculated across all 4 prediction steps
            loss = mse_loss(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            
            # Store predictions and targets
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Calculate average MSE
    avg_loss = total_loss / len(test_loader.dataset)
    
    # Combine all predictions and targets
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate MSE for each time step
    step_mse = []
    for step in range(all_preds.shape[1]):
        step_mse.append(np.mean((all_preds[:, step] - all_targets[:, step]) ** 2))
    
    results = {
        'mse': avg_loss,
        'step_mse': step_mse,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    return results

def plot_training_history(history):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot training and validation loss
    plt.subplot(1, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("Transformer/transformer_training_history.png")
    plt.show()

def main():
    argparser = argparse.ArgumentParser(description="Train a Transformer model for traffic flow prediction")
    argparser.add_argument('--save_id', type=str, default='test', help='ID for saving the model')
    
    args = argparser.parse_args()
    
    try:
        # Load data
        traffic_flow = load_time_series_data()
        
        # Prepare data for supervised learning with explicit feature columns
        data = traffic_flow.prepare_data_for_training(
            sequence_length=4,  # Use 4 time steps as input (1 hour)
            prediction_horizon=4,  # Predict 4 time steps ahead (1 hour)
            scale_method='standard'  # Standardize data
        )
        
        # Show sample data
        # show_sample_data(data)
        
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
        output_size = 4  # Predicting next 4 time steps (1 hour)
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
            num_epochs=30,
            patience=10,
        )
        
        # Evaluate model
        results = evaluate_model(
            model=trained_model,
            test_loader=dataloaders['test'],
            device=device
        )
        
        print("\nTraining complete!")
        print(f"Test MSE: {results['mse']:.4f}")
        
        # Plot training history
        plot_training_history(history)
        
        # Save model
        torch.save(trained_model.state_dict(), f"Transformer/save_models/transformer_model_{args.save_id}.pth")
        print(f"Model saved to Transformer/save_models/transformer_model_{args.save_id}.pth")
        
    except Exception as e:
        import traceback
        print(f"Error in main: {e}")
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    main()