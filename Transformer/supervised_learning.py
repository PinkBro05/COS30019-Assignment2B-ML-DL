import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

from utils.data_collector import load_time_series_data, create_time_series_dataloaders
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
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss
            loss = mse_loss(outputs.squeeze(), targets)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss = mse_loss(outputs.squeeze(), targets)
                
                # Track metrics
                val_loss += loss.item() * inputs.size(0)
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader.dataset)
        
        # Save history
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
    
    test_mse = 0.0
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
            
            # Track metrics
            test_mse += mse.item() * inputs.size(0)
    
    # Calculate average test losses
    test_mse = test_mse / len(test_loader.dataset)
    
    print(f'Test Results | MSE: {test_mse:.4f}')
    
    return {
        'mse': test_mse,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets)
    }
    
def plot_training_history(history):
    """ Plot training history for loss and metrics

    Args:
        history: Dictionary with training history
    """
    plt.figure(figsize=(5, 5))
    
    # Losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
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
            sequence_length=4,  # Use 1 time step as input
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
            num_epochs=1,
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