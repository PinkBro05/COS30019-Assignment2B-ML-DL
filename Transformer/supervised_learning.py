import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import sys

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.traffic_data_collector import TrafficDataCollector
from models.model import Transformer

def train_transformer(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    num_epochs,
    device,
    save_path=None
):
    """Train the Transformer model.
    
    Args:
        model: Transformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
        save_path: Path to save the best model
        
    Returns:
        Dictionary with training and validation losses
    """
    model.to(device)
    
    # To track the training and validation loss
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)
        
        # Save the best model
        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model to {save_path}")
        
        end_time = time.time()
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Time: {end_time - start_time:.2f}s")
    
    return history

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data parameters
    seq_length = 24  # 6 hours (15-min intervals)
    pred_length = 4  # 1 hour (15-min intervals)
    batch_size = 32
    
    # Model parameters
    d_model = 64  # Embedding dimension
    nhead = 8     # Number of attention heads
    num_encoder_layers = 3
    num_decoder_layers = 3
    dim_feedforward = 256
    dropout = 0.1
    
    # Training parameters
    num_epochs = 50
    learning_rate = 0.0001
    weight_decay = 1e-5
    
    # Paths
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'transformer_traffic_model.pth')
    
    # Load and prepare data
    print("Loading and preparing data...")
    data_collector = TrafficDataCollector()
    
    try:
        data_loaders = data_collector.get_data_loaders(
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Adjust based on your system
            random_state=42
        )
        
        train_loader = data_loaders['train_loader']
        val_loader = data_loaders['val_loader']
        
        # Get input and output dimensions from data
        X_batch, y_batch = next(iter(train_loader))
        input_dim = X_batch.shape[2]
        output_dim = y_batch.shape[2]
        
        print(f"Data loaded successfully:")
        print(f"  Input shape: [batch_size, seq_length, features] = {X_batch.shape}")
        print(f"  Output shape: [batch_size, pred_length, features] = {y_batch.shape}")
        
        # Create the Transformer model
        model = Transformer(
            input_dim=input_dim,
            output_dim=output_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            seq_length=seq_length,
            pred_length=pred_length
        )
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Train the model
        print("Starting training...")
        history = train_transformer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=num_epochs,
            device=device,
            save_path=save_path
        )
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'transformer_training_history.png'))
        plt.show()
        
        print(f"Training completed. Best model saved to {save_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the transform_traffic_data.py script first to prepare the data.")
        print(f"Command: python Utils/transform_traffic_data.py")

if __name__ == "__main__":
    main()