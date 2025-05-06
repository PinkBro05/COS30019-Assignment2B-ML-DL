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
from Transformer.utils.traffic_data_collector import TrafficDataCollector
from Transformer.models.model import TransformerModel

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

def prepare_data(embedding_dim=16, use_existing=True):
    """Prepare the data for training.
    
    Args:
        embedding_dim: Dimension for categorical embeddings (8, 16, or 32)
        use_existing: Whether to use existing processed data or prepare new data
        
    Returns:
        Dictionary with data loaders and metadata
    """
    # Create a data collector with specified embedding dimension
    data_collector = TrafficDataCollector(embedding_dim=embedding_dim)
    
    # Prepare data from long-format traffic data if needed
    if not use_existing:
        data_collector.prepare_data(
            input_file='sample_long_format_revised.csv',  # or use full dataset
            seq_len=24,  # 6 hours (15-minute intervals)
            pred_len=8,  # 2 hours prediction horizon
            step_size=4  # Create a new sequence every hour
        )
    
    # Get data loaders
    data_loaders = data_collector.get_data_loaders(
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    return data_loaders

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Data parameters
    embedding_dim = 16  # Dimension for categorical embeddings (8, 16, or 32)
    
    # Model parameters
    d_model = 64  # Dimension of transformer model
    num_heads = 8  # Number of attention heads
    d_ff = 256  # Dimension of feed-forward network
    num_layers = 3  # Number of encoder/decoder layers
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
    
    try:
        # Prepare data with categorical embeddings
        data_loaders = prepare_data(embedding_dim=embedding_dim, use_existing=True)
        
        train_loader = data_loaders['train_loader']
        val_loader = data_loaders['val_loader']
        categorical_indices = data_loaders['categorical_indices']
        categorical_metadata = data_loaders['categorical_metadata']
        
        # Get input dimension from data
        X_batch, y_batch = next(iter(train_loader))
        input_dim = X_batch.shape[2]
        output_size = 1  # We're predicting Flow only
        
        print(f"Data loaded successfully:")
        print(f"  Input shape: [batch_size, seq_length, features] = {X_batch.shape}")
        print(f"  Output shape: [batch_size, pred_length] = {y_batch.shape}")
        
        # Print categorical features info
        print("\nCategorical features:")
        for feature, metadata in categorical_metadata.items():
            print(f"  {feature}: {metadata['num_classes']} classes, embedding dim={metadata['embedding_dim']}")
        
        # Create the Transformer model with categorical embeddings
        model = TransformerModel(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout,
            categorical_metadata=categorical_metadata,
            categorical_indices=categorical_indices
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
        print("Please run the reshape_table_to_time_series.py script first to prepare the long-format data.")
        print(f"Command: python Utils/reshape_table_to_time_series.py --input Data/Transformed/sample_final.csv --output Data/Transformed/sample_long_format_revised.csv")

if __name__ == "__main__":
    main()