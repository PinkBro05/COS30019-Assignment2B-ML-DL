import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time
import sys
import argparse

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

def load_data(args):
    """Load data for training.
    
    Args:
        args: Command line arguments containing data parameters
        
    Returns:
        Dictionary with data loaders and metadata
    """
    # Create a data collector with specified embedding dimension
    data_collector = TrafficDataCollector(embedding_dim=args.embedding_dim)
    
    # Get data loaders from CSV file
    data_loaders = data_collector.get_data_loaders(
        data_file=args.data_file,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_seed
    )
    
    return data_loaders

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Transformer model for traffic flow prediction')
    
    # Data parameters
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to CSV data file')
    parser.add_argument('--embedding_dim', type=int, default=16,
                        help='Dimension for categorical embeddings (8, 16, or 32)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--shuffle', action='store_true', default=True,
                        help='Shuffle the training data')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='Ratio of test data')
    
    # Model parameters
    parser.add_argument('--d_model', type=int, default=64,
                        help='Dimension of transformer model')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=256,
                        help='Dimension of feed-forward network')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of encoder/decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for optimizer')
    
    # Other options
    parser.add_argument('--model_name', type=str, default='transformer_traffic_model.pth',
                        help='Model name for saving')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--no_plot', action='store_true',
                        help='Do not plot training history')
    parser.add_argument('--plot_name', type=str, default='transformer_training_history.png',
                        help='Name for the plot image')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    # Paths
    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, args.model_name)
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    
    try:
        # Load data
        data_loaders = load_data(args)
        
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
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            output_size=output_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            categorical_metadata=categorical_metadata,
            categorical_indices=categorical_indices
        )
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
            num_epochs=args.num_epochs,
            device=device,
            save_path=save_path
        )
        
        # Plot training history
        if not args.no_plot:
            plt.figure(figsize=(10, 6))
            plt.plot(history['train_loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            # Save the plot
            plot_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.plot_name)
            plt.savefig(plot_path)
            print(f"Training plot saved to {plot_path}")
            plt.show()
        
        print(f"Training completed. Best model saved to {save_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the CSV data file exists.")
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', 'Transformed')
        print(f"Example: python Transformer/supervised_learning.py --data_file {os.path.join(data_path, 'sample_long_format_revised.csv')}")

if __name__ == "__main__":
    main()