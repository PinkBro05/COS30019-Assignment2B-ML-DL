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
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Get prediction length from target
            pred_len = y_batch.size(1)
            
            # Forward pass with autoregressive prediction
            outputs = model(X_batch, pred_len=pred_len)
            
            # Compute loss
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
                
                # Get prediction length from target
                pred_len = y_batch.size(1)
                
                # Forward pass with autoregressive prediction
                outputs = model(X_batch, pred_len=pred_len)
                
                # Compute loss
                loss = criterion(outputs, y_batch)
                
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step(val_loss)
        
        end_time = time.time()
        
        # Print epoch results with improved formatting
        print("\n" + "="*20 + f" Epoch {epoch+1}/{num_epochs} " + "="*20)
        print(f"Train Loss (MSE): {train_loss:.6f}, Val Loss (MSE): {val_loss:.6f}")
        print(f"Learning Rate: {current_lr:.8f}, Time: {end_time - start_time:.2f}s")
        
        # Save the best model with improved logging
        if val_loss < best_val_loss and save_path:
            print(f"Validation loss improved from {best_val_loss:.6f} to {val_loss:.6f}")
            print(f"Saved!")
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
        else:
            print(f"Validation loss not improved - best: {best_val_loss:.6f}")
    
    return history

def test_transformer(model, test_loader, criterion, device, flow_scaler=None):
    """Test the Transformer model on the test dataset.
    
    Args:
        model: Trained Transformer model
        test_loader: Test data loader
        criterion: Loss function
        device: Device to test on
        flow_scaler: Scaler for the flow values (for denormalizing predictions)
        
    Returns:
        Dictionary with test metrics
    """
    model.to(device)
    model.eval()
    
    # Initialize metrics
    test_loss = 0
    mse_total = 0
    mae_total = 0
    predictions_list = []
    actuals_list = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Get prediction length from target
            pred_len = y_batch.size(1)
            
            # Forward pass with autoregressive prediction
            outputs = model(X_batch, pred_len=pred_len)
            
            # Compute loss
            loss = criterion(outputs, y_batch)
            
            # Calculate metrics
            test_loss += loss.item()
            mse = ((outputs - y_batch) ** 2).mean().item()
            mae = torch.abs(outputs - y_batch).mean().item()
            
            mse_total += mse
            mae_total += mae
            
            # Store predictions and actual values for further analysis
            if flow_scaler:
                # Convert to numpy and reshape for inverse transform
                outputs_np = outputs.cpu().numpy().reshape(-1, 1)
                y_batch_np = y_batch.cpu().numpy().reshape(-1, 1)
                
                # Inverse transform to get original scale
                outputs_original = flow_scaler.inverse_transform(outputs_np).flatten()
                y_batch_original = flow_scaler.inverse_transform(y_batch_np).flatten()
                
                predictions_list.extend(outputs_original)
                actuals_list.extend(y_batch_original)
    
    # Calculate average metrics
    num_batches = len(test_loader)
    avg_test_loss = test_loss / num_batches
    avg_mse = mse_total / num_batches
    avg_mae = mae_total / num_batches
    
    # Print metrics
    print("\n" + "="*50)
    print("Test Results:")
    print(f"Test Loss (MSE): {avg_test_loss:.6f}")
    print(f"Mean Squared Error: {avg_mse:.6f}")
    print(f"Mean Absolute Error: {avg_mae:.6f}")
    
    if flow_scaler and predictions_list:
        # Calculate metrics on original scale
        denorm_mse = np.mean((np.array(predictions_list) - np.array(actuals_list)) ** 2)
        denorm_mae = np.mean(np.abs(np.array(predictions_list) - np.array(actuals_list)))
        denorm_rmse = np.sqrt(denorm_mse)
        
        print("\nMetrics on Original Scale:")
        print(f"MSE: {denorm_mse:.2f}")
        print(f"MAE: {denorm_mae:.2f}")
        print(f"RMSE: {denorm_rmse:.2f}")
    
    print("="*50)
    
    # Return metrics as a dictionary
    metrics = {
        'test_loss': avg_test_loss,
        'mse': avg_mse,
        'mae': avg_mae
    }
    
    if flow_scaler and predictions_list:
        metrics.update({
            'denorm_mse': denorm_mse,
            'denorm_mae': denorm_mae,
            'denorm_rmse': denorm_rmse,
            'predictions': predictions_list,
            'actuals': actuals_list
        })
    
    return metrics

def load_data(args):
    """Load data for training.
    
    Args:
        args: Command line arguments containing data parameters
        
    Returns:
        Dictionary with data loaders and metadata
    """
    # Create a data collector with specified embedding dimension
    data_collector = TrafficDataCollector(embedding_dim=args.embedding_dim)
    
    # If chunking is enabled, return the data collector without loading data
    if args.use_chunking:
        print(f"Chunking enabled. Will process data in chunks of {args.chunk_size} rows.")
        return {
            'data_collector': data_collector,
            'chunking_enabled': True,
            'chunk_size': args.chunk_size
        }
    
    # Get data loaders from CSV file (non-chunked approach)
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
    parser.add_argument('--use_chunking', action='store_true',
                        help='Process data in chunks to handle large datasets')
    parser.add_argument('--chunk_size', type=int, default=100000,
                        help='Number of rows to process in each chunk when chunking is enabled')
    
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
    parser.add_argument('--epochs_per_chunk', type=int, default=10,
                        help='Number of epochs to train on each chunk when chunking is enabled')
    
    # Testing parameters
    parser.add_argument('--test', action='store_true',
                        help='Test the trained model instead of training')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the saved model for testing')
    parser.add_argument('--plot_test_results', default=True, action='store_true',
                        help='Plot test results when testing')
    parser.add_argument('--test_plot_name', type=str, default='transformer_test_results.png',
                        help='Name for the test results plot image')
    
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

def plot_test_results(predictions, actuals, test_plot_path):
    """Plot the test results.
    
    Args:
        predictions: List of predicted values
        actuals: List of actual values
        test_plot_path: Path to save the plot
    """
    # Create a figure for plotting test results
    plt.figure(figsize=(12, 6))
    
    # Get a sample of predictions to plot (to avoid overcrowding)
    sample_size = min(1000, len(predictions))
    indices = np.random.choice(len(predictions), sample_size, replace=False)
    
    pred_sample = np.array(predictions)[indices]
    actual_sample = np.array(actuals)[indices]
    
    # Scatter plot
    plt.scatter(actual_sample, pred_sample, alpha=0.5)
    
    # Add identity line (y=x)
    min_val = min(min(pred_sample), min(actual_sample))
    max_val = max(max(pred_sample), max(actual_sample))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add labels and title
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    plt.grid(True)
    
    # Add metrics to the plot
    mse = np.mean((pred_sample - actual_sample) ** 2)
    mae = np.mean(np.abs(pred_sample - actual_sample))
    rmse = np.sqrt(mse)
    
    plt.figtext(0.15, 0.85, f'MSE: {mse:.2f}\nMAE: {mae:.2f}\nRMSE: {rmse:.2f}', 
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(test_plot_path)
    print(f"Test results plot saved to {test_plot_path}")
    plt.show()

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
    
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save_figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    try:
        # Create data collector
        data_collector = TrafficDataCollector(embedding_dim=args.embedding_dim)
        
        # Load data
        print(f"Loading data from {args.data_file}...")
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
        
        train_loader = data_loaders['train_loader']
        val_loader = data_loaders['val_loader']
        test_loader = data_loaders['test_loader']
        categorical_indices = data_loaders['categorical_indices']
        categorical_metadata = data_loaders['categorical_metadata']
        encoders_scalers = data_loaders['encoders_scalers']
        
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
        
        # Define loss function
        criterion = nn.MSELoss()
        
        # Check if we're in test mode
        if args.test:
            # Set model path if not provided
            if args.model_path is None:
                args.model_path = save_path
                
            # Check if model exists
            if not os.path.exists(args.model_path):
                raise FileNotFoundError(f"Model file not found at {args.model_path}")
                
            print(f"Loading model from {args.model_path} for testing...")
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            
            # Test the model
            print("Testing the model...")
            test_metrics = test_transformer(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device,
                flow_scaler=encoders_scalers['flow_scaler']
            )
            
            # Plot test results if requested
            if args.plot_test_results and 'predictions' in test_metrics and 'actuals' in test_metrics:
                test_plot_path = os.path.join(figures_dir, args.test_plot_name)
                plot_test_results(
                    test_metrics['predictions'],
                    test_metrics['actuals'],
                    test_plot_path
                )
                
            print("Testing completed.")
            
        else:
            # Define optimizer and scheduler for training mode
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
                # Create subplot figure with 2 plots - loss and learning rate
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]})
                
                # Plot 1: Loss During Training
                ax1.plot(history['train_loss'], label='Training Loss', color='blue')
                ax1.plot(history['val_loss'], label='Validation Loss', color='orange')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Mean Squared Error')
                ax1.set_title('Loss During Training')
                ax1.legend()
                ax1.grid(True)
                
                # Plot 2: Learning Rate
                ax2.semilogy(history['learning_rate'], color='red')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Learning Rate')
                ax2.set_title('Learning Rate')
                ax2.grid(True)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save the plot
                plot_path = os.path.join(figures_dir, args.plot_name)
                plt.savefig(plot_path)
                print(f"Training plot saved to {plot_path}")
                plt.show()
            
            print(f"Training completed. Best model saved to {save_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the CSV data file and model files exist.")
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', 'Transformed')
        print(f"Example: python Transformer/supervised_learning.py --data_file {os.path.join(data_path, '_sample_final_time_series.csv')}")
        print(f"For testing: python Transformer/supervised_learning.py --data_file {os.path.join(data_path, '_sample_final_time_series.csv')} --test")

if __name__ == "__main__":
    main()