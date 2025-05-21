import os
import argparse
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score

# Add matplotlib configuration at the top before other imports
import matplotlib
matplotlib.use('TkAgg')  # TkAgg works well with PyCharm

# Add the parent directory to sys.path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Now import from local directories
from utils.data_processing import DataProcessor
from utils.visualization import plot_predictions, plot_training_history, plot_metrics, plot_accuracy_history
from model.traffic_prediction_model import TrafficPredictionModel

# Constants
DATA_PATH = "../Data/Raw/main/Scat_Data.csv"
SEQ_LEN = 10
EPOCHS = 50
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "save_models/lstm_traffic_model.pth" # Default model path


# Add this function to train.py
def analyze_dataset(df):
    """Analyze key characteristics of the dataset"""
    print("\n===== Dataset Analysis =====")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of unique SCATS sites: {df['NB_SCATS_SITE'].nunique()}")
    print(f"Day types: {df['day_type'].unique().tolist()}")
    print(f"Categorical columns: {[col for col in df.columns if df[col].dtype == 'object']}")
    
    # Show value distribution for traffic columns
    v_cols = [col for col in df.columns if col.startswith('V') and '_' in col]
    traffic_values = df[v_cols].values.flatten()
    print(f"Traffic values - min: {np.min(traffic_values)}, max: {np.max(traffic_values)}")
    print(f"Traffic values - mean: {np.mean(traffic_values):.2f}, std: {np.std(traffic_values):.2f}")

def build_lstm_model(input_shape, output_shape):
    """Build LSTM model architecture using custom implementation"""
    input_size = input_shape[1]  # Features
    hidden_size = 128
    output_size = output_shape
    
    model = TrafficPredictionModel(input_size, hidden_size, output_size)
    return model

def load_model(model, model_path):
    """Load pre-trained model weights"""
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print(f"Model loaded from {model_path}")
        return True
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return False

def train_model(model, train_loader, val_loader, epochs, device):
    """Train the PyTorch model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    # Remove the 'verbose' parameter to fix the deprecation warning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    lr_history = []
    train_accuracies = []  # Using R² as accuracy measure for regression
    val_accuracies = []    # Add validation R² tracking
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_targets = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
            # Store predictions and targets for accuracy calculation
            all_train_preds.append(y_pred.detach().cpu().numpy())
            all_train_targets.append(y_batch.detach().cpu().numpy())
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Calculate R² as an "accuracy" metric for regression
        all_train_preds = np.vstack(all_train_preds)
        all_train_targets = np.vstack(all_train_targets)
        train_r2 = r2_score(all_train_targets.flatten(), all_train_preds.flatten())
        train_accuracies.append(train_r2)
        
        # Validation
        model.eval()
        val_loss = 0
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
                # Store predictions and targets for validation accuracy calculation
                all_val_preds.append(y_pred.cpu().numpy())
                all_val_targets.append(y_batch.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate validation R²
        all_val_preds = np.vstack(all_val_preds)
        all_val_targets = np.vstack(all_val_targets)
        val_r2 = r2_score(all_val_targets.flatten(), all_val_preds.flatten())
        val_accuracies.append(val_r2)
        
        # Track best validation performance
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
        
        # Update learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        # Print learning rate change manually instead of relying on verbose=True
        if old_lr != current_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {current_lr:.6f}")
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Train R²: {train_r2:.4f}, Val R²: {val_r2:.4f}, LR: {current_lr:.6f}")
    
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'lr_history': lr_history,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_epoch': best_epoch
    }
    """Train the PyTorch model"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    train_losses = []
    val_losses = []
    lr_history = []
    train_accuracies = []  # Using R² as accuracy measure for regression
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        all_train_preds = []
        all_train_targets = []
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X_batch.size(0)
            
            # Store predictions and targets for accuracy calculation
            all_train_preds.append(y_pred.detach().cpu().numpy())
            all_train_targets.append(y_batch.detach().cpu().numpy())
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Calculate R² as an "accuracy" metric for regression
        all_train_preds = np.vstack(all_train_preds)
        all_train_targets = np.vstack(all_train_targets)
        train_r2 = r2_score(all_train_targets.flatten(), all_train_preds.flatten())
        train_accuracies.append(train_r2)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Store current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)
        
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, Training R²: {train_r2:.4f}, LR: {current_lr:.6f}")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'lr_history': lr_history,
        'train_accuracies': train_accuracies
    }



# Rest of the code remains the same

def prepare_data_loaders(X_train, y_train, X_test, y_test, batch_size):
    """Convert numpy arrays to PyTorch tensors and create data loaders"""
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create TensorDatasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Split training data for validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Traffic Prediction with LSTM')
    parser.add_argument('--load', action='store_true', help='Load pre-trained model instead of training')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Path to pre-trained model')
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs("save_models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Load and preprocess data using DataProcessor class
    print("Loading data...")
    data_processor = DataProcessor(DATA_PATH, SEQ_LEN)
    df = data_processor.load_data()
    analyze_dataset(df)  # Call the analysis function
    
    print("Preparing sequences...")
    X_train, X_test, y_train, y_test = data_processor.scale_and_split_data()
    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Testing data shape: {X_test.shape}, {y_test.shape}")
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(
        X_train, y_train, X_test, y_test, BATCH_SIZE
    )
    
    # Build model
    print("Building model...")
    input_shape = (SEQ_LEN, X_train.shape[2])
    output_shape = y_train.shape[1]
    model = build_lstm_model(input_shape, output_shape)
    model = model.to(DEVICE)
    print(model)
    
    # Either train or load the model based on arguments
    if not args.load:
        print("Training model...")
        history = train_model(
            model, train_loader, val_loader, EPOCHS, DEVICE
        )
        
        # Plot training history
        plot_training_history(
            history['train_losses'], 
            history['val_losses'],
            history['lr_history'],
            save_path="plots/training_history.png"
        )
        
        # Plot training and validation accuracy (R²)
        plot_accuracy_history(
            history['train_accuracies'],
            history['val_accuracies'],
            history['best_epoch'],
            save_path="plots/accuracy_history.png"
        )
        
        # Save model
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}!")
    else:
        print(f"Loading pre-trained model from {args.model_path}...")
        if not load_model(model, args.model_path):
            print("Failed to load model. Exiting.")
            return
    
    # Evaluate
    print("Evaluating model...")
    model.eval()
    y_pred_list = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(DEVICE)
            y_pred = model(X_batch)
            y_pred_list.append(y_pred.cpu().numpy())
    
    y_pred = np.vstack(y_pred_list)
    
    # Plot predictions
    plot_save_path = "plots/prediction_result.png"
    plot_predictions(y_test, y_pred, save_path=plot_save_path)
    print(f"Prediction plot saved to {plot_save_path}")
    
    # Plot performance metrics
    metrics_save_path = "plots/performance_metrics.png"
    metrics = plot_metrics(y_test, y_pred, save_path=metrics_save_path)
    print(f"Performance metrics plot saved to {metrics_save_path}")
    print(f"Model performance: MSE={metrics['MSE']:.6f}, RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}, R²={metrics['R2']:.6f}")

if __name__ == "__main__":
    main()