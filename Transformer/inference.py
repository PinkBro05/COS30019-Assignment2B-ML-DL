import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import argparse
from datetime import datetime, timedelta

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.traffic_data_collector import TrafficDataCollector
from models.model import Transformer

def load_model(model_path, model_params, device):
    """Load a trained Transformer model.
    
    Args:
        model_path: Path to the saved model
        model_params: Dictionary with model parameters
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    model = Transformer(
        input_dim=model_params['input_dim'],
        output_dim=model_params['output_dim'],
        d_model=model_params['d_model'],
        nhead=model_params['nhead'],
        num_encoder_layers=model_params['num_encoder_layers'],
        num_decoder_layers=model_params['num_decoder_layers'],
        dim_feedforward=model_params['dim_feedforward'],
        dropout=model_params['dropout'],
        seq_length=model_params['seq_length'],
        pred_length=model_params['pred_length']
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def predict(model, input_seq, device):
    """Make a prediction using the Transformer model.
    
    Args:
        model: Trained Transformer model
        input_seq: Input sequence (shape: [batch_size, seq_length, input_dim])
        device: Device to run the prediction on
        
    Returns:
        Predicted sequence (shape: [batch_size, pred_length, output_dim])
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_seq).to(device)
        output = model(input_tensor)
    
    return output.cpu().numpy()

def plot_prediction(true_seq, pred_seq, site_id, date, scaler, feature_idx=0, time_interval=15):
    """Plot the true and predicted sequences.
    
    Args:
        true_seq: True sequence (shape: [pred_length, output_dim])
        pred_seq: Predicted sequence (shape: [pred_length, output_dim])
        site_id: Traffic site ID
        date: Date of the sequence
        scaler: Dictionary with normalization parameters
        feature_idx: Index of the feature to plot
        time_interval: Time interval in minutes
    """
    # Get the prediction length
    pred_length = true_seq.shape[0]
    
    # Inverse transform the sequences
    data_collector = TrafficDataCollector()
    true_seq_orig = data_collector.inverse_transform(true_seq, scaler, is_target=True)
    pred_seq_orig = data_collector.inverse_transform(pred_seq, scaler, is_target=True)
    
    # Extract the specified feature
    true_values = true_seq_orig[:, feature_idx]
    pred_values = pred_seq_orig[:, feature_idx]
    
    # Create time points for x-axis
    start_time = datetime.strptime('00:00', '%H:%M')
    time_points = [(start_time + timedelta(minutes=i*time_interval)).strftime('%H:%M') 
                   for i in range(pred_length)]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, true_values, 'b-', marker='o', label='Actual Traffic Flow')
    plt.plot(time_points, pred_values, 'r-', marker='x', label='Predicted Traffic Flow')
    plt.xlabel('Time')
    plt.ylabel('Traffic Flow Volume')
    plt.title(f'Traffic Flow Prediction for Site {site_id} on {date}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions')
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, f'prediction_site{site_id}_{date}.png'))
    plt.show()
    
    # Calculate evaluation metrics
    mae = np.mean(np.abs(true_values - pred_values))
    mape = np.mean(np.abs((true_values - pred_values) / (true_values + 1e-5))) * 100
    rmse = np.sqrt(np.mean(np.square(true_values - pred_values)))
    
    print(f"Evaluation Metrics for Site {site_id} on {date}:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: {rmse:.2f}")

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Model parameters (should match trained model)
    model_params = {
        'input_dim': None,  # Will be set from data
        'output_dim': None,  # Will be set from data
        'd_model': 64,
        'nhead': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.1,
        'seq_length': 24,
        'pred_length': 4
    }
    
    # Paths
    model_path = args.model_path
    
    # Load data
    print("Loading test data...")
    data_collector = TrafficDataCollector()
    
    try:
        # Get test data loader
        data_loaders = data_collector.get_data_loaders(
            batch_size=1,  # For inference we use batch size of 1
            shuffle=False,
            random_state=42
        )
        
        test_loader = data_loaders['test_loader']
        scaler = data_loaders['scaler']
        
        # Get input and output dimensions from data
        X_sample, y_sample = next(iter(test_loader))
        model_params['input_dim'] = X_sample.shape[2]
        model_params['output_dim'] = y_sample.shape[2]
        
        print(f"Data loaded successfully.")
        print(f"  Test samples: {len(test_loader)}")
        print(f"  Input shape: {X_sample.shape}")
        print(f"  Output shape: {y_sample.shape}")
        
        # Load model
        model = load_model(model_path, model_params, device)
        print(f"Model loaded from {model_path}")
        
        # Choose a random sample for visualization if not specified
        if args.test_idx is None:
            test_idx = random.randint(0, len(test_loader) - 1)
        else:
            test_idx = args.test_idx
            
        # Get the sample
        for i, (X, y) in enumerate(test_loader):
            if i == test_idx:
                input_seq = X.numpy()
                true_seq = y.numpy()[0]  # Remove batch dimension
                
                # Get site_id and date
                sites_dates = data_collector.load_data()
                site_id = sites_dates['sites'][test_idx]
                date = sites_dates['dates'][test_idx]
                
                # Make prediction
                pred_seq = predict(model, input_seq, device)[0]  # Remove batch dimension
                
                # Plot prediction
                plot_prediction(true_seq, pred_seq, site_id, date, scaler, feature_idx=args.feature_idx)
                break
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the transform_traffic_data.py script first to prepare the data.")
        print(f"Command: python Utils/transform_traffic_data.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Traffic Flow Prediction using Transformer')
    parser.add_argument('--model_path', type=str, 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                           'save_models', 'transformer_traffic_model.pth'),
                        help='Path to the trained model')
    parser.add_argument('--test_idx', type=int, default=None,
                        help='Index of the test sample to visualize (random if not specified)')
    parser.add_argument('--feature_idx', type=int, default=0,
                        help='Index of the feature to visualize (default: 0 - first flow feature)')
    
    args = parser.parse_args()
    main(args)