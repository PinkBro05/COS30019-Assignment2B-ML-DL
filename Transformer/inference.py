import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
from datetime import datetime, timedelta

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Transformer.utils.traffic_data_collector import TrafficDataCollector
from Transformer.models.model import TransformerModel

def load_model(model_path, args, device, input_dim, output_size, categorical_metadata=None, categorical_indices=None):
    """Load a trained Transformer model.
    
    Args:
        model_path: Path to the saved model
        args: Command line arguments
        device: Device to load the model on
        input_dim: Input dimension
        output_size: Output dimension
        categorical_metadata: Metadata for categorical features
        categorical_indices: Indices of categorical features
        
    Returns:
        Loaded model
    """
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
        Predicted sequence
    """
    model.eval()
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_seq).to(device)
        output = model(input_tensor)
    
    return output.cpu().numpy()

def plot_prediction(true_values, pred_values, site_id, date_str, time_points, args):
    """Plot the true and predicted sequences.
    
    Args:
        true_values: True values
        pred_values: Predicted values
        site_id: Traffic site ID
        date_str: Date string
        time_points: Time points for x-axis
        args: Command line arguments
    """
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, true_values, 'b-', marker='o', label='Actual Traffic Flow')
    plt.plot(time_points, pred_values, 'r-', marker='x', label='Predicted Traffic Flow')
    plt.xlabel('Time')
    plt.ylabel('Traffic Flow Volume')
    plt.title(f'Traffic Flow Prediction for Site {site_id} on {date_str}')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predictions')
    os.makedirs(plot_dir, exist_ok=True)
    plot_filename = f'prediction_site{site_id}_{date_str}.png'
    plt.savefig(os.path.join(plot_dir, plot_filename))
    print(f"Prediction plot saved to {os.path.join(plot_dir, plot_filename)}")
    
    if args.show_plot:
        plt.show()
    
    # Calculate evaluation metrics
    mae = np.mean(np.abs(true_values - pred_values))
    mape = np.mean(np.abs((true_values - pred_values) / (true_values + 1e-5))) * 100
    rmse = np.sqrt(np.mean(np.square(true_values - pred_values)))
    
    print(f"Evaluation Metrics for Site {site_id} on {date_str}:")
    print(f"  MAE: {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: {rmse:.2f}")

def load_data(args):
    """Load data for inference.
    
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
        batch_size=1,  # For inference we use batch size of 1
        shuffle=False,
        random_state=args.random_seed
    )
    
    return data_loaders

def parse_args():
    """Parse command line arguments for inference."""
    parser = argparse.ArgumentParser(description='Traffic Flow Prediction Inference using Transformer')
    
    # Data parameters
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to CSV data file')
    parser.add_argument('--embedding_dim', type=int, default=16,
                        help='Dimension for categorical embeddings (8, 16, or 32)')
    
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
    
    # Inference parameters
    parser.add_argument('--model_path', type=str, 
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                           'save_models', 'transformer_traffic_model.pth'),
                        help='Path to the trained model')
    parser.add_argument('--test_idx', type=int, default=None,
                        help='Index of the test sample to visualize (random if not specified)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--show_plot', action='store_true', default=True,
                        help='Show the prediction plot')
    parser.add_argument('--time_interval', type=int, default=15,
                        help='Time interval in minutes')
    
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
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    
    try:
        # Load data
        data_loaders = load_data(args)
        
        test_loader = data_loaders['test_loader']
        categorical_indices = data_loaders['categorical_indices']
        categorical_metadata = data_loaders['categorical_metadata']
        encoders_scalers = data_loaders['encoders_scalers']
        
        # Get input dimension from data
        X_sample, y_sample = next(iter(test_loader))
        input_dim = X_sample.shape[2]
        output_size = 1  # We're predicting Flow only
        
        print(f"Data loaded successfully:")
        print(f"  Test samples: {len(test_loader)}")
        print(f"  Input shape: [batch_size, seq_length, features] = {X_sample.shape}")
        print(f"  Output shape: [batch_size, pred_length] = {y_sample.shape}")
        
        # Load model
        model = load_model(
            args.model_path, 
            args, 
            device, 
            input_dim,
            output_size,
            categorical_metadata, 
            categorical_indices
        )
        print(f"Model loaded from {args.model_path}")
        
        # Choose a random sample for visualization if not specified
        if args.test_idx is None:
            import random
            random.seed(args.random_seed)
            test_idx = random.randint(0, len(test_loader) - 1)
        else:
            test_idx = args.test_idx
            
        print(f"Using test sample index: {test_idx}")
            
        # Get the sample and make prediction
        data_collector = TrafficDataCollector(embedding_dim=args.embedding_dim)
        metadata = data_collector.get_metadata()
        
        for i, (X, y) in enumerate(test_loader):
            if i == test_idx:
                input_seq = X.numpy()
                true_seq = y.numpy()[0]  # Remove batch dimension
                
                # Get metadata for this sample (if available)
                site_id = f"Sample_{test_idx}"
                date_str = "Test_Date"
                
                # Try to get actual site_id and date if available
                if 'sites' in metadata and len(metadata['sites']) > test_idx:
                    site_id = metadata['sites'][test_idx]
                if 'dates' in metadata and len(metadata['dates']) > test_idx:
                    date_str = metadata['dates'][test_idx]
                    if isinstance(date_str, np.datetime64):
                        date_str = str(date_str).split('T')[0]
                
                # Make prediction
                pred_seq = predict(model, input_seq, device)[0]  # Remove batch dimension
                
                # Create time points for x-axis
                start_time = datetime.strptime('00:00', '%H:%M')
                time_points = [(start_time + timedelta(minutes=i*args.time_interval)).strftime('%H:%M') 
                              for i in range(len(true_seq))]
                
                # Get original scale values if possible
                try:
                    true_values = data_collector.inverse_transform_flow(true_seq, encoders_scalers)
                    pred_values = data_collector.inverse_transform_flow(pred_seq, encoders_scalers)
                except:
                    # If inverse transform fails, use the normalized values
                    true_values = true_seq
                    pred_values = pred_seq
                
                # Plot prediction
                plot_prediction(true_values, pred_values, site_id, date_str, time_points, args)
                break
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the CSV data file exists.")
        data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data', 'Transformed')
        print(f"Example: python Transformer/inference.py --data_file {os.path.join(data_path, 'sample_long_format_revised.csv')} --model_path Transformer/save_models/transformer_traffic_model.pth")

if __name__ == "__main__":
    main()