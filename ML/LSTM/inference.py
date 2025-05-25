"""
Inference script for the LSTM model for traffic flow prediction.
This script allows users to input a path to a test file, a row index, and get predictions 
for the next time steps based on input sequence data.
"""

import os
import sys
import argparse
import numpy as np
import torch
# import pandas as pd  # Not directly used, accessed through data_processor
import random

# Add parent directory to path to import from utils
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import from local directories
from utils.data_processing import DataProcessor
from utils.visualization import plot_sequential_comparison, plot_daily_comparison, plot_metrics
from model.traffic_prediction_model import TrafficPredictionModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with the trained LSTM model and generate visualizations')
    
    # Input parameters
    parser.add_argument('--input_path', type=str, 
                        default=os.path.join(os.path.dirname(current_dir), 
                                            "Data/Transformed/2006_final_scats_data.csv"),
                        help='Path to the test CSV file')
    parser.add_argument('--index', type=int, default=random.randint(0, 10000),
                        help='Row index in the CSV file to use as the prediction point')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='Length of input sequence (default: 10)')
    parser.add_argument('--num_steps', type=int, default=4,
                        help='Number of future time steps to predict (default: 4)')
    parser.add_argument('--model_path', type=str, 
                        default=os.path.join(current_dir, "save_models/lstm_traffic_model.pth"),
                        help='Path to the saved model')
    
    # Visualization parameters (simplified to two main plots like train.py)
    parser.add_argument('--n_points', type=int, default=500,
                        help='Number of points for sequential comparison plot (default: 500)')
    parser.add_argument('--segment_index', type=int, default=0,
                        help='Which daily segment to plot for daily comparison (default: 0)')
    parser.add_argument('--output_dir', type=str, default='plots',
                        help='Directory to save visualization plots (default: plots)')
    
    return parser.parse_args()


def load_model(model_path, input_size, hidden_size, output_size):
    """Load the trained LSTM model.
    
    Args:
        model_path: Path to the saved model
        input_size: Input dimension for the model
        hidden_size: Hidden dimension for the LSTM layers
        output_size: Output dimension for the model
        
    Returns:
        Loaded model
    """
    # Create model with the same architecture as used in training
    model = TrafficPredictionModel(input_size, hidden_size, output_size)
    
    try:
        # Load the saved state dict
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def prepare_data_for_inference(data_processor, df, index, seq_len):
    """Prepare data for inference.
    
    Args:
        data_processor: DataProcessor object
        df: DataFrame with the test data
        index: Index of the row to predict from
        seq_len: Length of the input sequence
        
    Returns:
        Dictionary with processed data for inference
    """
    # Check if index is valid
    if index < seq_len:
        raise ValueError(f"Index {index} is too small. Need at least {seq_len} previous time steps.")
    if index >= len(df):
        raise ValueError(f"Index {index} is out of range. DataFrame has {len(df)} rows.")
    
    # Prepare feature data similar to training
    if 'Flow' in df.columns:
        # Use Flow column directly as in training - ensure only numeric data
        feature_data = df[['Flow']].values
    else:
        # Use traffic volume columns if available (V00_0 to V95_0)
        feature_cols = [col for col in df.columns if col.startswith('V') and '_' in col]
        
        # Include categorical features if needed
        categorical_cols = [col for col in df.columns if col.startswith('day_') or col.startswith('scat_')]
        if categorical_cols:
            feature_cols.extend(categorical_cols)
            
        # If no specific features found, use all numeric columns (excluding Date column)
        if not feature_cols:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            # Remove non-flow related columns that might not be useful for prediction
            exclude_cols = ['Date', 'SCATS_Number']  # Exclude ID-like columns
            feature_cols = [col for col in feature_cols if col not in exclude_cols]
            
        feature_data = df[feature_cols].values
    
    # Fit and transform the data (since we need to fit the scaler for inference)
    scaled_data = data_processor.scaler.fit_transform(feature_data)
    
    # Extract the input sequence
    input_sequence = scaled_data[index-seq_len:index]
    
    # Extract actual values for comparison if available
    actuals = None
    if index + 1 < len(df):
        # Only get the target feature values for actuals - handle Flow column specifically
        if 'Flow' in df.columns:
            actuals = df['Flow'].iloc[index:index+1].values.reshape(-1, 1)
        else:
            actuals = df[feature_cols].iloc[index:index+1].values
        actuals = data_processor.inverse_transform(actuals)
    
    # Convert to torch tensor and add batch dimension
    X = torch.FloatTensor(input_sequence).unsqueeze(0)  # [1, seq_len, features]
    
    return {
        'X': X,
        'actuals': actuals,
        'index': index,
        'original_data': df,
        'scaled_data': scaled_data
    }


def predict(model, X, num_steps=1):
    """Make predictions with the model.
    
    Args:
        model: Trained model
        X: Input tensor [batch_size, seq_len, features]
        num_steps: Number of steps to predict
        
    Returns:
        Predicted values
    """
    with torch.no_grad():
        predictions = []
        current_input = X.clone()
        
        for step in range(num_steps):
            # Make prediction for current step
            output = model(current_input)
            predictions.append(output.numpy())
            
            # For multi-step, update input sequence by removing first element and adding prediction
            if step < num_steps - 1:
                # Create new input by shifting the sequence and adding the prediction
                new_input = torch.cat([
                    current_input[:, 1:, :],  # Remove first timestep
                    output.unsqueeze(1)       # Add prediction as new timestep
                ], dim=1)
                current_input = new_input
        
        return np.concatenate(predictions, axis=1)


def create_evaluation_visualizations(data_processor, input_data, predictions, df, index, seq_len, num_steps, n_points, segment_index, output_dir):
    """Create evaluation visualizations using adaptable functions from visualization.py."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert input data and predictions to proper format
    input_sequence = input_data.squeeze(0).numpy()
    input_sequence_original = data_processor.inverse_transform(input_sequence)
    predictions_original = data_processor.inverse_transform(predictions.reshape(-1, predictions.shape[-1]))
    
    # Extract predicted values
    predicted_values = predictions_original[:, -1]
    
    # Get actual values for the prediction period if available
    actual_values = None
    if index + num_steps <= len(df) and 'Flow' in df.columns:
        # Get historical actual values (for context) + future actual values
        history_length = min(n_points - num_steps, index)  # Get history but leave room for predictions
        start_idx = max(0, index - history_length)
        
        # Get historical + future actual values
        historical_actual = df['Flow'].iloc[start_idx:index].values
        future_actual = df['Flow'].iloc[index:index+num_steps].values
        full_actual = np.concatenate([historical_actual, future_actual])
        
        # Create matching predictions array (NaN for historical part)
        historical_pad = np.full(len(historical_actual), np.nan)
        full_predicted = np.concatenate([historical_pad, predicted_values])
        
        # Get just future actual values for other plots
        actual_values = future_actual
        
        # 1. Sequential comparison plot (like in train.py)
        seq_save_path = os.path.join(output_dir, f"inference_sequential_comparison_points{n_points}.png")
        plot_sequential_comparison(full_actual, full_predicted, 
                                  n_points=min(n_points, len(full_actual)), 
                                  save_path=seq_save_path)
        print(f"Sequential comparison plot saved to {seq_save_path}")
        
        # 2. Daily comparison plot (if we have enough data)
        if len(actual_values) >= 96:  # At least one day of data (assuming 15-min intervals)
            daily_save_path = os.path.join(output_dir, f"inference_daily_comparison_segment{segment_index+1}.png")
            plot_daily_comparison(actual_values, predicted_values, 
                                 segment_index=segment_index, 
                                 tick_interval=4, 
                                 save_path=daily_save_path)
            print(f"Daily comparison plot saved to {daily_save_path}")
        
        # 3. Performance metrics plot
        metrics_save_path = os.path.join(output_dir, "inference_performance_metrics.png")
        metrics = plot_metrics(actual_values.reshape(-1, 1), predicted_values.reshape(-1, 1), 
                              save_path=metrics_save_path)
        print(f"Performance metrics plot saved to {metrics_save_path}")
        print(f"Inference performance: MSE={metrics['MSE']:.6f}, RMSE={metrics['RMSE']:.6f}, MAE={metrics['MAE']:.6f}, R²={metrics['R2']:.6f}")
    
    else:
        print("No actual values available for comparison. Prediction completed successfully.")
        print(f"Generated {num_steps} predictions starting from index {index}")
        print(f"Predicted values: {predicted_values}")
        
        # For predictions-only case, create a simple visualization using basic output
        print(f"Predicted traffic flow values for the next {num_steps} time steps:")
        for i, val in enumerate(predicted_values):
            print(f"  Step {i+1}: {val:.2f}")
        
        print(f"Predictions saved successfully. To visualize with comparison data, ensure actual values are available.")

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    try:
        print(f"Loading data from {args.input_path}...")
        
        # Initialize data processor
        data_processor = DataProcessor(args.input_path, args.seq_len)
        
        # Load data
        df = data_processor.load_data()
        print(f"Data loaded, shape: {df.shape}")
        
        # Prepare data for inference
        print(f"Preparing data for inference at index {args.index}...")
        inference_data = prepare_data_for_inference(data_processor, df, args.index, args.seq_len)
        
        # Determine model parameters
        input_size = inference_data['X'].shape[2]  # Number of features
        hidden_size = 128  # Same as in training
        output_size = 1    # Predicting a single value
        
        # Load model
        print(f"Loading model from {args.model_path}...")
        model = load_model(args.model_path, input_size, hidden_size, output_size)
        
        if model is None:
            print("Failed to load model. Exiting.")
            return
        
        # Make prediction
        print(f"Making prediction for the next {args.num_steps} time steps...")
        predictions = predict(model, inference_data['X'], args.num_steps)
        
        # Create evaluation visualizations
        print("Creating evaluation visualizations...")
        create_evaluation_visualizations(
            data_processor,
            inference_data['X'],
            predictions,
            inference_data['original_data'],
            args.index,
            args.seq_len,
            args.num_steps,
            args.n_points,
            args.segment_index,
            args.output_dir
        )
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()