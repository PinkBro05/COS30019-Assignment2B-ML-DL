#!/usr/bin/env python3
"""
Inference script for the Transformer model for traffic flow prediction.
This script allows users to input a path to a test file, a row index, and get predictions 
for the next 4 time steps based on 24 input time steps.
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime, timedelta

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Transformer.utils.traffic_data_collector import TrafficDataCollector
from Transformer.models.model import TransformerModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions with the trained Transformer model')
    
    # Input parameters
    parser.add_argument('--input_path', type=str, default='Data/Transformed/_sample_final_time_series.csv',
                        help='Path to the test CSV file')
    parser.add_argument('--index', type=int, default= random.randint(0, 10000),
                        help='Row index in the CSV file to use as the prediction point')
    parser.add_argument('--num_steps', type=int, default=4,
                        help='Number of future time steps to predict (default: 4)')
    parser.add_argument('--model_path', type=str, default="Transformer/save_models/transformer_traffic_model.pth",
                        help='Path to the saved model (default: uses the model in save_models directory)')
    parser.add_argument('--embedding_dim', type=int, default=16,
                        help='Embedding dimension used during training (must match)')
    parser.add_argument('--d_model', type=int, default=64,
                        help='Model dimension used during training (must match)')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads used during training (must match)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers used during training (must match)')
    parser.add_argument('--d_ff', type=int, default=256,
                        help='Feed-forward dimension used during training (must match)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate used during training (must match)')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save the prediction plot (default: shows plot without saving)')
    
    return parser.parse_args()


def load_model(model_path, model_params, categorical_metadata, categorical_indices):
    """Load the trained model.
    
    Args:
        model_path: Path to the saved model
        model_params: Dictionary with model parameters
        categorical_metadata: Dictionary with categorical feature metadata
        categorical_indices: Dictionary with categorical feature indices
        
    Returns:
        Loaded model
    """
    # Create model with output_size=1 to match the saved model
    model = TransformerModel(
        input_dim=model_params['input_dim'],
        d_model=model_params['d_model'],
        num_heads=model_params['num_heads'],
        d_ff=model_params['d_ff'],
        output_size=1,  # The saved model outputs 1 value per prediction
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout'],
        categorical_metadata=categorical_metadata,
        categorical_indices=categorical_indices
    )
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model


def prepare_data_for_inference(data_collector, df, index, args, seq_len=24):
    """Prepare data for inference.
    
    Args:
        data_collector: TrafficDataCollector object
        df: DataFrame with the test data
        index: Index of the row to predict from
        args: Command line arguments
        seq_len: Number of time steps to use for prediction
        
    Returns:
        Dictionary with processed data for inference
    """
    # Check if index is valid
    if index < seq_len - 1:
        raise ValueError(f"Index {index} is too small. Need at least {seq_len-1} previous time steps.")
    if index >= len(df):
        raise ValueError(f"Index {index} is out of range. DataFrame has {len(df)} rows.")
    
    # Create cyclical features
    df_processed = data_collector._create_cyclical_features(df.copy())
    
    # Get SCATS site ID for the row of interest
    site_id = df.loc[index, 'NB_SCATS_SITE']
    
    # Process categorical features
    df_processed['NB_SCATS_SITE_encoded'] = data_collector.site_encoder.fit_transform(df_processed['NB_SCATS_SITE'])
    df_processed['scat_type_encoded'] = data_collector.scat_type_encoder.fit_transform(df_processed['scat_type'])
    df_processed['day_type_encoded'] = data_collector.day_type_encoder.fit_transform(df_processed['day_type'])
    
    # Standardize numerical features
    df_processed['school_count_scaled'] = data_collector.school_count_scaler.fit_transform(df_processed[['school_count']])
    df_processed['Flow_scaled'] = data_collector.flow_scaler.fit_transform(df_processed[['Flow']])
    
    # Create feature columns list (must match the order in training)
    feature_cols = [
        'day_of_week_sin', 'day_of_week_cos',  # Date features
        'month_sin', 'month_cos',
        'hour_sin', 'hour_cos',  # Time features
        'minute_sin', 'minute_cos',
        'NB_SCATS_SITE_encoded',  # Site ID
        'scat_type_encoded',  # Scat type
        'day_type_encoded',  # Day type
        'school_count_scaled',  # Standardized school count
        'Flow_scaled'  # Standardized flow (target)
    ]
    
    # Define categorical feature indices 
    categorical_indices = {
        'NB_SCATS_SITE': feature_cols.index('NB_SCATS_SITE_encoded'),
        'day_type': feature_cols.index('day_type_encoded')
    }
    
    # Get metadata for categorical features
    site_classes = len(data_collector.site_encoder.classes_)
    day_type_classes = len(data_collector.day_type_encoder.classes_)
    
    categorical_metadata = {
        'NB_SCATS_SITE': {
            'num_classes': site_classes,
            'embedding_dim': data_collector.embedding_dim
        },
        'day_type': {
            'num_classes': day_type_classes,
            'embedding_dim': data_collector.embedding_dim
        }
    }
    
    # Select features for the input sequence (index-seq_len+1 to index)
    features = df_processed.loc[index-seq_len+1:index, feature_cols].values
    
    # Prepare arrays for the next time steps as well (for autoregressive prediction)
    # If we don't have enough rows, create empty arrays with the same structure
    future_features = None
    if index + args.num_steps < len(df):
        # We have enough future data in the DataFrame
        future_features = df_processed.loc[index+1:index+args.num_steps, feature_cols].values
    else:
        # Create empty array with the same structure for future predictions
        future_shape = (min(args.num_steps, len(df) - index - 1), len(feature_cols))
        future_features = np.zeros(future_shape)
        
    # Original times for visualization
    times = df.loc[index-seq_len+1:index, 'time'].values
    dates = df.loc[index-seq_len+1:index, 'date'].values
    
    # Get times for future steps as well (for visualization)
    future_times = None
    future_dates = None
    if index + args.num_steps < len(df):
        future_times = df.loc[index+1:index+args.num_steps, 'time'].values
        future_dates = df.loc[index+1:index+args.num_steps, 'date'].values
        times = np.concatenate([times, future_times])
        dates = np.concatenate([dates, future_dates])
    else:
        # If we don't have enough future data, estimate the times
        last_time = pd.to_datetime(f"{dates[-1]} {times[-1]}")
        for i in range(1, args.num_steps + 1):
            # Assuming 15-minute intervals
            next_time = last_time + timedelta(minutes=15*i)
            new_date = next_time.strftime('%Y-%m-%d')
            new_time = next_time.strftime('%H:%M:%S')
            dates = np.append(dates, new_date)
            times = np.append(times, new_time)
            
    # Convert to datetime objects
    datetimes = [pd.to_datetime(f"{date} {time}") for date, time in zip(dates, times)]
    
    # Get actual values for the next steps (for comparison)
    actuals = None
    if index + args.num_steps < len(df):
        actuals = df.loc[index+1:index+args.num_steps, 'Flow'].values
    
    # Prepare input data for the model
    X = np.expand_dims(features, axis=0)  # Add batch dimension
    
    return {
        'X': X,
        'site_id': site_id,
        'categorical_indices': categorical_indices,
        'categorical_metadata': categorical_metadata,
        'times': datetimes,
        'actuals': actuals,
        'feature_cols': feature_cols,
        'features': features,
        'future_features': future_features,
        'encoders_scalers': {
            'site_encoder': data_collector.site_encoder,
            'day_type_encoder': data_collector.day_type_encoder,
            'flow_scaler': data_collector.flow_scaler
        }
    }


def predict_next_steps(model, X, num_steps=4, future_features=None, encoders_scalers=None):
    """Make autoregressive predictions for multiple steps.
    
    Args:
        model: Trained model
        X: Input data (numpy array with batch dimension)
        num_steps: Number of steps to predict
        future_features: Features for future time steps (excluding Flow)
        encoders_scalers: Dictionary with encoders and scalers
        
    Returns:
        Predicted values for num_steps time steps
    """
    with torch.no_grad():
        # Convert to tensor
        X_tensor = torch.FloatTensor(X)
        
        # Use the model's autoregressive prediction function directly
        predictions = model(X_tensor, pred_len=num_steps)
        
        # Convert predictions to numpy
        predictions = predictions.detach().numpy()
    
    return predictions


def visualize_prediction(input_data, predictions, flow_scaler, times, site_id, actuals=None, output_file=None):
    """Visualize the prediction.
    
    Args:
        input_data: Input data (numpy array)
        predictions: Predicted values (numpy array)
        flow_scaler: Scaler for the flow values
        times: List of datetime objects for x-axis
        site_id: SCATS site ID
        actuals: Actual values for comparison (optional)
        output_file: Path to save the plot (optional)
    """
    # Denormalize the predictions
    input_flows_scaled = input_data[:, -1]  # Last feature is the scaled flow
    input_flows_reshaped = input_flows_scaled.reshape(-1, 1)
    
    # Perform inverse transform to get the original values
    input_flows = flow_scaler.inverse_transform(input_flows_reshaped).flatten()
    
    # Denormalize the predictions
    predictions_reshaped = predictions.reshape(-1, 1)
    predicted_flows = flow_scaler.inverse_transform(predictions_reshaped).flatten()
    
    # Create a figure
    plt.figure(figsize=(14, 7))
    
    # Get x values for the time axis
    x_values = range(len(input_flows) + len(predicted_flows))
    
    # Define the prediction start point (last input point)
    pred_start_idx = len(input_flows) - 1
    
    # Plot the input flows
    plt.plot(x_values[:pred_start_idx+1], input_flows, 'b-', label='Input (Historical)')
    
    # Plot the predictions starting from the last input point
    plt.plot(x_values[pred_start_idx:pred_start_idx+len(predicted_flows)+1], 
             np.concatenate([[input_flows[-1]], predicted_flows]), 
             'r-', label='Prediction')
    
    # If we have actuals, plot them correctly starting from the last input point
    if actuals is not None:
        # Include the last input point to show continuity with actual values
        # Make sure we don't try to plot more actual values than we have
        actual_len = min(len(actuals), len(predicted_flows))
        plt.plot(x_values[pred_start_idx:pred_start_idx+actual_len+1], 
                 np.concatenate([[input_flows[-1]], actuals[:actual_len]]), 
                 'g-', label='Actual')
    
    # Format the x-axis with the dates including both date and time
    date_time_labels = [t.strftime('%Y-%m-%d %H:%M') for t in times]
    
    # Set date-time labels for the x-axis
    plt.xticks(x_values, date_time_labels, rotation=45, ha='right')
    
    # Set every 4th tick visible to avoid crowding
    for i, tick in enumerate(plt.gca().xaxis.get_major_ticks()):
        if i % 4 != 0:
            tick.set_visible(False)
    
    # Add a marker at the prediction start point
    plt.axvline(x=pred_start_idx, color='gray', linestyle='--', alpha=0.7)
    plt.text(pred_start_idx+0.5, plt.ylim()[1] * 0.95, 'Prediction starts', 
             fontsize=10, color='gray', ha='left', va='top',
             bbox=dict(facecolor='white', alpha=0.5))
    
    # Add grid, labels, title
    plt.grid(True, alpha=0.3)
    plt.xlabel('Date and Time')
    plt.ylabel('Traffic Flow (Vehicles)')
    plt.title(f'Traffic Flow Prediction for SCATS Site {site_id}')
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save the plot if an output file is specified
    if output_file:
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
    
    # Show the plot
    plt.show()


def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Create a data collector
        data_collector = TrafficDataCollector(embedding_dim=args.embedding_dim)
        
        # Load the test file
        print(f"Loading test data from {args.input_path}...")
        df = pd.read_csv(args.input_path)
        
        # Prepare data for inference
        print(f"Preparing data for inference at index {args.index}...")
        inference_data = prepare_data_for_inference(data_collector, df, args.index, args)
        
        # Set model path
        if args.model_path is None:
            args.model_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'save_models',
                'transformer_traffic_model.pth'
            )
        
        print(f"Loading model from {args.model_path}...")
        
        # Check if model file exists
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found at {args.model_path}")
        
        # Set model parameters
        model_params = {
            'input_dim': inference_data['X'].shape[2],
            'd_model': args.d_model,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'd_ff': args.d_ff,
            'dropout': args.dropout,
            'output_size': 1  # The saved model outputs 1 value per prediction
        }
        
        # Load the model
        model = load_model(
            args.model_path,
            model_params,
            inference_data['categorical_metadata'],
            inference_data['categorical_indices']
        )
        
        # Make prediction for the next steps
        print(f"Making prediction for the next {args.num_steps} time steps...")
        predictions = predict_next_steps(
            model, 
            inference_data['X'], 
            num_steps=args.num_steps,
            future_features=inference_data.get('future_features'),
            encoders_scalers=inference_data['encoders_scalers']
        )
        
        # Print the predictions
        flow_scaler = inference_data['encoders_scalers']['flow_scaler']
        predictions_reshaped = predictions.reshape(-1, 1)
        predictions_original = flow_scaler.inverse_transform(predictions_reshaped).flatten()
        
        print(f"\nPredictions for the next {args.num_steps} time steps:")
        for i, pred in enumerate(predictions_original):
            print(f"  Step {i+1}: {pred:.2f} vehicles")
        
        # Visualize the prediction
        print("\nVisualizing prediction...")
        visualize_prediction(
            inference_data['features'],
            predictions.flatten(),
            inference_data['encoders_scalers']['flow_scaler'],
            inference_data['times'],
            inference_data['site_id'],
            inference_data.get('actuals'),
            args.output_file
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