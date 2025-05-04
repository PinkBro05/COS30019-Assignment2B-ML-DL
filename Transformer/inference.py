from models.vanila_encoder import TransformerModel
from utils.data_collector import load_time_series_data
import torch
import numpy as np
import datetime
import os
import argparse
import pandas as pd

def inference(model, scats_number, time_str, device, data_dict=None):
    """ Predicting next 1 time step (15 mins) traffic flow from given time and location

    Args:
        model: Trained TransformerModel
        scats_number: SCATS Number for the location
        time_str: Time string in format "DD/MM/YYYY HH:MM:SS"
        device: Device to evaluate on
        data_dict: Dictionary containing scalers and feature information

    Returns:
        List of predicted flow value for the next time step (15 mins)
    """
    model.to(device)
    model.eval()
    
    # Parse time string to datetime
    try:
        dt = datetime.datetime.strptime(time_str, "%d/%m/%Y %H:%M:%S")
    except ValueError:
        print("Error: Time format should be DD/MM/YYYY HH:MM:SS")
        return None
    
    # Create features for the next time frame
    predictions = []
    
    # Get feature columns from data_dict if available
    feature_columns = data_dict.get('feature_columns', 
                                   ['SCATS_Number', 'Hour', 'Minute', 'DayOfWeek'])
    
    # Get an estimate of average flow for fallback purposes
    estimated_flow = 100.0  # Default fallback value
    if data_dict and 'original_data' in data_dict:
        try:
            # Try to find the average flow at this time of day and day of week
            original_data = data_dict['original_data']
            day_of_week = dt.weekday()
            hour = dt.hour
            
            # Filter data for this SCATS number, hour, and day of week
            similar_data = original_data[
                (original_data['SCATS_Number'] == scats_number) & 
                (original_data['DayOfWeek'] == day_of_week) & 
                (original_data['Hour'] == hour)
            ]
            
            if not similar_data.empty and 'Flow' in similar_data.columns:
                # Use the average flow as our estimate
                estimated_flow = similar_data['Flow'].mean()
                print(f"Using estimated flow of {estimated_flow:.1f} based on historical data")
            else:
                # If no data for this specific time, use overall average
                all_flows = original_data[original_data['SCATS_Number'] == scats_number]['Flow']
                if not all_flows.empty:
                    estimated_flow = all_flows.mean()
                    print(f"Using general average flow of {estimated_flow:.1f}")
        except Exception as e:
            print(f"Error estimating flow: {e}")
    
    # Create a sequence of 12 time steps (15-min intervals) as input
    sequence_data = []
    current_time = dt
    
    # If we have the training data, get the categorical encodings for SCATS_Number
    scats_encoding = None
    if data_dict and 'original_data' in data_dict:
        # Get the mapping of SCATS numbers to their encoded values
        try:
            original_data = data_dict['original_data']
            if 'SCATS_Number' in original_data.columns:
                # Find the corresponding encoded value for this SCATS number
                scats_data = original_data[original_data['SCATS_Number'] == scats_number]
                if not scats_data.empty:
                    # Use the first encoded value found
                    scats_encoding = pd.Categorical(scats_data['SCATS_Number']).codes[0]
                else:
                    # Default fallback if this SCATS number wasn't in training
                    print(f"Warning: SCATS number {scats_number} not found in training data")
                    # Use the raw value, which might cause issues
                    scats_encoding = scats_number
        except Exception as e:
            print(f"Warning: Could not encode SCATS number: {e}")
            # Fallback to using the raw value
            scats_encoding = scats_number
    else:
        # No data dictionary available, use raw value
        scats_encoding = scats_number
    
    # Create a sequence of the current time and the 11 previous 15-minute intervals
    sequence_times = []
    for i in range(12):
        time_step = current_time - datetime.timedelta(minutes=(11-i)*15)
        sequence_times.append(time_step)
    
    # If we have the original data, try to find actual Flow values for these times
    actual_flows = [None] * 12  # Initialize with None values
    
    if data_dict and 'original_data' in data_dict:
        try:
            original_data = data_dict['original_data']
            
            # Make sure we have DateTime column to match by time
            if 'DateTime' not in original_data.columns and 'Date' in original_data.columns:
                # Convert 'Date' column to datetime if it's not already
                original_data['DateTime'] = pd.to_datetime(original_data['Date'])
            
            # Process each time in our sequence
            for i, seq_time in enumerate(sequence_times):
                # Find data points closest to each sequence time
                matching_data = original_data[
                    (original_data['SCATS_Number'] == scats_number) & 
                    (original_data['DateTime'].dt.date == seq_time.date()) &
                    (original_data['DateTime'].dt.hour == seq_time.hour) &
                    (original_data['DateTime'].dt.minute == seq_time.minute)
                ]
                
                if not matching_data.empty and 'Flow' in matching_data.columns:
                    # Use the actual Flow value
                    actual_flows[i] = matching_data['Flow'].values[0]
                    print(f"Found actual flow ({actual_flows[i]}) for {seq_time}")
            
            # Fill in any missing values with estimates
            for i in range(len(actual_flows)):
                if actual_flows[i] is None:
                    # If we don't have an actual value, try to estimate based on similar time patterns
                    time_step = sequence_times[i]
                    day_of_week = time_step.weekday()
                    hour = time_step.hour
                    minute = time_step.minute
                    
                    # Try to find data for the same SCATS, day of week, hour, and minute
                    similar_data = original_data[
                        (original_data['SCATS_Number'] == scats_number) & 
                        (original_data['DayOfWeek'] == day_of_week) & 
                        (original_data['Hour'] == hour) &
                        (original_data['Minute'] == minute)
                    ]
                    
                    if not similar_data.empty and 'Flow' in similar_data.columns:
                        # Use the average flow for this specific time slot
                        actual_flows[i] = similar_data['Flow'].mean()
                        print(f"Using similar time flow ({actual_flows[i]:.1f}) for {time_step}")
                    else:
                        # Try with just hour
                        similar_hour_data = original_data[
                            (original_data['SCATS_Number'] == scats_number) & 
                            (original_data['DayOfWeek'] == day_of_week) & 
                            (original_data['Hour'] == hour)
                        ]
                        
                        if not similar_hour_data.empty and 'Flow' in similar_hour_data.columns:
                            actual_flows[i] = similar_hour_data['Flow'].mean()
                            print(f"Using same hour flow ({actual_flows[i]:.1f}) for {time_step}")
                        else:
                            # Last resort - use overall average
                            actual_flows[i] = estimated_flow
                            print(f"Using estimated flow ({estimated_flow:.1f}) for {time_step}")
            
            print(f"Using actual/estimated flow values for sequence: {[round(f, 1) if f is not None else None for f in actual_flows]}")
            
        except Exception as e:
            print(f"Error finding actual flow values: {e}")
            # Fall back to using estimated flow for all time steps
            actual_flows = [estimated_flow] * 12
            print(f"Using default flow value of {estimated_flow:.1f} for all time steps")
    else:
        # No data dictionary available, use default for all time steps
        actual_flows = [estimated_flow] * 12
        print(f"Using default flow value of {estimated_flow:.1f} for all time steps (no historical data available)")
    
    # Create sequence data with actual or estimated flow values
    for i in range(12):
        time_step = sequence_times[i]
        
        # Extract features
        features = {
            'SCATS_Number': scats_encoding,
            'Hour': time_step.hour,
            'Minute': time_step.minute,
            'DayOfWeek': time_step.weekday()
        }
        
        # Keep only the features mentioned in feature_columns
        features = {k: features[k] for k in feature_columns if k in features}
        
        # Add to sequence - convert dict to list in the order of feature_columns
        feature_values = [features[col] for col in feature_columns if col in features]
        
        # Add the actual/estimated flow as an additional feature if the model was trained with it
        # Check if our model expects flow as a feature by examining input dimension
        include_target_as_feature = False
        if data_dict and 'X_train' in data_dict:
            # If X_train has more features than our feature_columns, 
            # it likely includes the target as a feature
            expected_num_features = data_dict['X_train'].shape[-1]
            if expected_num_features > len(feature_values):
                include_target_as_feature = True
                feature_values.append(actual_flows[i])
        
        sequence_data.append(feature_values)
    
    # Convert to numpy array
    input_sequence = np.array(sequence_data)
    
    # Scale features if scalers are available
    if data_dict and 'scalers' in data_dict and data_dict['scalers'] is not None:
        scaler_X = data_dict['scalers']['X']
        input_sequence = scaler_X.transform(input_sequence)
    
    # Convert to PyTorch tensor and add batch dimension
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        predicted_values = output.cpu().numpy()[0]  # Remove batch dimension
    
    # Inverse transform if scalers are available
    if data_dict and 'scalers' in data_dict and data_dict['scalers'] is not None:
        scaler_y = data_dict['scalers']['y']
        predicted_values = scaler_y.inverse_transform(predicted_values.reshape(-1, 1)).flatten()
    
    # Create a prediction with timestamp
    next_time = dt + datetime.timedelta(minutes=15)
    predictions.append({
        'time': next_time.strftime("%H:%M:%S"),
        'flow': round(float(predicted_values[0]), 1)
    })
    
    return predictions

def predict_flow(model_path, scats_number, time_str, device=None, d_model=64, num_heads=8, d_ff=256, num_layers=2, dropout=0.1):
    """
    User-friendly function to predict traffic flow for the next 15 minutes
    
    Args:
        model_path: Path to the trained model file
        scats_number: SCATS Number for the location
        time_str: Time string in format "DD/MM/YYYY HH:MM:SS"
        device: Device to run inference on (None for automatic selection)
        d_model: Hidden dimension size
        num_heads: Number of attention heads
        d_ff: Feed-forward layer dimension
        num_layers: Number of transformer layers
        dropout: Dropout rate
        
    Returns:
        Dictionary with predictions for the next 15 minutes
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data to get scalers and feature info
    try:
        traffic_flow = load_time_series_data()
        data = traffic_flow.prepare_data_for_training(
            sequence_length=12,
            prediction_horizon=1,
            scale_method='standard',
            include_target_as_feature=True  # Include previous Flow values as features
        )
        # Store original data for proper categorical encoding
        data['original_data'] = traffic_flow.data
    except Exception as e:
        print(f"Warning: Could not load training data for scaling: {e}")
        print("Predictions will not be properly scaled.")
        data = None
    
    # Load model
    try:
        # Get input dimensionality
        if data:
            input_dim = data['X_train'].shape[-1]
        else:
            input_dim = 4  # Default: SCATS Number, Hour, Minute, DayOfWeek
        
        # Create model with same architecture as training
        model = TransformerModel(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            output_size=1,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Load trained weights
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Run inference
    predictions = inference(model, scats_number, time_str, device, data)
    
    # Format results
    result = {
        'scats_number': scats_number,
        'prediction_time': time_str,
        'predictions': predictions
    }
    
    return result

def main():
    """
    Example usage of the traffic flow prediction model.
    """
    print("Traffic Flow Prediction Example")
    print("="*40)
    
    argparser = argparse.ArgumentParser(description="Traffic Flow Prediction")
    argparser.add_argument('--model_path', type=str, default='Transformer/save_models/transformer_model_test.pth', help='Path to the trained model file')
    argparser.add_argument('--scats_number', type=int, default=970, help='SCATS number for the location')
    argparser.add_argument('--time', type=str, default="10/1/2006 08:00:00", help='Time in DD/MM/YYYY HH:MM:SS format')
    
    # Model hyperparameters
    argparser.add_argument('--d_model', type=int, default=64, help='Hidden dimension size')
    argparser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    argparser.add_argument('--d_ff', type=int, default=256, help='Feed-forward layer dimension')
    argparser.add_argument('--num_layers', type=int, default=2, help='Number of transformer layers')
    argparser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    args = argparser.parse_args()
    
    # Path to the trained model
    model_path = args.model_path
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure you've trained the model first.")
        return
    
    # Specify device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # SCATS number for prediction
    scats_number = args.scats_number
    
    # Example time (format: DD/MM/YYYY HH:MM:SS)
    time_str = args.time
    
    print(f"\nPredicting traffic flow for:")
    print(f"SCATS #{scats_number}")
    print(f"Time: {time_str}\n")
    
    # Make prediction
    try:
        result = predict_flow(
            model_path=model_path,
            scats_number=scats_number,
            time_str=time_str,
            device=device,
            d_model=args.d_model,
            num_heads=args.num_heads,
            d_ff=args.d_ff,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        
        # Display results
        if result:
            print("Prediction Results:")
            print(f"SCATS #{result['scats_number']}")
            print(f"Reference time: {result['prediction_time']}")
            print("\nPredicted traffic flow for the next 15 minutes:")
            
            for i, pred in enumerate(result['predictions']):
                print(f"{pred['time']}: {pred['flow']} vehicles")
                
        else:
            print("Prediction failed. See error messages above.")
            
    except Exception as e:
        import traceback
        print(f"Error during prediction: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()