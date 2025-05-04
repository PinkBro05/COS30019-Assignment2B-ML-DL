from models.vanila_encoder import TransformerModel
from utils.data_collector import load_time_series_data
import torch
import numpy as np
import datetime
import os
import argparse

def inference(model, location, time_str, device, data_dict=None):
    """ Predicting next 1 time step (15 mins) traffic flow from given time and location

    Args:
        model: Trained TransformerModel
        location: Tuple of (SCATS Number, Location string)
        time_str: Time string in format "DD/MM/YYYY HH:MM:SS"
        device: Device to evaluate on
        data_dict: Dictionary containing scalers and feature information

    Returns:
        List of predicted flow value for the next time step (15 mins)
    """
    model.to(device)
    model.eval()
    
    # Parse location tuple
    scats_number, location_str = location
    
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
                                   ['SCATS Number', 'Location', 'Hour', 'Minute', 'DayOfWeek'])
    
    # Create a sequence of 12 time steps (15-min intervals) as input
    sequence_data = []
    current_time = dt
    
    # Create a sequence of the current time and the 11 previous 15-minute intervals
    for i in range(12):
        time_step = current_time - datetime.timedelta(minutes=(11-i)*15)
        
        # Extract features
        features = {
            'SCATS Number': scats_number,  # Should be encoded, but for now use as is
            'Location': 0,  # Placeholder, will be encoded
            'Hour': time_step.hour,
            'Minute': time_step.minute,
            'DayOfWeek': time_step.weekday()
        }
        
        # Keep only the features mentioned in feature_columns
        features = {k: features[k] for k in feature_columns if k in features}
        
        # Add to sequence
        sequence_data.append([features[col] for col in feature_columns])
    
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

def predict_flow(model_path, location, time_str, device=None, d_model=64, num_heads=8, d_ff=256, num_layers=2, dropout=0.1):
    """
    User-friendly function to predict traffic flow for the next 15 minutes
    
    Args:
        model_path: Path to the trained model file
        location: Tuple of (SCATS Number, Location string)
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
            scale_method='standard'
        )
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
            input_dim = 5  # Default: SCATS Number, Location, Hour, Minute, DayOfWeek
        
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
    predictions = inference(model, location, time_str, device, data)
    
    # Format results
    result = {
        'location': location,
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
    argparser.add_argument('--location_name', type=str, default="WARRIGAL_RD N of HIGH STREET_RD", help='Location name')
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
    
    # Example location (SCATS number, location name)
    location = (args.scats_number, args.location_name)
    
    # Example time (format: DD/MM/YYYY HH:MM:SS)
    time_str = args.time
    
    print(f"\nPredicting traffic flow for:")
    print(f"Location: {location[1]} (SCATS #{location[0]})")
    print(f"Time: {time_str}\n")
    
    # Make prediction
    try:
        result = predict_flow(
            model_path=model_path,
            location=location,
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
            print(f"Location: {result['location'][1]} (SCATS #{result['location'][0]})")
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