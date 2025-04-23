from models.vanila_encoder import TransformerModel
from utils.data_collector import load_time_series_data
import torch
import numpy as np
import datetime
import os
import argparse

def inference(model, location, time_str, device, data_dict=None):
    """ Predicting next 1 hour traffic flow from given time and location

    Args:
        model: Trained TransformerModel
        location: Tuple of (SCATS Number, Location string)
        time_str: Time string in format "DD/MM/YYYY HH:MM:SS"
        device: Device to evaluate on
        data_dict: Dictionary containing scalers and feature information

    Returns:
        List of predicted flow values for the next 4 time frames (1 hour)
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
    
    # Create features for the next 4 time frames
    predictions = []
    
    # For each 15-minute interval in the next hour
    for i in range(4):
        # Calculate the target time (current time + i*15 minutes)
        target_time = dt + datetime.timedelta(minutes=15 * i)
        
        # Extract time features
        hour = target_time.hour
        minute = target_time.minute
        day_of_week = target_time.weekday()  # 0 = Monday, 6 = Sunday
        
        # Create input features: [SCATS Number, Location, Hour, Minute, DayOfWeek]
        features = [scats_number, hash(location_str) % 10000, hour, minute, day_of_week]  # Hash location to numeric
        
        # Convert to numpy array
        input_features = np.array([features], dtype=np.float32)
        
        # Apply scaling if scalers are available
        if data_dict and 'scalers' in data_dict and data_dict['scalers']:
            scaler_X = data_dict['scalers']['X']
            input_features = scaler_X.transform(input_features)
        
        # Convert to tensor and add sequence dimension if model expects it
        input_tensor = torch.tensor(input_features, dtype=torch.float32).unsqueeze(1)
        input_tensor = input_tensor.to(device)
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            
        # Convert prediction back to original scale
        if data_dict and 'scalers' in data_dict and data_dict['scalers']:
            scaler_y = data_dict['scalers']['y']
            pred = scaler_y.inverse_transform(output.cpu().numpy().reshape(-1, 1)).item()
        else:
            pred = output.item()
        
        predictions.append({
            'time': target_time.strftime("%H:%M"),
            'flow': max(0, round(pred))  # Ensure flow is non-negative and rounded to integer
        })
    
    return predictions

def predict_flow(model_path, location, time_str, device=None):
    """
    User-friendly function to predict traffic flow for the next hour
    
    Args:
        model_path: Path to the trained model file
        location: Tuple of (SCATS Number, Location string)
        time_str: Time string in format "DD/MM/YYYY HH:MM:SS"
        device: Device to run inference on (None for automatic selection)
        
    Returns:
        Dictionary with predictions for the next hour
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data to get scalers and feature info
    try:
        traffic_flow = load_time_series_data()
        data = traffic_flow.prepare_data_for_training(
            sequence_length=1,
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
            d_model=64,
            num_heads=8,
            d_ff=256,
            output_size=1,
            num_layers=2,
            dropout=0.1
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
    argparser.add_argument('--model_path', type=str, default='Transformer/save_models/transformer_model_test.pth',)
    
    # Path to the trained model
    model_path = argparser.parse_args().model_path
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please make sure you've trained the model first.")
        return
    
    # Specify device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Example location (SCATS number, location name)
    location = (970, "Swan St - Punt Rd")
    
    # Example time (format: DD/MM/YYYY HH:MM:SS)
    time_str = "23/04/2025 08:00:00"
    
    print(f"\nPredicting traffic flow for:")
    print(f"Location: {location[1]} (SCATS #{location[0]})")
    print(f"Time: {time_str}\n")
    
    # Make prediction
    try:
        result = predict_flow(
            model_path=model_path,
            location=location,
            time_str=time_str,
            device=device
        )
        
        # Display results
        if result:
            print("Prediction Results:")
            print(f"Location: {result['location'][1]} (SCATS #{result['location'][0]})")
            print(f"Reference time: {result['prediction_time']}")
            print("\nPredicted traffic flow for the next hour:")
            
            for i, pred in enumerate(result['predictions']):
                print(f"{pred['time']}: {pred['flow']} vehicles")
                
        else:
            print("Prediction failed. See error messages above.")
    
    except Exception as e:
        import traceback
        print(f"Error during prediction: {e}")
        traceback.print_exc()
    
if __name__ == "main":
    main()