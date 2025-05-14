import os
import torch
import json

def save_model_metadata(model, filepath):
    """
    Save model architecture metadata to a JSON file
    
    Args:
        model: The model instance
        filepath: Path to save the metadata
    """
    metadata = {}
    
    # Save categorical metadata if available
    if hasattr(model, 'categorical_metadata') and model.categorical_metadata:
        metadata['categorical_metadata'] = model.categorical_metadata
    
    # Save categorical indices if available
    if hasattr(model, 'categorical_indices') and model.categorical_indices:
        metadata['categorical_indices'] = model.categorical_indices
    
    # Save input dimension
    if hasattr(model, 'input_dim'):
        metadata['input_dim'] = model.input_dim
    
    # Add other model parameters
    for key, value in model.__dict__.items():
        if key.startswith('_') or callable(value) or key in metadata:
            continue
        if isinstance(value, (int, float, str, bool, list, dict)):
            metadata[key] = value
    
    # Save metadata file
    metadata_path = os.path.splitext(filepath)[0] + '.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_path

def load_model_metadata(filepath):
    """
    Load model architecture metadata from a JSON file
    
    Args:
        filepath: Path to the model file
        
    Returns:
        dict: Model metadata
    """
    metadata_path = os.path.splitext(filepath)[0] + '.json'
    
    if not os.path.exists(metadata_path):
        return None
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    return metadata

def load_model_with_embedding_fix(model, model_path, device):
    """
    Load a model with a fix for embedding layer size mismatches
    
    This function handles embedding dimension mismatches when loading pre-trained models.
    Specifically, it addresses issues with:
    1. NB_SCATS_SITE embedding layer (4487 vs 1 classes)
    2. day_type embedding layer (10 vs 9 classes)
    
    Args:
        model: The model instance to load weights into
        model_path: Path to the saved model weights
        device: Device to load the model on
    
    Returns:
        model: The model with loaded weights
    """
    # Load the state dict from the saved model
    state_dict = torch.load(model_path, map_location=device)
    
    # Check if model metadata exists
    metadata = load_model_metadata(model_path)
    
    # If no metadata, attempt to load with normal approach or partial matching
    if not metadata:
        # Try to match parameters that have the same size
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    else:
        # If metadata exists, use it to fix embedding layers
        if 'categorical_metadata' in metadata:
            # Check embedding layer sizes
            for feature_name, metadata_feature in metadata['categorical_metadata'].items():
                # Get embedding layer from model
                if feature_name in model.embedding_layers:
                    saved_num_classes = metadata_feature['num_classes']
                    current_num_classes = model.categorical_metadata[feature_name]['num_classes']
                    embedding_dim = metadata_feature['embedding_dim']
                    
                    # If dimensions don't match, create a new embedding layer with correct dimensions
                    if saved_num_classes != current_num_classes:
                        print(f"Fixing embedding layer mismatch for '{feature_name}': {current_num_classes} -> {saved_num_classes}")
                        
                        # Create new embedding layer with dimensions matching saved model
                        model.embedding_layers[feature_name] = torch.nn.Embedding(saved_num_classes, embedding_dim)
                        
                        # Update current model's metadata to match saved model
                        model.categorical_metadata[feature_name]['num_classes'] = saved_num_classes
        
        # Now try to load the full state dict
        model.load_state_dict(state_dict)
    
    return model
