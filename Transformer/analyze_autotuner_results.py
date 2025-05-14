"""
Utility script to analyze and visualize AutoTuner results.
"""
import os
import sys
import argparse
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import optuna
import optuna.visualization as vis
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_study(study_file):
    """
    Load an Optuna study from file.
    
    Args:
        study_file: Path to the Optuna study file
        
    Returns:
        Optuna study object
    """
    if not os.path.exists(study_file):
        print(f"Error: Study file {study_file} not found")
        return None
    
    try:
        study = joblib.load(study_file)
        return study
    except Exception as e:
        print(f"Error loading study: {e}")
        return None

def create_analysis_plots(study, output_dir):
    """
    Create analysis plots for an Optuna study.
    
    Args:
        study: Optuna study object
        output_dir: Directory to save the plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Plot optimization history
        history_fig = vis.plot_optimization_history(study)
        history_fig.write_image(os.path.join(output_dir, 'optimization_history.png'))
        
        # Plot parameter importances
        importance_fig = vis.plot_param_importances(study)
        importance_fig.write_image(os.path.join(output_dir, 'param_importances.png'))
        
        # Plot parallel coordinate
        parallel_fig = vis.plot_parallel_coordinate(study)
        parallel_fig.write_image(os.path.join(output_dir, 'parallel_coordinate.png'))
        
        # Plot slice for important parameters
        slice_fig = vis.plot_slice(study)
        slice_fig.write_image(os.path.join(output_dir, 'param_slices.png'))
        
        # Plot contour
        try:
            contour_fig = vis.plot_contour(study)
            contour_fig.write_image(os.path.join(output_dir, 'contour.png'))
        except Exception as e:
            print(f"Could not generate contour plot: {e}")
        
        print(f"Analysis plots saved to {output_dir}")
    except Exception as e:
        print(f"Error generating plots: {e}")

def print_study_summary(study):
    """
    Print a summary of the study results.
    
    Args:
        study: Optuna study object
    """
    print("\n" + "="*60)
    print("Study Summary")
    print("="*60)
    
    # Basic statistics
    print(f"Number of completed trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    # Best trial
    best_trial = study.best_trial
    print("\nBest Trial:")
    print(f"  Value (Loss): {best_trial.value:.6f}")
    print(f"  Trial number: {best_trial.number}")
    
    # Best parameters
    print("\nBest Parameters:")
    for param_name, param_value in best_trial.params.items():
        print(f"  {param_name}: {param_value}")
    
    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print("\nParameter Importance:")
        for param_name, score in importance.items():
            print(f"  {param_name}: {score:.4f}")
    except Exception as e:
        print(f"\nCould not compute parameter importance: {e}")
    
    print("="*60)

def compare_models(results_dir):
    """
    Compare models from multiple autotuning runs.
    
    Args:
        results_dir: Directory containing multiple autotuning result directories
    """
    # Find all metadata files
    metadata_files = []
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.endswith('_metadata.json') and 'best_model' in file:
                metadata_files.append(os.path.join(root, file))
    
    if not metadata_files:
        print("No model metadata files found")
        return
    
    # Load metadata
    models_data = []
    for file_path in metadata_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Add dirname as run name
                data['run_name'] = os.path.basename(os.path.dirname(file_path))
                models_data.append(data)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Extract metrics and parameters
    comparison = []
    for data in models_data:
        metrics = data.get('test_metrics', {})
        params = data.get('params', {})
        
        row = {
            'run_name': data['run_name'],
            'val_loss': data.get('val_loss', float('inf')),
            'mse': metrics.get('mse', float('inf')),
            'mae': metrics.get('mae', float('inf')),
            'd_model': params.get('d_model', 'N/A'),
            'num_heads': params.get('num_heads', 'N/A'),
            'num_layers': params.get('num_layers', 'N/A'),
            'dropout': params.get('dropout', 'N/A'),
            'learning_rate': params.get('learning_rate', 'N/A'),
            'batch_size': params.get('batch_size', 'N/A')
        }
        comparison.append(row)
    
    # Sort by validation loss
    comparison.sort(key=lambda x: x['val_loss'])
    
    # Print comparison
    print("\n" + "="*100)
    print("Model Comparison")
    print("="*100)
    header = ('Run Name', 'Val Loss', 'MSE', 'MAE', 'd_model', 'Heads', 'Layers', 'Dropout', 'LR', 'Batch')
    print(f"{header[0]:<20} {header[1]:<10} {header[2]:<10} {header[3]:<10} {header[4]:<8} "
          f"{header[5]:<8} {header[6]:<8} {header[7]:<10} {header[8]:<12} {header[9]:<8}")
    print("-" * 100)
    
    for row in comparison:
        print(f"{row['run_name']:<20} {row['val_loss']:<10.6f} {row['mse']:<10.6f} {row['mae']:<10.6f} "
              f"{row['d_model']:<8} {row['num_heads']:<8} {row['num_layers']:<8} {row['dropout']:<10.4f} "
              f"{row['learning_rate']:<12.6f} {row['batch_size']:<8}")
    
    print("="*100)

def main():
    parser = argparse.ArgumentParser(description='Analyze AutoTuner results')
    
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing AutoTuner results')
    parser.add_argument('--study_file', type=str, default=None,
                        help='Specific Optuna study file to analyze')
    parser.add_argument('--compare', action='store_true',
                        help='Compare models from multiple autotuning runs')
    
    args = parser.parse_args()
    
    if args.compare:
        # Compare models from multiple runs
        compare_models(args.results_dir)
        return
    
    # Find the study file if not provided
    if args.study_file is None:
        study_files = []
        for file in os.listdir(args.results_dir):
            if file.endswith('.pkl') and 'transformer_autotuner' in file:
                study_files.append(os.path.join(args.results_dir, file))
        
        if not study_files:
            print(f"No study files found in {args.results_dir}")
            return
        
        # Use the most recent study file
        study_files.sort(key=os.path.getmtime, reverse=True)
        args.study_file = study_files[0]
        print(f"Using most recent study file: {args.study_file}")
    
    # Load the study
    study = load_study(args.study_file)
    if study is None:
        return
    
    # Print study summary
    print_study_summary(study)
    
    # Create analysis plots
    output_dir = os.path.join(args.results_dir, 'analysis_plots')
    create_analysis_plots(study, output_dir)

if __name__ == "__main__":
    main()
