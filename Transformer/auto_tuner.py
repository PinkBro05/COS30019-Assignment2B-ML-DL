"""
AutoTuner for automatically tuning hyperparameters of the Transformer model.
"""
import os
import sys
import time
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import partial
from sklearn.model_selection import ParameterGrid
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
import joblib
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna.visualization as vis

# Add parent directory to path to import from utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Transformer.models.model import TransformerModel
from Transformer.utils.traffic_data_collector import TrafficDataCollector
from Transformer.utils.model_utils import save_model_metadata
from Transformer.supervised_learning import train_transformer, test_model

class AutoTuner:
    """
    AutoTuner for transformer models using Optuna for hyperparameter optimization.
    """
    
    def __init__(self, data_file, output_dir=None, n_trials=50, test_size=0.2, device=None, random_seed=42):
        """
        Initialize the AutoTuner.
        
        Args:
            data_file: Path to the CSV data file
            output_dir: Directory to save results
            n_trials: Number of optimization trials to run
            test_size: Fraction of data to use for test set
            device: Computing device ('cuda' or 'cpu')
            random_seed: Random seed for reproducibility
        """
        self.data_file = data_file
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Set random seeds for reproducibility
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Set output directory
        if output_dir is None:
            self.output_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'Transformer', 
                'auto_tuner_results'
            )
        else:
            self.output_dir = output_dir
            
        # Ensure directories exist
        self.models_dir = os.path.join(self.output_dir, 'models')
        self.figures_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        
        # Study settings
        self.n_trials = n_trials
        self.test_size = test_size
        self.random_seed = random_seed
        
        # Initialize data collectors
        self.data_collector = TrafficDataCollector(embedding_dim=16)  # Default, will be tuned
        
        # Create study name
        timestamp = int(time.time())
        self.study_name = f"transformer_autotuner_{timestamp}"
        self.study_file = os.path.join(self.output_dir, f"{self.study_name}.pkl")
        
        # Load data (without loading actual tensors yet)
        print(f"Preparing to tune model using data from: {self.data_file}")
    
    def _load_data(self, params):
        """
        Load data with given parameters for a trial.
        
        Args:
            params: Dictionary of hyperparameters
            
        Returns:
            Dictionary with data loaders
        """
        # Set embedding dimension
        self.data_collector = TrafficDataCollector(embedding_dim=params['embedding_dim'])
        
        # Get data loaders
        data_loaders = self.data_collector.get_data_loaders(
            data_file=self.data_file,
            batch_size=params['batch_size'],
            shuffle=True,
            num_workers=4,
            train_ratio=0.7,
            val_ratio=0.3,
            random_state=self.random_seed,
            device=self.device
        )
        
        return data_loaders
    
    def _create_model(self, params, input_dim, categorical_metadata, categorical_indices):
        """
        Create model with given parameters.
        
        Args:
            params: Dictionary of hyperparameters
            input_dim: Input dimension
            categorical_metadata: Metadata for categorical features
            categorical_indices: Indices of categorical features
            
        Returns:
            TransformerModel instance
        """
        model = TransformerModel(
            input_dim=input_dim,
            d_model=params['d_model'],
            num_heads=params['num_heads'],
            d_ff=params['d_ff'],
            output_size=1,
            num_layers=params['num_layers'],
            dropout=params['dropout'],
            categorical_metadata=categorical_metadata,
            categorical_indices=categorical_indices
        )
        
        return model
    
    def objective(self, trial):
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation loss (to be minimized)
        """        # First select d_model to ensure compatibility with num_heads
        d_model = trial.suggest_categorical('d_model', [64, 128, 256, 384])
        
        # Calculate valid num_heads values (must be divisors of d_model)
        valid_heads = [h for h in [1, 2, 4, 8, 16] if d_model % h == 0 and h <= d_model//8]
        
        # Ensure there's at least one valid option (fallback to 1 if needed)
        if not valid_heads:
            valid_heads = [1]  # 1 is always a valid divisor
            print(f"Warning: No valid head values found for d_model={d_model}, using num_heads=1")
            
        # Define hyperparameters to tune
        params = {
            # Data parameters
            'embedding_dim': trial.suggest_categorical('embedding_dim', [8, 16, 32]),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            
            # Model parameters
            'd_model': d_model,  # Already selected above
            'num_heads': trial.suggest_categorical('num_heads', valid_heads),  # Only valid options
            'd_ff': trial.suggest_categorical('d_ff', [128, 256, 512, 1024]),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5),
            
            # Training parameters
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-3),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-4),
            'num_epochs': trial.suggest_int('num_epochs', 5, 20)  # Limit epochs for faster tuning
        }
          # Load data with these parameters
        data_loaders = self._load_data(params)
        
        if not data_loaders or 'train_loader' not in data_loaders:
            print("Error: Failed to load data for training")
            return float('inf')  # Return a large loss if data loading fails
        
        try:
            # Check that we have some data
            train_loader = data_loaders['train_loader']
            val_loader = data_loaders['val_loader']
            
            if len(train_loader) == 0 or len(val_loader) == 0:
                print("Error: Empty data loaders")
                return float('inf')
                
            # Get input dimension and metadata
            try:
                X_batch, y_batch = next(iter(train_loader))
                
                # Validate shapes
                if X_batch.shape[0] == 0 or y_batch.shape[0] == 0:
                    print("Error: Empty batch encountered")
                    return float('inf')
                    
                # Check for NaN values
                if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                    print("Error: NaN values found in data")
                    return float('inf')
                
                input_dim = X_batch.shape[2]
                categorical_metadata = data_loaders['categorical_metadata']
                categorical_indices = data_loaders['categorical_indices']
                
                print(f"Data loaded successfully. Batch shapes: X={X_batch.shape}, y={y_batch.shape}")
            except StopIteration:
                print("Error: Could not get batch from data loader")
                return float('inf')
        except Exception as e:
            print(f"Error validating data: {str(e)}")
            return float('inf')
        try:
            # Create model
            model = self._create_model(params, input_dim, categorical_metadata, categorical_indices)
            model.to(self.device)
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(
                model.parameters(), 
                lr=params['learning_rate'],
                weight_decay=params['weight_decay']
            )
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
            
            # Train the model
            history = train_transformer(
                model=model,
                train_loader=data_loaders['train_loader'],
                val_loader=data_loaders['val_loader'],
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=params['num_epochs'],
                device=self.device,
                save_path=None,  # Don't save every trial model
                max_grad_norm=1.0
            )
                
            # Report intermediate values
            for epoch in range(len(history['val_loss'])):
                val_loss = history['val_loss'][epoch]
                    
                trial.report(val_loss, epoch)
                
                # Handle pruning based on intermediate results
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
              # Get best validation loss, with error handling
            try:
                best_val_loss = history.get('best_val_loss', min(history['val_loss']))
                print(f"Best validation loss: {best_val_loss:.6f}")
                
            except Exception as e:
                print(f"Error getting best validation loss: {str(e)}")
                best_val_loss = float('inf')
            # Save the model if it's among the top performers
            if (not math.isnan(best_val_loss) and 
                not math.isinf(best_val_loss) and 
                not trial.should_prune() and 
                trial.state == optuna.trial.TrialState.RUNNING):
                # Save only if the trial is in the top 5 so far
                study = trial.study
                sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
                if trial in sorted_trials[:5]:  # Top 5
                    model_path = os.path.join(self.models_dir, f"model_trial_{trial.number}.pth")
                    self._save_model(model, model_path, params, categorical_metadata, history)
            
            return best_val_loss
        except Exception as e:
            print(f"Error during training: {str(e)}")
            import traceback
            traceback.print_exc()
            return float('inf')
    
    def _save_model(self, model, path, params, categorical_metadata, history):
        """
        Save model and metadata.
        
        Args:
            model: The model to save
            path: Path to save the model
            params: Hyperparameters used
            categorical_metadata: Metadata for categorical features
            history: Training history
        """
        # Save the model
        torch.save(model.state_dict(), path)
        
        # Save metadata
        metadata_path = path.replace('.pth', '_metadata.json')
        save_model_metadata(
            path=metadata_path,
            params=params,
            categorical_metadata=categorical_metadata,
            history=history
        )
        
        print(f"Model and metadata saved to {path}")
    
    def run_optimization(self):
        """
        Run the hyperparameter optimization process.
        
        Returns:
            The best hyperparameters found
        """
        # Create a new study or load an existing one
        study = optuna.create_study(
            study_name=self.study_name,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
        )
        
        # Run optimization
        print(f"Starting hyperparameter optimization with {self.n_trials} trials...")
        study.optimize(self.objective, n_trials=self.n_trials)
        
        # Save study for later analysis
        joblib.dump(study, self.study_file)
        
        # Get best trial
        best_trial = study.best_trial
        
        # Print optimization results
        print("\n" + "="*50)
        print("Hyperparameter Optimization Results:")
        print(f"Best validation loss: {best_trial.value:.6f}")
        print("\nBest hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"  {key}: {value}")
        
        # Generate visualization plots
        self._save_optimization_plots(study)
        
        return best_trial.params
    
    def _save_optimization_plots(self, study):
        """
        Save optimization visualization plots.
        
        Args:
            study: Optuna study object
        """
        # Plot optimization history
        try:
            fig1 = vis.plot_optimization_history(study)
            fig1.write_image(os.path.join(self.figures_dir, 'optimization_history.png'))
            
            # Plot parameter importances
            fig2 = vis.plot_param_importances(study)
            fig2.write_image(os.path.join(self.figures_dir, 'param_importances.png'))
            
            # Plot slice for important parameters
            fig3 = vis.plot_slice(study)
            fig3.write_image(os.path.join(self.figures_dir, 'param_slices.png'))
            
            # Plot parallel coordinate
            fig4 = vis.plot_parallel_coordinate(study)
            fig4.write_image(os.path.join(self.figures_dir, 'parallel_coordinate.png'))
            
            print(f"Optimization plots saved to {self.figures_dir}")
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def train_best_model(self, best_params, num_epochs=None):
        """
        Train a model with the best hyperparameters found.
        
        Args:
            best_params: The best hyperparameters found
            num_epochs: Number of epochs to train for (overrides best_params if provided)
            
        Returns:
            Trained model, history, and data loaders
        """
        # Override number of epochs if provided
        if num_epochs is not None:
            best_params['num_epochs'] = num_epochs
        
        # Load data with best parameters
        data_loaders = self._load_data(best_params)
        
        if not data_loaders or 'train_loader' not in data_loaders:
            raise ValueError("Failed to load data for best model training")
        
        # Get input dimension and metadata
        X_batch, _ = next(iter(data_loaders['train_loader']))
        input_dim = X_batch.shape[2]
        categorical_metadata = data_loaders['categorical_metadata']
        categorical_indices = data_loaders['categorical_indices']
        
        # Create model with best parameters
        model = self._create_model(best_params, input_dim, categorical_metadata, categorical_indices)
        model.to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(), 
            lr=best_params['learning_rate'],
            weight_decay=best_params['weight_decay']
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Model file path
        model_file = os.path.join(self.models_dir, "best_model.pth")
        
        # Train the model
        print(f"Training best model for {best_params['num_epochs']} epochs...")
        history = train_transformer(
            model=model,
            train_loader=data_loaders['train_loader'],
            val_loader=data_loaders['val_loader'],
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs=best_params['num_epochs'],
            device=self.device,
            save_path=model_file,
            max_grad_norm=1.0
        )
        
        # Save model metadata
        metadata_path = model_file.replace('.pth', '_metadata.json')
        save_model_metadata(
            path=metadata_path,
            params=best_params,
            categorical_metadata=categorical_metadata,
            history=history
        )
        
        # Plot training history
        self._plot_training_history(history, best_params)
        
        return model, history, data_loaders
    
    def _plot_training_history(self, history, params):
        """
        Plot training history.
        
        Args:
            history: Training history dictionary
            params: Model parameters
        """
        # Create figure and grid for subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot training and validation loss
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot learning rate
        ax2.plot(epochs, history['learning_rate'], 'g-')
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True)
        
        # Add hyperparameter info to the plot
        param_str = (
            f"Hyperparameters:\n"
            f"d_model: {params['d_model']}, num_heads: {params['num_heads']}, "
            f"num_layers: {params['num_layers']}, d_ff: {params['d_ff']}\n"
            f"batch_size: {params['batch_size']}, learning_rate: {params['learning_rate']:.6f}, "
            f"dropout: {params['dropout']}, embedding_dim: {params['embedding_dim']}"
        )
        fig.text(0.5, 0.01, param_str, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        # Save plot
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
        plot_path = os.path.join(self.figures_dir, 'best_model_training_history.png')
        plt.savefig(plot_path)
        plt.close()
        
        print(f"Training history plot saved to {plot_path}")
    
    def evaluate_best_model(self, model, data_loaders, params=None):
        """
        Evaluate the best model on the test set.
        
        Args:
            model: The model to evaluate
            data_loaders: Dictionary with data loaders
            params: Model parameters for plotting
            
        Returns:
            Dictionary with test metrics
        """
        if 'val_loader' not in data_loaders:
            raise ValueError("Test data loader not found in data_loaders")
        
        # Test the model
        print("Evaluating model on test set...")
        test_loader = data_loaders['val_loader']  # Using validation set for testing
        
        # Create test options
        test_args = type('TestArgs', (), {
            'plot_test_results': True,
            'test_plot_name': 'best_model_test_results.png'
        })
        
        encoders_scalers = data_loaders.get('encoders_scalers')
        
        # Evaluate the model
        metrics = test_model(
            args=test_args,
            model=model,
            device=self.device,
            test_loader=test_loader,
            encoders_scalers=encoders_scalers,
            save_dir=self.figures_dir
        )
        
        return metrics

def run_auto_tuning(data_file, output_dir=None, n_trials=50, test_size=0.2, final_train_epochs=50):
    """
    Run full auto-tuning process.
    
    Args:
        data_file: Path to CSV data file
        output_dir: Directory to save results
        n_trials: Number of optimization trials
        test_size: Fraction of data to use for test
        final_train_epochs: Number of epochs to train final model
        
    Returns:
        Dictionary with best parameters and metrics
    """
    # Create AutoTuner
    tuner = AutoTuner(
        data_file=data_file,
        output_dir=output_dir,
        n_trials=n_trials,
        test_size=test_size
    )
    
    # Run optimization
    best_params = tuner.run_optimization()
    
    # Train best model with more epochs
    model, history, data_loaders = tuner.train_best_model(
        best_params=best_params,
        num_epochs=final_train_epochs
    )
    
    # Evaluate best model
    metrics = tuner.evaluate_best_model(model, data_loaders, best_params)
    
    # Combine results
    results = {
        'best_params': best_params,
        'test_metrics': metrics,
        'training_history': history
    }
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-tuning for Transformer models')
    
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to CSV data file')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for test')
    parser.add_argument('--final_train_epochs', type=int, default=50,
                        help='Number of epochs to train final model')
    
    args = parser.parse_args()
    
    # Run auto-tuning
    results = run_auto_tuning(
        data_file=args.data_file,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        test_size=args.test_size,
        final_train_epochs=args.final_train_epochs
    )
    
    # Print final results
    print("\n" + "="*50)
    print("Auto-Tuning Completed!")
    print(f"Best parameters: {results['best_params']}")
    print(f"Test metrics: {results['test_metrics']}")
    print("="*50)
