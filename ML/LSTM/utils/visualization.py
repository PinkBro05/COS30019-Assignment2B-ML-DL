import matplotlib
matplotlib.use('TkAgg')  # TkAgg works well with PyCharm
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Visualization:
    def __init__(self, data=None):
        self.data = data

    @staticmethod
    def plot_predictions(y_test, y_pred, index=0, save_path=None):
        """Plot actual vs predicted values"""
        true_vals = y_test[index]
        pred_vals = y_pred[index]
        
        time_labels = pd.date_range(start='00:00', periods=96, freq='15min').strftime('%H:%M')
        time_floats = np.linspace(0, 24, len(time_labels))
        
        plt.figure(figsize=(16, 7))
        
        for row in y_test:
            plt.scatter(time_floats, row, color='gray', alpha=0.2, s=10)
        
        plt.plot(time_floats, true_vals, color='blue', label='Actual values')
        plt.plot(time_floats, pred_vals, 'k^', label='Predictions')
        
        plt.title('Traffic flow data based model')
        plt.xlabel('Time of day')
        plt.ylabel('Travel time (second)')
        
        xtick_positions = np.linspace(0, 24, 25)
        xtick_labels = [f"{int(h):02d}:00" for h in xtick_positions]
        plt.xticks(xtick_positions, xtick_labels, rotation=45)
        
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Handle potential backends incompatibility
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path)
        else:
            try:
                plt.show()
            except AttributeError as e:
                print(f"Warning: Could not display plot interactively: {e}")
                default_path = "plots/prediction_plot.png"
                os.makedirs(os.path.dirname(default_path), exist_ok=True)
                plt.savefig(default_path)
                print(f"Plot saved to {default_path} instead of displaying interactively")
        
        plt.close()
    
    @staticmethod
    def plot_training_history(train_losses, val_losses, lr_history=None, save_path=None):
        """Plot training and validation losses and learning rate (optional)"""
        fig, axes = plt.subplots(1 + (1 if lr_history else 0), 1, figsize=(12, 8), sharex=True)
        
        # If lr_history is provided, we'll have 2 subplots
        if lr_history:
            loss_ax = axes[0]
            lr_ax = axes[1]
        else:
            loss_ax = axes if isinstance(axes, plt.Axes) else axes[0]
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot losses
        loss_ax.plot(epochs, train_losses, 'b-', label='Training Loss')
        loss_ax.plot(epochs, val_losses, 'r-', label='Validation Loss')
        loss_ax.set_title('Training and Validation Loss')
        loss_ax.set_ylabel('Loss (MSE)')
        loss_ax.legend()
        loss_ax.grid(True)
        
        # Plot learning rate if provided
        if lr_history:
            lr_ax.plot(epochs, lr_history, 'g-')
            lr_ax.set_title('Learning Rate')
            lr_ax.set_xlabel('Epochs')
            lr_ax.set_ylabel('Learning Rate')
            lr_ax.grid(True)
        else:
            loss_ax.set_xlabel('Epochs')
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path)
        else:
            try:
                plt.show()
            except AttributeError as e:
                print(f"Warning: Could not display plot interactively: {e}")
                default_path = "plots/training_history.png"
                os.makedirs(os.path.dirname(default_path), exist_ok=True)
                plt.savefig(default_path)
                print(f"Plot saved to {default_path} instead of displaying interactively")
        
        plt.close()
    
    @staticmethod
    def plot_metrics(y_true, y_pred, save_path=None):
        """Plot various performance metrics for regression model performance"""
        # Calculate metrics
        mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
        r2 = r2_score(y_true.flatten(), y_pred.flatten())
        
        # Plot metrics
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['MSE', 'RMSE', 'MAE', '1-R²']
        values = [mse, rmse, mae, 1-r2]  # Using 1-R² so smaller is better for all metrics
        
        bars = ax.bar(metrics, values, color=['#3274A1', '#E1812C', '#3A923A', '#C03D3E'])
        
        # Add values on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom')
        
        ax.set_title('Model Performance Metrics')
        ax.set_ylabel('Value (lower is better)')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add legend with actual R² value
        ax.text(0.02, 0.95, f'R² Score: {r2:.4f}', transform=ax.transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path)
        else:
            try:
                plt.show()
            except AttributeError as e:
                print(f"Warning: Could not display plot interactively: {e}")
                default_path = "plots/performance_metrics.png"
                os.makedirs(os.path.dirname(default_path), exist_ok=True)
                plt.savefig(default_path)
                print(f"Plot saved to {default_path} instead of displaying interactively")
        
        plt.close()
        
        return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}
    
    @staticmethod
    def plot_accuracy_history(train_accuracies, val_accuracies=None, best_epoch=None, save_path=None):
        """Plot training and validation accuracy (R²) with optional best epoch indicator"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_accuracies) + 1)
        
        plt.plot(epochs, train_accuracies, 'g-', label='Training R²')
        
        if val_accuracies is not None:
            plt.plot(epochs, val_accuracies, 'b-', label='Validation R²')
        
        if best_epoch is not None:
            plt.axvline(x=best_epoch+1, color='r', linestyle='--', 
                       label=f'Best epoch ({best_epoch+1})')
        
        plt.title('Model Accuracy (R²)')
        plt.xlabel('Epochs')
        plt.ylabel('R²')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path)
        else:
            try:
                plt.show()
            except AttributeError as e:
                print(f"Warning: Could not display plot interactively: {e}")
                default_path = "plots/accuracy_history.png"
                os.makedirs(os.path.dirname(default_path), exist_ok=True)
                plt.savefig(default_path)
                print(f"Plot saved to {default_path} instead of displaying interactively")
        
        plt.close()

# Add standalone functions for backward compatibility
def plot_predictions(y_test, y_pred, index=0, save_path=None):
    """Stand-alone function that calls the static method for backward compatibility"""
    return Visualization.plot_predictions(y_test, y_pred, index, save_path)

def plot_training_history(train_losses, val_losses, lr_history=None, save_path=None):
    """Stand-alone function that calls the static method for backward compatibility"""
    return Visualization.plot_training_history(train_losses, val_losses, lr_history, save_path)

def plot_metrics(y_true, y_pred, save_path=None):
    """Stand-alone function that calls the static method for backward compatibility"""
    return Visualization.plot_metrics(y_true, y_pred, save_path)

def plot_accuracy_history(train_accuracies, val_accuracies=None, best_epoch=None, save_path=None):
    """Stand-alone function that calls the static method for backward compatibility"""
    return Visualization.plot_accuracy_history(train_accuracies, val_accuracies, best_epoch, save_path)