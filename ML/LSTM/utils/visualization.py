import matplotlib
matplotlib.use('Agg', force=True)  # Non-interactive backend, more stable for saving plots - force=True prevents other modules from overriding this
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Visualization:
    def __init__(self, data=None):
        self.data = data
        self.base_path = base_path or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    

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
    
    @staticmethod
    def plot_sequential_comparison(y_true, y_pred, n_points=2000, save_path=None):
        """
        Plot a comparison of predicted vs actual values for a large number of sequential observations.
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            n_points: Number of points to plot (default: 2000, uses last n_points)
            save_path: Optional path to save the figure
        """
        plt.figure(figsize=(20, 12))
        
        # Use the last n_points of data for visualization
        if len(y_true) > n_points:
            y_true_plot = y_true[-n_points:]
            y_pred_plot = y_pred[-n_points:]
        else:
            y_true_plot = y_true
            y_pred_plot = y_pred
        
        plt.plot(y_true_plot, label='Actual', linestyle='-')
        plt.plot(y_pred_plot, label='Predicted', linestyle='--')
        
        plt.title('Predicted vs Actual Flow')
        plt.xlabel('Number of observations')
        plt.ylabel('Flow')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            # Add n_points information to the filename if not already there
            dirname = os.path.dirname(save_path)
            basename = os.path.basename(save_path)
            
            # Check if n_points information is already in the filename
            if f"points{n_points}" not in basename:
                # Insert n_points info before the extension
                name, ext = os.path.splitext(basename)
                new_basename = f"{name}_points{n_points}{ext}"
                save_path = os.path.join(dirname, new_basename)
            
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            try:
                plt.show()
            except AttributeError as e:
                print(f"Warning: Could not display plot interactively: {e}")
                default_path = "plots/sequential_comparison.png"
                os.makedirs(os.path.dirname(default_path), exist_ok=True)
                plt.savefig(default_path)
                print(f"Plot saved to {default_path} instead of displaying interactively")
        
        plt.close()
    
    @staticmethod
    def plot_daily_comparison(y_true, y_pred, periods_per_day=96, segment_index=0, tick_interval=4, save_path=None, title=None):
        """
        Plot a day-specific comparison of predicted vs actual values.
        
        Args:
            y_true: Array of true values
            y_pred: Array of predicted values
            periods_per_day: Number of time periods in a day (default: 96 for 15-min intervals)
            segment_index: Which segment/day to plot (0 for first day, 1 for second day, etc.)
            tick_interval: Show x-axis ticks every n intervals (default: 4, meaning hourly ticks for 15-min data)
            save_path: Optional path to save the figure
            title: Custom title for the plot (optional)
        """
        # Calculate the indices for the selected segment
        start_idx = segment_index * periods_per_day
        end_idx = start_idx + periods_per_day
        
        # Check if we have enough data for this segment
        if end_idx > len(y_true):
            available_segments = len(y_true) // periods_per_day
            print(f"Warning: Not enough data for segment {segment_index}. Only {available_segments} segments available.")
            if available_segments == 0:
                raise ValueError("Not enough data to create a daily plot.")
            segment_index = available_segments - 1
            start_idx = segment_index * periods_per_day
            end_idx = min(start_idx + periods_per_day, len(y_true))
            print(f"Using segment {segment_index} instead.")
        
        # Generate time labels for the x-axis
        time_labels = pd.date_range(start='00:00', periods=periods_per_day, freq='15min').strftime('%I:%M %p')
        
        plt.figure(figsize=(18, 10))
        
        # Plot directly using time_labels on the x-axis
        plt.plot(time_labels, y_true[start_idx:end_idx], label='Actual', linestyle='-')
        plt.plot(time_labels, y_pred[start_idx:end_idx], label='Predicted', linestyle='--')
        
        # Set x-ticks at appropriate intervals
        plt.xticks(time_labels[::tick_interval], rotation=45)
        
        # Set title
        if title:
            plt.title(title)
        else:
            plt.title(f'Predicted vs Actual Flow (Segment {segment_index+1})')
            
        plt.xlabel('Time of Day')
        plt.ylabel('Flow')
        plt.legend()
        plt.tight_layout()
        plt.grid(True)
        
        if save_path:
            # Add segment information to the filename if not already there
            dirname = os.path.dirname(save_path)
            basename = os.path.basename(save_path)
            
            # Check if segment information is already in the filename
            if f"segment{segment_index+1}" not in basename:
                # Insert segment info before the extension
                name, ext = os.path.splitext(basename)
                new_basename = f"{name}_segment{segment_index+1}{ext}"
                save_path = os.path.join(dirname, new_basename)
            
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            try:
                plt.show()
            except AttributeError as e:
                print(f"Warning: Could not display plot interactively: {e}")
                default_path = f"plots/daily_comparison_segment{segment_index+1}.png"
                os.makedirs(os.path.dirname(default_path), exist_ok=True)
                plt.savefig(default_path)
                print(f"Plot saved to {default_path} instead of displaying interactively")
        
        plt.close()

# Add standalone functions for backward compatibility

def plot_training_history(train_losses, val_losses, lr_history=None, save_path=None):
    """Stand-alone function that calls the static method for backward compatibility"""
    return Visualization.plot_training_history(train_losses, val_losses, lr_history, save_path)

def plot_metrics(y_true, y_pred, save_path=None):
    """Stand-alone function that calls the static method for backward compatibility"""
    return Visualization.plot_metrics(y_true, y_pred, save_path)

def plot_accuracy_history(train_accuracies, val_accuracies=None, best_epoch=None, save_path=None):
    """Stand-alone function that calls the static method for backward compatibility"""
    return Visualization.plot_accuracy_history(train_accuracies, val_accuracies, best_epoch, save_path)

def plot_sequential_comparison(y_true, y_pred, n_points=2000, save_path=None):
    """Stand-alone function that calls the static method for backward compatibility"""
    return Visualization.plot_sequential_comparison(y_true, y_pred, n_points, save_path)

# Replace the plot_daily_comparison standalone function
def plot_daily_comparison(y_true, y_pred, segment_index=0, periods_per_day=96, tick_interval=8, save_path=None, title=None):
    """Stand-alone function that calls the static method for backward compatibility"""
    return Visualization.plot_daily_comparison(
        y_true=y_true,
        y_pred=y_pred,
        periods_per_day=periods_per_day,
        segment_index=segment_index,
        tick_interval=tick_interval,
        save_path=save_path,
        title=title
    )