"""
Script to run the AutoTuner for the Transformer model.
"""
import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Transformer.utils.auto_tuner import run_auto_tuning

def main():
    parser = argparse.ArgumentParser(description='Run AutoTuner for the Transformer model')
    
    # Data parameters
    parser.add_argument('--data_file', type=str, default=None,
                        help='Path to CSV data file. If not provided, will use sample file.')
    
    # Tuning parameters
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of optimization trials')
    parser.add_argument('--final_epochs', type=int, default=30,
                        help='Number of epochs to train the final model')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Fraction of data to use for testing')
    
    # Output parameters
    parser.add_argument('--output_name', type=str, default=None,
                        help='Name for the output directory. If not provided, will use a timestamp.')
    
    args = parser.parse_args()
    
    # Set default data file if not provided
    if args.data_file is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'Data', 'Transformed')
        args.data_file = os.path.join(data_dir, '_sample_final_time_series.csv')
        
        # Check if 2024_final_time_series.csv exists (better data)
        better_data = os.path.join(data_dir, '2024_final_time_series.csv')
        if os.path.exists(better_data):
            args.data_file = better_data
            print(f"Using 2024 data file: {args.data_file}")
        else:
            print(f"Using sample data file: {args.data_file}")
    
    # Set output directory
    output_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auto_tuner_results')
    os.makedirs(output_base, exist_ok=True)
    
    if args.output_name:
        output_dir = os.path.join(output_base, args.output_name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_base, f"run_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Starting AutoTuner with {args.n_trials} trials")
    print(f"Data file: {args.data_file}")
    print(f"Output directory: {output_dir}")
    print("Note: d_model will always be divisible by num_heads to avoid compatibility errors")
    
    # Run the auto-tuning
    results = run_auto_tuning(
        data_file=args.data_file,
        output_dir=output_dir,
        n_trials=args.n_trials,
        test_size=args.test_size,
        final_train_epochs=args.final_epochs
    )
    
    # Print final results summary
    print("\n" + "="*50)
    print("Auto-Tuning Complete!")
    print(f"Results saved to: {output_dir}")
    print("Best parameters:")
    for k, v in results['best_params'].items():
        print(f"  {k}: {v}")
    
    # Print test metrics if available
    if 'test_metrics' in results and results['test_metrics']:
        print("\nTest Metrics:")
        for k, v in results['test_metrics'].items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.6f}")
    
    print("="*50)

if __name__ == "__main__":
    main()
