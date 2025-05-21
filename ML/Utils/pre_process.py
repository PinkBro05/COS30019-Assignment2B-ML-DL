import os
import pandas as pd
import argparse

# Import from the modular components
from traffic_data_transformation import TrafficDataTransformer
from feature_engineering import feature_engineering
from time_series_reshaping import reshape_traffic_data, process_data_in_date_ranges


def preprocess_traffic_data(data_path=None, years=None, split=False, start_date=None, end_date=None):
    """
    Complete preprocessing pipeline for traffic data.
    
    Args:
        data_path: Path to the Raw data directory
        years: List of years to process
        split: Whether to split the output into date ranges
        start_date: Start date for filtering (YYYY-MM-DD)
        end_date: End date for filtering (YYYY-MM-DD)
    """
    # Determine the base directory
    if data_path:
        # Check if data_path already contains 'Raw/main'
        if "Raw" in data_path and "main" in data_path:
            # data_path is already pointing to 'Raw/main', so base_dir is 2 levels up
            base_dir = os.path.dirname(os.path.dirname(data_path))
        else:
            # For backward compatibility
            base_dir = os.path.dirname(os.path.dirname(data_path))
    else:
        # Getting the directory of the current file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # If we're in the Utils directory, go one level up for the project root
        if os.path.basename(base_dir) == "Utils":
            base_dir = os.path.dirname(base_dir)
            
        # Now construct data_path
        data_path = os.path.join(base_dir, "Data", "Raw", "main")
    
    # Check if base_dir contains 'Data'
    if os.path.basename(base_dir) == "Data":
        transformed_dir = os.path.join(base_dir, "Transformed")
    else:
        transformed_dir = os.path.join(base_dir, "Data", "Transformed")
    
    print(f"Using base directory: {base_dir}")
    print(f"Using data path: {data_path}")
    print(f"Using transformed directory: {transformed_dir}")
    
    # Step 1: Transform raw traffic data
    print("\n=== Step 1: Transforming raw traffic data ===\n")
    transformer = TrafficDataTransformer(data_path, years=years)
    
    # First create a sample of the transformed data
    print("Creating sample data...")
    sample_data = transformer.create_sample()
    
    # Process all years and save each to a separate CSV file
    processed_years = transformer.process_all_years()
    
    if not processed_years:
        print("Error: No data was transformed successfully.")
        return
    
    # Step 2: Feature Engineering
    print("\n=== Step 2: Applying feature engineering ===\n")
    featured_files = feature_engineering(base_dir)
    
    if not featured_files:
        print("Error: Feature engineering did not produce any output files.")
        return
    
    # Step 3: Combine all years into one file if needed
    print("\n=== Step 3: Combining years and reshaping to time series ===\n")
    
    # Combine the files if we have multiple
    if len(featured_files) > 1:
        print(f"Combining {len(featured_files)} featured files...")
        combined_dfs = []
        for file in featured_files:
            df = pd.read_csv(file)
            combined_dfs.append(df)
        
        all_years_df = pd.concat(combined_dfs, ignore_index=True)
        combined_file = os.path.join(transformed_dir, "combined_featured_data.csv")
        all_years_df.to_csv(combined_file, index=False)
        print(f"Saved combined data to {combined_file}")
        input_file = combined_file
    else:
        # Use the single file directly
        input_file = featured_files[0]
    
    # Step 4: Reshape to time series format
    output_prefix = os.path.join(transformed_dir, "final_time_series")
    
    if split:
        # Define date ranges for processing
        date_ranges = [
            ('2014-01-01', '2019-12-31', 'final_time_series_2014_2019'),
            ('2020-01-01', '2024-12-31', 'final_time_series_2020_2024')
        ]
        
        final_files = process_data_in_date_ranges(input_file, output_prefix, date_ranges)
        print("\n=== Preprocessing complete! ===\n")
        print(f"Final output files: {final_files}")
    else:
        output_file = f"{output_prefix}.csv"
        final_file = reshape_traffic_data(input_file, output_file, start_date, end_date)
        print("\n=== Preprocessing complete! ===\n")
        print(f"Final output file: {final_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess traffic data from raw files to time series format')
    
    parser.add_argument('--data-path', type=str, help='Path to the Raw/main directory containing year folders')
    parser.add_argument('--years', type=str, nargs='+', help='Years to process (e.g., 2014 2015 2016)')
    parser.add_argument('--split', action='store_true', help='Split output into date ranges (2014-2019 and 2020-2024)')
    parser.add_argument('--start-date', type=str, help='Start date for filtering (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for filtering (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    preprocess_traffic_data(
        data_path=args.data_path,
        years=args.years,
        split=args.split,
        start_date=args.start_date,
        end_date=args.end_date
    )