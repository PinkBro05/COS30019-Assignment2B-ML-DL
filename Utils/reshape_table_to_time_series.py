import pandas as pd
import os
import argparse
from datetime import datetime, timedelta

def create_time_column(time_idx):
    """Convert V00_0, V01_0, ... to actual time strings"""
    idx = int(time_idx.replace('V', '').replace('_0', ''))
    hours = idx // 4
    minutes = (idx % 4) * 15
    return f"{hours:02d}:{minutes:02d}"

def reshape_traffic_data(input_file, output_file):
    """
    Reshapes the wide-format traffic data into a long-format time series dataset
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    """
    print(f"Reading data from {input_file}...")
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    print("Reshaping data...")
    # Select the columns to keep in the final output
    id_vars = ['NB_SCATS_SITE', 'date', 'scat_type', 'day_type', 'school_count']
    
    # Get all V*_0 columns
    flow_columns = [col for col in df.columns if col.startswith('V') and col.endswith('_0')]
    
    # Melt the dataframe to convert from wide to long format
    long_df = pd.melt(
        df,
        id_vars=id_vars,
        value_vars=flow_columns,
        var_name='time_idx',
        value_name='Flow'
    )
    
    # Convert time_idx (e.g., V00_0, V01_0) to actual time
    long_df['time'] = long_df['time_idx'].apply(create_time_column)
    
    # Create a temporary datetime column for sorting purposes only
    long_df['datetime_temp'] = pd.to_datetime(long_df['date'] + ' ' + long_df['time'])
    
    # Sort by site and datetime
    long_df = long_df.sort_values(['NB_SCATS_SITE', 'datetime_temp'])
    
    # Drop the time_idx and temporary datetime columns
    long_df = long_df.drop(['time_idx', 'datetime_temp'], axis=1)
    
    # Reorder columns for final output
    result = long_df[['date', 'time', 'NB_SCATS_SITE', 'scat_type', 'day_type', 'school_count', 'Flow']]
    
    print(f"Writing reshaped data to {output_file}...")
    # Write the result to a CSV file
    result.to_csv(output_file, index=False)
    
    print(f"Conversion complete! Reshaped data saved to {output_file}")
    print(f"Original shape: {df.shape}, New shape: {result.shape}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reshape traffic data from wide to long format')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file (wide format)')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file (long format)')
    
    args = parser.parse_args()
    
    reshape_traffic_data(args.input, args.output)