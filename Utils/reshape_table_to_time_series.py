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

def reshape_traffic_data(input_file, output_file, start_date=None, end_date=None):
    """
    Reshapes the wide-format traffic data into a long-format time series dataset
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to output CSV file
    start_date (str, optional): Start date in 'YYYY-MM-DD' format to filter data
    end_date (str, optional): End date in 'YYYY-MM-DD' format to filter data
    """
    print(f"Reading data from {input_file}...")
    
    # Use chunking to avoid memory issues
    chunk_size = 300000  # Adjust this value based on your available memory
    chunks = []
    
    # Create an iterator for the chunks
    df_iterator = pd.read_csv(input_file, chunksize=chunk_size)
    
    # Initialize total rows counter
    total_rows = 0
    
    # Process each chunk
    for i, chunk in enumerate(df_iterator):
        print(f"Processing chunk {i+1}...")
        total_rows += len(chunk)
        
        # Filter by date range if specified
        if start_date and end_date:
            chunk['date'] = pd.to_datetime(chunk['date'])
            chunk = chunk[(chunk['date'] >= start_date) & (chunk['date'] <= end_date)]
            
            # Skip empty chunks
            if len(chunk) == 0:
                continue
                
            # Convert back to string format for consistency
            chunk['date'] = chunk['date'].dt.strftime('%Y-%m-%d')
        
        # Select the columns to keep in the final output
        id_vars = ['NB_SCATS_SITE', 'date', 'scat_type', 'day_type', 'school_count']
        
        # Get all V*_0 columns
        flow_columns = [col for col in chunk.columns if col.startswith('V') and col.endswith('_0')]
        
        # Melt the dataframe to convert from wide to long format
        long_chunk = pd.melt(
            chunk,
            id_vars=id_vars,
            value_vars=flow_columns,
            var_name='time_idx',
            value_name='Flow'
        )
        
        # Convert time_idx (e.g., V00_0, V01_0) to actual time
        long_chunk['time'] = long_chunk['time_idx'].apply(create_time_column)
        
        # Create a temporary datetime column for sorting purposes only
        long_chunk['datetime_temp'] = pd.to_datetime(long_chunk['date'] + ' ' + long_chunk['time'])
        
        # Sort by site and datetime
        long_chunk = long_chunk.sort_values(['NB_SCATS_SITE', 'datetime_temp'])
        
        # Drop the time_idx and temporary datetime columns
        long_chunk = long_chunk.drop(['time_idx', 'datetime_temp'], axis=1)
        
        # Reorder columns for final output
        result_chunk = long_chunk[['date', 'time', 'NB_SCATS_SITE', 'scat_type', 'day_type', 'school_count', 'Flow']]
        
        chunks.append(result_chunk)
        
        # Free memory
        del long_chunk
        del result_chunk
    
    print(f"Combining processed chunks...")
    # Combine all chunks
    result = pd.concat(chunks, ignore_index=True)
    
    print(f"Writing reshaped data to {output_file}...")
    # Write the result to a CSV file
    result.to_csv(output_file, index=False)
    
    print(f"Conversion complete! Reshaped data saved to {output_file}")
    print(f"Total rows processed: {total_rows}, Rows in result: {len(result)}")
    
def process_data_in_date_ranges(input_file, output_prefix, date_ranges):
    """
    Process data in multiple date ranges and save to separate files
    
    Parameters:
    input_file (str): Path to input CSV file
    output_prefix (str): Prefix for output files (will be appended with date range)
    date_ranges (list): List of tuples with (start_date, end_date, output_suffix)
    """
    for start_date, end_date, suffix in date_ranges:
        print(f"\nProcessing data from {start_date} to {end_date}...")
        output_file = f"{output_prefix}_{suffix}.csv"
        reshape_traffic_data(input_file, output_file, start_date, end_date)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Reshape traffic data from wide to long format')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file (wide format)')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file (long format)')
    parser.add_argument('--split', action='store_true', help='Split processing by date ranges')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    if args.split:
        # Define date ranges for processing
        date_ranges = [
            ('2014-01-01', '2019-12-31', '2014_2019'),
            ('2020-01-01', '2024-12-31', '2020_2024')
        ]
        
        # Strip file extension for the output prefix
        output_prefix = os.path.splitext(args.output)[0]
        
        process_data_in_date_ranges(args.input, output_prefix, date_ranges)
    elif args.start_date and args.end_date:
        # Process a specific date range
        reshape_traffic_data(args.input, args.output, args.start_date, args.end_date)
    else:
        # Process the entire dataset
        reshape_traffic_data(args.input, args.output)