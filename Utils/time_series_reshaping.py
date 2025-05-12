import os
import pandas as pd
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

def create_time_column(time_idx):
    """Convert V00_0, V01_0, V00_0.0, V01_0.0, ... to actual time strings"""
    # Handle both _0 and _0.0 endings
    cleaned_idx = time_idx.replace('V', '')
    if '_0.0' in cleaned_idx:
        cleaned_idx = cleaned_idx.replace('_0.0', '')
    else:
        cleaned_idx = cleaned_idx.replace('_0', '')
    
    idx = int(cleaned_idx)
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
        # Get all V*_0 or V*_0.0 columns
        flow_columns = [col for col in chunk.columns if col.startswith('V') and (col.endswith('_0') or col.endswith('_0.0'))]
        
        if len(flow_columns) == 0:
            print(f"WARNING: No flow columns found in chunk {i+1}. Available columns: {chunk.columns.tolist()}")
            continue
            
        print(f"Chunk {i+1}: Found {len(flow_columns)} flow columns. First few: {flow_columns[:5]}...")
        print(f"Chunk {i+1}: Number of rows before melt: {len(chunk)}")
        
        # Verify that id_vars columns exist in the chunk
        missing_id_cols = [col for col in id_vars if col not in chunk.columns]
        if missing_id_cols:
            print(f"WARNING: Missing required ID columns in chunk {i+1}: {missing_id_cols}")
            print(f"Available columns: {chunk.columns.tolist()}")
            # Try to use only available columns
            id_vars = [col for col in id_vars if col in chunk.columns]
            if not id_vars:
                print(f"ERROR: No ID columns available, skipping chunk {i+1}")
                continue
                
        # Melt the dataframe to convert from wide to long format
        long_chunk = pd.melt(
            chunk,
            id_vars=id_vars,
            value_vars=flow_columns,
            var_name='time_idx',
            value_name='Flow'
        )
        
        print(f"Chunk {i+1}: Number of rows after melt: {len(long_chunk)}")
        # Check for any rows with NaN in Flow
        nan_count = long_chunk['Flow'].isna().sum()
        if nan_count > 0:
            print(f"WARNING: Found {nan_count} rows with NaN Flow values in chunk {i+1}")
            # Don't drop NaN values here, keep them for debugging
        
        # Check for negative flow values
        neg_count = (long_chunk['Flow'] < 0).sum()
        if neg_count > 0:
            print(f"WARNING: Found {neg_count} rows with negative Flow values in chunk {i+1}")
            # Don't filter here, keep them for debugging
        
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
    if not chunks:
        print("ERROR: No chunks were processed successfully, result will be empty!")
        result = pd.DataFrame()
    else:
        result = pd.concat(chunks, ignore_index=True)
        
        # Additional debug information
        print(f"Data shape before saving: {result.shape}")
        if len(result) > 0:
            print(f"Flow column stats: min={result['Flow'].min()}, max={result['Flow'].max()}, mean={result['Flow'].mean():.2f}")
            print(f"Number of sites: {result['NB_SCATS_SITE'].nunique()}")
            print(f"Number of dates: {result['date'].nunique()}")
            print(f"Sample of first few rows:")
            print(result.head(3))
        else:
            print("WARNING: Result dataframe is empty! Checking chunks for issues...")
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i+1} shape: {chunk.shape}")
    
    print(f"Writing reshaped data to {output_file}...")
    # Write the result to a CSV file
    result.to_csv(output_file, index=False)
    
    print(f"Conversion complete! Reshaped data saved to {output_file}")
    print(f"Total rows processed: {total_rows}, Rows in result: {len(result)}")
    return output_file

def process_data_in_date_ranges(input_file, output_prefix, date_ranges):
    """
    Process data in multiple date ranges and save to separate files
    
    Parameters:
    input_file (str): Path to input CSV file
    output_prefix (str): Prefix for output files (will be appended with date range)
    date_ranges (list): List of tuples with (start_date, end_date, output_suffix)
    """
    results = []
    # Parallelize date range processing
    with ThreadPoolExecutor(max_workers=min(len(date_ranges), os.cpu_count() or 1)) as executor:
        future_to_range = {}
        for start_date, end_date, suffix in date_ranges:
            print(f"\nScheduling processing for {start_date} to {end_date}...")
            output_file = f"{output_prefix}_{suffix}.csv"
            future = executor.submit(reshape_traffic_data, input_file, output_file, start_date, end_date)
            future_to_range[future] = (start_date, end_date, suffix)
        for future in as_completed(future_to_range):
            start_date, end_date, suffix = future_to_range[future]
            try:
                output_path = future.result()
                print(f"Completed processing for {start_date} to {end_date}: {output_path}")
                results.append(output_path)
            except Exception as e:
                print(f"Error processing range {start_date}-{end_date}: {e}")
    return results