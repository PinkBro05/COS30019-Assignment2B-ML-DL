import pandas as pd
import os
from typing import Dict, List, Optional, Tuple, Union

def find_column_by_pattern(df: pd.DataFrame, patterns: List[str]) -> Optional[str]:
    """
    Find a column name in a DataFrame that matches any of the given patterns
    
    Args:
        df: DataFrame to search in
        patterns: List of patterns to match against column names
        
    Returns:
        Column name that matches any pattern, or None if no match found
    """
    for col in df.columns:
        for pattern in patterns:
            if pattern.lower() in col.lower():
                return col
    return None

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file with error handling
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path, low_memory=False)

def process_scats_data(
    input_path: str, 
    site_type_path: str, 
    output_path: str
) -> pd.DataFrame:
    """
    Complete SCATS data processing pipeline:
    1. Load and prepare data
    2. Combine directional flows
    3. Map site types
    4. Remove unnecessary columns
    
    Args:
        input_path: Path to the input transformed SCATS data
        site_type_path: Path to the site type mapping data
        output_path: Path to save the final processed data
        
    Returns:
        Processed DataFrame
    """
    print(f"Loading and processing SCATS data from {input_path}")
    
    # Load the data
    scats_df = load_data(input_path)
    print(f"Loaded {len(scats_df)} records with {len(scats_df.columns)} columns")
    
    # Identify important columns
    scats_col = 'SCATS Number' if 'SCATS Number' in scats_df.columns else 'SCATS_Number'
    melway_col = find_column_by_pattern(scats_df, ['CD_MELWAY', 'melway'])
    
    if not melway_col:
        print("Warning: Could not identify MELWAY column. Site type mapping may fail.")
    
    # Preprocessing: Handle dates and ensure correct types
    scats_df['Date'] = pd.to_datetime(scats_df['Date'], errors='coerce')
    scats_df = scats_df.dropna(subset=['Date'])
    scats_df[scats_col] = scats_df[scats_col].astype(str)
    
    # Special case handling for SCATS number 4335 with HF VicRoads Internal = 15772
    if 'HF VicRoads Internal' in scats_df.columns:
        mask = (scats_df[scats_col] == '4335') & (scats_df['HF VicRoads Internal'] == 15772)
        if mask.any():
            print(f"Applying special case handling for SCATS number 4335 with HF VicRoads Internal = 15772")
            print(f"Found {mask.sum()} matching records")
            # Divide the flow by 2 and round to integer
            scats_df.loc[mask, 'Flow'] = (scats_df.loc[mask, 'Flow'] / 2).round().astype(int)
    
    # Step 1: Combine directional flows
    print("Combining directional flows...")
    agg_dict = {
        'Flow': 'sum',
        'Location': lambda x: ', '.join(sorted(set(x)))
    }
    
    # Preserve MELWAY column if it exists
    if melway_col:
        agg_dict[melway_col] = 'first'
    
    combined_df = scats_df.groupby(['Date', scats_col]).agg(agg_dict).reset_index()
    combined_df = combined_df.sort_values(by=[scats_col, 'Date'])
    print(f"Data combined into {len(combined_df)} records")
    
    # Step 2: Map site types
    print("Mapping site types...")
    site_df = load_data(site_type_path)
    
    # Find Site_Number column in the site type data
    site_num_col = find_column_by_pattern(site_df, ['Site_Number'])
    if not site_num_col:
        print("Warning: Could not find Site_Number column in site type data.")
        print(f"Available columns: {site_df.columns.tolist()}")
        final_df = combined_df
    else:
        # Create mapping based on Site_Number/SCATS_Number and apply
        mapping_dict = dict(zip(site_df[site_num_col].astype(str), site_df['Site_Type']))
        print(f"Created mapping for {len(mapping_dict)} sites based on Site_Number.")
        
        # Apply mapping - use the SCATS_Number directly
        combined_df['Site_Type'] = combined_df[scats_col].map(mapping_dict)
        
        mapped_count = combined_df['Site_Type'].notna().sum()
        print(f"Mapped {mapped_count} out of {len(combined_df)} records ({mapped_count/len(combined_df):.2%})")
        
        # Step 3: Remove unnecessary columns
        cols_to_remove = ['VR Internal Stat', 'HF VicRoads Internal', melway_col, 'Location']
        keep_cols = [col for col in combined_df.columns if col not in cols_to_remove]
        final_df = combined_df[keep_cols]
        
        # If mapping was unsuccessful, log a warning
        if mapped_count == 0:
            print("WARNING: No records were successfully mapped to a Site_Type.")
            print(f"SCATS_Number sample values: {combined_df[scats_col].head(5).tolist()}")
            print(f"Site_Number sample values: {site_df[site_num_col].head(5).tolist()}")
    
    # Save final result
    final_df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    print(f"Final dataset has {len(final_df)} rows and {len(final_df.columns)} columns")
    
    return final_df

def main():
    """Main entry point for processing SCATS data"""
    # Define paths with relative path handling
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transformed_data_path = os.path.join(base_dir, 'Data', 'Transformed', 'transformed_scats_data.csv')
    site_type_path = os.path.join(base_dir, 'Data', 'Raw', 'scat_type.csv')
    final_data_path = os.path.join(base_dir, 'Data', 'Transformed', 'final_scats_data.csv')
    
    # Fallback to relative paths if absolute paths don't work
    if not os.path.exists(transformed_data_path):
        transformed_data_path = 'Data/Transformed/transformed_scats_data.csv'
        site_type_path = 'Data/Raw/scat_type.csv'
        final_data_path = 'Data/Transformed/final_scats_data.csv'
    
    try:
        final_df = process_scats_data(
            input_path=transformed_data_path,
            site_type_path=site_type_path,
            output_path=final_data_path
        )
        print("Data processing completed successfully.")
        return final_df
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return None

if __name__ == "__main__":
    main()