import pandas as pd
import os
import sys
from pathlib import Path
import re

def remove_duplicated_columns(file_path, output_path=None):
    """
    Removes duplicated columns with the pattern V00_X.Y from a CSV file.
    
    Args:
        file_path: Path to the input CSV file
        output_path: Path to save the cleaned file. If None, overwrites the original file.
    
    Returns:
        Path to the cleaned file
    """
    print(f"Reading file: {file_path}")
    
    # Read the CSV file
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"Error reading file: {e}")
        return None
    
    print(f"Original columns count: {len(df.columns)}")
    
    # Identify columns matching the pattern V00_X.Y
    pattern = re.compile(r'^V\d+_\d+\.\d+$')
    duplicate_columns = [col for col in df.columns if pattern.match(col)]
    
    print(f"Found {len(duplicate_columns)} columns matching the V00_X.Y pattern")
    
    # Remove the duplicated columns
    if duplicate_columns:
        df = df.drop(columns=duplicate_columns)
        print(f"Removed {len(duplicate_columns)} duplicated columns")
    else:
        print("No duplicated columns found matching the pattern")
    
    # Determine output path
    if output_path is None:
        output_path = file_path
    
    # Save the cleaned dataframe
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned file to: {output_path}")
    print(f"New columns count: {len(df.columns)}")
    
    return output_path

if __name__ == "__main__":
    # Get the base directory of the project
    base_dir = Path(__file__).parent.parent
    
    # Define path to the file
    file_path = base_dir / "Data" / "Transformed" / "2014-2024_final.csv"
    
    # Run the cleanup
    remove_duplicated_columns(file_path)