import pandas as pd
import os
from datetime import datetime
import glob
import numpy as np

def add_scat_type(transformed_df, scat_type_df, traffic_lights_df):
    """
    Maps scat_type from the scat_type.csv to the transformed dataset.
    If not found in scat_type.csv, tries to find it in Traffic_Lights.csv
    """
    # Check if NB_SCATS_SITE column exists
    if 'NB_SCATS_SITE' not in transformed_df.columns:
        print("WARNING: NB_SCATS_SITE column not found in dataset. Using 'Unknown' for scat_type.")
        transformed_df['scat_type'] = "Unknown"
        return transformed_df
    
    try:
        # Convert to same type (integer) if possible
        if pd.api.types.is_numeric_dtype(transformed_df['NB_SCATS_SITE']):
            scat_type_df['Site_Number'] = pd.to_numeric(scat_type_df['Site_Number'], errors='coerce')
            traffic_lights_df['SITE_NO'] = pd.to_numeric(traffic_lights_df['SITE_NO'], errors='coerce')
        else:
            # Convert both to strings if numeric conversion isn't appropriate
            transformed_df['NB_SCATS_SITE'] = transformed_df['NB_SCATS_SITE'].astype(str)
            scat_type_df['Site_Number'] = scat_type_df['Site_Number'].astype(str)
            traffic_lights_df['SITE_NO'] = traffic_lights_df['SITE_NO'].astype(str)
    except Exception as e:
        print(f"WARNING: Type conversion issue: {e}. Proceeding with original types.")
    
    # Create mapping dictionaries
    site_type_mapping = dict(zip(scat_type_df['Site_Number'], scat_type_df['Site_Type']))
    traffic_lights_mapping = dict(zip(traffic_lights_df['SITE_NO'], traffic_lights_df['SITE_TYPE']))
    
    # Map the NB_SCATS_SITE to get the scat_type
    transformed_df['scat_type'] = transformed_df['NB_SCATS_SITE'].map(site_type_mapping)
    
    # For any missing values, try to fill from traffic_lights_mapping
    missing_mask = transformed_df['scat_type'].isna()
    if missing_mask.any():
        transformed_df.loc[missing_mask, 'scat_type'] = transformed_df.loc[missing_mask, 'NB_SCATS_SITE'].map(traffic_lights_mapping)
    
    # Fill any remaining missing values with "Unknown"
    transformed_df['scat_type'] = transformed_df['scat_type'].fillna("Unknown")
    
    return transformed_df

def add_day_type(transformed_df, holidays_df):
    """
    Adds a day_type column to the transformed dataset indicating if the day is a holiday
    """
    # Find the date column regardless of case
    date_column = None
    for col in transformed_df.columns:
        if col.lower() == 'date':
            date_column = col
            break
    
    if date_column is None:
        print("WARNING: Date column not found in dataset. Using default 'normal' for day_type.")
        transformed_df['day_type'] = 'normal'
        return transformed_df
    
    # Filter holidays for Victoria only (case insensitive)
    vic_holidays_df = holidays_df[holidays_df['Jurisdiction'].str.lower() == 'vic']
    
    # Create a dictionary mapping date to holiday name for faster lookups
    holiday_dict = {}
    
    # Process all holidays
    for _, row in vic_holidays_df.iterrows():
        date_str = row['Date']
        
        # Skip empty date strings, NaN values, or empty strings
        if pd.isna(date_str) or not isinstance(date_str, str) or not date_str.strip():
            continue
        
        # The format in the file is YYYYMMDD (e.g., 20250101)
        try:
            # Extract year, month and day directly
            if len(date_str) == 8 and date_str.isdigit():  # YYYYMMDD format
                year = int(date_str[0:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                
                # Create MM-DD format key for lookup
                date_key = f"{month:02d}-{day:02d}"  # MM-DD format
                holiday_dict[date_key] = row['Holiday Name']
            else:
                continue
        except (ValueError, IndexError):
            continue
    
    # If no Victorian holidays were found, use hardcoded fallback for common holidays
    if not holiday_dict:
        holiday_dict = {
            "01-01": "New Year's Day",
            "01-26": "Australia Day",
            "03-10": "Labour Day",
            "04-18": "Good Friday",
            "04-19": "Saturday before Easter Sunday",
            "04-20": "Easter Sunday",
            "04-21": "Easter Monday",
            "04-25": "ANZAC Day",
            "06-09": "King's Birthday",
            "11-04": "Melbourne Cup",
            "12-25": "Christmas Day",
            "12-26": "Boxing Day"
        }
    
    # Function to check if a date is a holiday
    def is_holiday(date_str):
        if pd.isna(date_str) or not isinstance(date_str, str) or not date_str.strip():
            return "normal"
            
        try:
            # Parse the date (assuming format like "YYYY-MM-DD")
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Create key in MM-DD format for lookup
            date_key = f"{dt.month:02d}-{dt.day:02d}"
            
            # Check if the date is in our holiday dictionary
            if date_key in holiday_dict:
                return holiday_dict[date_key]
            return "normal"
        except (ValueError, TypeError):
            return "normal"
    
    # Apply the function to get day_type
    transformed_df['day_type'] = transformed_df[date_column].apply(is_holiday)
    
    return transformed_df

def add_school_dic(transformed_df, school_dic_df):
    """
    Maps school count from school_dic.csv to the transformed dataset based on SCAT number
    """
    # Check if NB_SCATS_SITE column exists
    if 'NB_SCATS_SITE' not in transformed_df.columns:
        print("WARNING: NB_SCATS_SITE column not found in dataset. Using 0 for school_count.")
        transformed_df['school_count'] = 0
        return transformed_df
    
    try:
        # Convert to same type (float) if possible
        if pd.api.types.is_numeric_dtype(transformed_df['NB_SCATS_SITE']):
            school_dic_df['scat_number'] = pd.to_numeric(school_dic_df['scat_number'], errors='coerce')
        else:
            # Convert both to strings if numeric conversion isn't appropriate
            transformed_df['NB_SCATS_SITE'] = transformed_df['NB_SCATS_SITE'].astype(str)
            school_dic_df['scat_number'] = school_dic_df['scat_number'].astype(str)
    except Exception as e:
        print(f"WARNING: Type conversion issue: {e}. Proceeding with original types.")
    
    # Create mapping dictionary
    school_count_mapping = dict(zip(school_dic_df['scat_number'], school_dic_df['school_count']))
    
    # Map the NB_SCATS_SITE to get the school_count
    transformed_df['school_count'] = transformed_df['NB_SCATS_SITE'].map(school_count_mapping)
    
    # Fill any missing values with 0
    transformed_df['school_count'] = transformed_df['school_count'].fillna(0)
    
    return transformed_df

def process_file(file_path, scat_type_df, traffic_lights_df, holidays_df, school_dic_df, output_dir):
    """
    Process a single transformed data file to add feature engineering columns
    """
    print(f"Processing {file_path}...")
    
    try:
        # Read the CSV file
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading CSV file {file_path}: {e}")
            return False
        
        # Add scat_type
        try:
            df = add_scat_type(df, scat_type_df, traffic_lights_df)
        except Exception as e:
            print(f"Error adding scat_type: {e}")
        
        # Add day_type
        try:
            df = add_day_type(df, holidays_df)
        except Exception as e:
            print(f"Error adding day_type: {e}")
        
        # Add school_count
        try:
            df = add_school_dic(df, school_dic_df)
        except Exception as e:
            print(f"Error adding school_count: {e}")
        
        # Prepare output path
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name.replace('transformed', 'featured'))
        
        # Save the enhanced dataframe
        try:
            df.to_csv(output_path, index=False)
            print(f"Saved to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving file to {output_path}: {e}")
            return False
        
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}")
        return False

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    transformed_dir = os.path.join(base_dir, "Data", "Transformed")
    raw_dir = os.path.join(base_dir, "Data", "Raw", "main")
    output_dir = transformed_dir  # Save back to the same directory
    
    # Load mapping data
    scat_type_path = os.path.join(raw_dir, "scat_type.csv")
    traffic_lights_path = os.path.join(raw_dir, "Traffic_Lights.csv")
    holiday_path = os.path.join(raw_dir, "2025_public_holiday.csv")
    school_dic_path = os.path.join(transformed_dir, "school_dic.csv")
    
    scat_type_df = pd.read_csv(scat_type_path)
    traffic_lights_df = pd.read_csv(traffic_lights_path)
    holidays_df = pd.read_csv(holiday_path)
    school_dic_df = pd.read_csv(school_dic_path)
    
    print(f"Loaded reference data: {len(scat_type_df)} scat types, {len(traffic_lights_df)} traffic lights, {len(holidays_df)} holidays, {len(school_dic_df)} school counts")
    
    # Get all transformed data files for years 2014-2024
    pattern = os.path.join(transformed_dir, "*_transformed_scats_data.csv")
    files = glob.glob(pattern)
    
    # Filter for years 2014-2024
    files = [f for f in files if any(str(year) in f for year in range(2014, 2025))]
    
    if not files:
        print("No transformed files found!")
        return
    
    print(f"Found {len(files)} files to process.")
    
    # Process each file
    success_count = 0
    for file in files:
        if process_file(file, scat_type_df, traffic_lights_df, holidays_df, school_dic_df, output_dir):
            success_count += 1
    
    print(f"Processing complete. Successfully processed {success_count} out of {len(files)} files.")


if __name__ == "__main__":
    # run the full process (add_scat_type => add_day_type => add_school_dic):
    main()