import os
import pandas as pd
import numpy as np
import re
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed

from path_utilities import PathManager
from traffic_data_transformation import load_csv_with_fallback_encoding


def haversine_distance(lon1, lat1, lon2, lat2):
    """Calculate haversine distance in meters between two points on Earth."""
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Radius of Earth in meters
    r = 6371000
    return c * r


def prepare_school_data(base_dir):
    """Prepare school count data for each SCAT site."""
    # Use the path manager to construct correct paths
    raw_dir = PathManager.get_raw_dir(base_dir)
    transformed_dir = PathManager.get_transformed_dir(base_dir)
    
    traffic_lights_path = os.path.join(raw_dir, "Traffic_Lights.csv")
    school_path = os.path.join(raw_dir, "school.csv")
    output_path = os.path.join(transformed_dir, "school_dic.csv")
    
    # Create directory if it doesn't exist
    PathManager.ensure_dir_exists(os.path.dirname(output_path))
    
    print(f"Loading traffic lights data from: {traffic_lights_path}")
    print(f"Loading school data from: {school_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Load data with explicit encoding and handle errors
    try:
        traffic_df = load_csv_with_fallback_encoding(traffic_lights_path)
        school_df = load_csv_with_fallback_encoding(school_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Trying alternative paths...")
        
        # Try alternative path structures
        alt_paths = [
            os.path.join(os.path.dirname(base_dir), "Data", "Raw", "main"),
            os.path.join(base_dir, "Raw", "main")
        ]
        
        traffic_file = PathManager.find_file("Traffic_Lights.csv", alt_paths)
        school_file = PathManager.find_file("school.csv", alt_paths)
        
        if not traffic_file or not school_file:
            raise FileNotFoundError(f"Could not find required data files in any of the tried paths: {alt_paths}")
        
        traffic_df = load_csv_with_fallback_encoding(traffic_file)
        school_df = load_csv_with_fallback_encoding(school_file)
    
    # Extract coordinates from traffic lights data
    traffic_df[['long', 'lat']] = traffic_df['geometry'].str.extract(r'POINT \((.*) (.*)\)')
    traffic_df['long'] = pd.to_numeric(traffic_df['long'], errors='coerce')
    traffic_df['lat'] = pd.to_numeric(traffic_df['lat'], errors='coerce')
    
    # Extract coordinates from school data
    school_df['long'] = school_df['X'].astype(float)
    school_df['lat'] = school_df['Y'].astype(float)
    
    # Initialize result dataframe
    result_df = pd.DataFrame(columns=['scat_number', 'school_count', 'long', 'lat'])
    
    # Parallelize distance calculations and school counting
    print("Calculating distances and counting schools within 500 meters in parallel...")
    records = traffic_df.to_dict('records')
    result_data = []
    max_workers = min(len(records), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_record = {}
        for rec in records:
            if pd.isna(rec['long']) or pd.isna(rec['lat']):
                continue
            future = executor.submit(count_nearby_schools, rec, school_df, 500)
            future_to_record[future] = rec
        for future in as_completed(future_to_record):
            rec = future_to_record[future]
            try:
                count = future.result()
                result_data.append({
                    'scat_number': rec['SITE_NO'],
                    'school_count': count,
                    'long': rec['long'],
                    'lat': rec['lat']
                })
            except Exception as e:
                print(f"Error counting schools for site {rec.get('SITE_NO')}: {e}")
    
    # Convert to DataFrame
    result_df = pd.DataFrame(result_data)
    
    # Save to CSV
    print(f"Saving school count data to {output_path}...")
    result_df.to_csv(output_path, index=False)
    print("School count data preparation complete!")
    
    return result_df


def count_nearby_schools(traffic_point, school_df, max_distance):
    """Count schools within specified distance of a traffic light.
    
    Args:
        traffic_point: Row from traffic_df with long and lat
        school_df: DataFrame containing school locations
        max_distance: Maximum distance in meters
        
    Returns:
        Number of schools within the specified distance
    """
    count = 0
    traffic_long = traffic_point['long']
    traffic_lat = traffic_point['lat']
    
    for _, school_row in school_df.iterrows():
        if pd.isna(school_row['long']) or pd.isna(school_row['lat']):
            continue
            
        # Calculate distance
        distance = haversine_distance(
            traffic_long, traffic_lat,
            school_row['long'], school_row['lat']
        )
        
        # Check if within max_distance
        if distance <= max_distance:
            count += 1
            
    return count


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
            from datetime import datetime
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
            return True, output_path
        except Exception as e:
            print(f"Error saving file to {output_path}: {e}")
            return False, None
        
    except Exception as e:
        print(f"Unexpected error processing {file_path}: {e}")
        return False, None


def feature_engineering(base_dir, transformed_files=None):
    """Apply feature engineering to transformed data files."""
    # Define paths - check if base_dir already ends with 'Data'
    if base_dir.endswith('Data'):
        transformed_dir = os.path.join(base_dir, "Transformed")
        raw_dir = os.path.join(base_dir, "Raw", "main")
    else:
        transformed_dir = os.path.join(base_dir, "Data", "Transformed")
        raw_dir = os.path.join(base_dir, "Data", "Raw", "main")
    
    output_dir = transformed_dir  # Save back to the same directory
    
    # Load mapping data
    scat_type_path = os.path.join(raw_dir, "scat_type.csv")
    traffic_lights_path = os.path.join(raw_dir, "Traffic_Lights.csv")
    holiday_path = os.path.join(raw_dir, "2025_public_holiday.csv")
    school_dic_path = os.path.join(transformed_dir, "school_dic.csv")
    
    # Print the paths for debugging
    print(f"Using the following paths:")
    print(f"- Scat type: {scat_type_path}")
    print(f"- Traffic lights: {traffic_lights_path}")
    print(f"- Holiday data: {holiday_path}")
    print(f"- School data: {school_dic_path}")
    
    # Prepare school data if it doesn't exist
    if not os.path.exists(school_dic_path):
        print("School dictionary not found. Generating...")
        prepare_school_data(base_dir)
    
    try:
        scat_type_df = pd.read_csv(scat_type_path)
        traffic_lights_df = pd.read_csv(traffic_lights_path)
        holidays_df = pd.read_csv(holiday_path)
        school_dic_df = pd.read_csv(school_dic_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Checking alternate path structure...")
        
        # Try alternative path structure
        alt_base_dir = os.path.dirname(base_dir) if base_dir.endswith('Data') else os.path.join(base_dir, "Data")
        alt_raw_dir = os.path.join(alt_base_dir, "Raw", "main")
        
        print(f"Trying alternative paths:")
        print(f"- Alt raw dir: {alt_raw_dir}")
        
        # Try with alternative paths
        scat_type_path = os.path.join(alt_raw_dir, "scat_type.csv")
        traffic_lights_path = os.path.join(alt_raw_dir, "Traffic_Lights.csv")
        holiday_path = os.path.join(alt_raw_dir, "2025_public_holiday.csv")
        
        print(f"- Alt scat type: {scat_type_path}")
        print(f"- Alt traffic lights: {traffic_lights_path}")
        print(f"- Alt holiday data: {holiday_path}")
        
        scat_type_df = pd.read_csv(scat_type_path)
        traffic_lights_df = pd.read_csv(traffic_lights_path)
        holidays_df = pd.read_csv(holiday_path)
        school_dic_df = pd.read_csv(school_dic_path)
    
    print(f"Loaded reference data: {len(scat_type_df)} scat types, {len(traffic_lights_df)} traffic lights, {len(holidays_df)} holidays, {len(school_dic_df)} school counts")
    
    # Get files to process
    if transformed_files is None:
        # Get all transformed data files for years 2014-2025
        pattern = os.path.join(transformed_dir, "*_transformed_scats_data.csv")
        files = glob(pattern)
        
        # Filter for years 2014-2025
        files = [f for f in files if any(str(year) in f for year in range(2014, 2026))]
    else:
        files = transformed_files if isinstance(transformed_files, list) else [transformed_files]
    
    if not files:
        print("No transformed files found!")
        return []
    
    print(f"Found {len(files)} files to process.")
    
    # Process each transformed file in parallel
    print(f"Processing {len(files)} feature engineering tasks in parallel...")
    success_files = []
    max_workers = min(len(files), os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file, scat_type_df, traffic_lights_df, holidays_df, school_dic_df, output_dir): file for file in files}
        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                success, out_path = future.result()
                if success:
                    success_files.append(out_path)
            except Exception as e:
                print(f"Error processing file {file}: {e}")

    print(f"Feature engineering complete. Successfully processed {len(success_files)} out of {len(files)} files.")
    return success_files