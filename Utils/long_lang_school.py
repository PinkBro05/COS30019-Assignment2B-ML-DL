import pandas as pd
import numpy as np
import os
import math
from pathlib import Path

# Function to calculate haversine distance in meters
def haversine_distance(lon1, lat1, lon2, lat2):
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

def main():
    # Define paths
    base_dir = Path(__file__).parent.parent
    traffic_lights_path = base_dir / "Data" / "Raw" / "main" / "Traffic_Lights.csv"
    school_path = base_dir / "Data" / "Raw" / "main" / "school.csv"
    output_path = base_dir / "Data" / "Transformed" / "school_dic.csv"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load data with explicit encoding
    print("Loading traffic lights data...")
    # Try different encodings - first try utf-8, if fails try others
    try:
        traffic_df = pd.read_csv(traffic_lights_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            traffic_df = pd.read_csv(traffic_lights_path, encoding='latin-1')
        except UnicodeDecodeError:
            traffic_df = pd.read_csv(traffic_lights_path, encoding='cp1252')
    
    print("Loading school data...")
    # Try different encodings for school.csv as well
    try:
        school_df = pd.read_csv(school_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            school_df = pd.read_csv(school_path, encoding='latin-1')
        except UnicodeDecodeError:
            school_df = pd.read_csv(school_path, encoding='cp1252')
    
    # Extract coordinates from traffic lights data directly to separate columns
    # The format is like: POINT (144.377875000086 -38.2366569988309)
    traffic_df[['long', 'lat']] = traffic_df['geometry'].str.extract(r'POINT \((.*) (.*)\)')
    traffic_df['long'] = pd.to_numeric(traffic_df['long'], errors='coerce')
    traffic_df['lat'] = pd.to_numeric(traffic_df['lat'], errors='coerce')
    
    # Extract coordinates from school data
    # X is longitude, Y is latitude
    school_df['long'] = school_df['X'].astype(float)
    school_df['lat'] = school_df['Y'].astype(float)
    
    # Initialize result dataframe
    result_df = pd.DataFrame(columns=['scat_number', 'school_count', 'long', 'lat'])
    
    # Iterate through each traffic light
    print("Calculating distances and counting schools within 100 meters...")
    for index, traffic_row in traffic_df.iterrows():
        if pd.isna(traffic_row['long']) or pd.isna(traffic_row['lat']):
            continue
            
        site_no = traffic_row['SITE_NO']
        long = traffic_row['long']
        lat = traffic_row['lat']
        
        # Count schools within 100 meters
        school_count = 0
        for _, school_row in school_df.iterrows():
            if pd.isna(school_row['long']) or pd.isna(school_row['lat']):
                continue
                
            # Calculate distance
            distance = haversine_distance(
                long, lat,
                school_row['long'], school_row['lat']
            )
            
            # Check if within 500 meters
            if distance <= 500:
                school_count += 1
        
        # Add to result dataframe
        result_df = result_df._append({
            'scat_number': site_no,
            'school_count': school_count,
            'long': long,
            'lat': lat
        }, ignore_index=True)
    
    # Save to CSV
    print(f"Saving results to {output_path}...")
    result_df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main()