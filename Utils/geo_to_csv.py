
import json
import csv
import os
import sys
from pathlib import Path

def geojson_to_csv(geojson_file_path, csv_output_path):
    """
    Convert Traffic_Lights.geojson to map.csv
    Extracts SITE_NO as Scats and coordinates as longitude and latitude
    
    Args:
        geojson_file_path (str): Path to the GeoJSON file
        csv_output_path (str): Path to save the output CSV file
    """
    print(f"Converting {geojson_file_path} to {csv_output_path}...")
    
    # Read the GeoJSON file
    with open(geojson_file_path, 'r') as geojson_file:
        # Skip the first line if it contains comment marker (//), and read the rest
        first_line = geojson_file.readline().strip()
        if first_line.startswith('//'):
            # Reopen the file and skip the first line
            geojson_file.seek(0)
            data_lines = geojson_file.readlines()[1:]
            geojson_data = json.loads(''.join(data_lines))
        else:
            # If first line is not a comment, reset file pointer and read whole file
            geojson_file.seek(0)
            geojson_data = json.load(geojson_file)
    
    # Create CSV file
    with open(csv_output_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        # Write header
        csv_writer.writerow(['Scats', 'longitude', 'latitude', 'location'])
        
        # Extract data from each feature
        for feature in geojson_data['features']:
            properties = feature['properties']
            geometry = feature['geometry']
            
            # Extract SITE_NO as Scats
            scats = properties.get('SITE_NO')
            
            # Extract location
            location = properties.get('SITE_NAME')
            
            # Extract coordinates
            if geometry['type'] == 'Point':
                longitude, latitude = geometry['coordinates']
                
                # Write to CSV
                csv_writer.writerow([scats, longitude, latitude, location])
    
    print(f"Conversion completed. CSV file saved to {csv_output_path}")

def main():
    # Define file paths
    base_dir = Path(__file__).parents[1]  # Go up one directory from Utils
    geojson_file = base_dir / "ML" / "Data" / "Raw" / "main" / "Traffic_Lights.geojson"
    csv_output = base_dir / "Data" /"map.csv"
    
    # Convert GeoJSON to CSV
    geojson_to_csv(geojson_file, csv_output)

if __name__ == "__main__":
    main()