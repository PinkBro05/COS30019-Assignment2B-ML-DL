import os
import sys
from data_collector import TrafficFlow

def main():
    # Path to the data file
    data_path = '../Data/Raw/Scat_Data.csv'
    
    # Check if the file exists
    if not os.path.exists(data_path):
        # Try another path
        data_path = 'Data/Raw/Scat_Data.csv'
        if not os.path.exists(data_path):
            print(f"Error: Could not find the data file at {data_path}")
            return
    
    print(f"Reading data from: {data_path}")
    
    # Create TrafficFlow object
    traffic_flow = TrafficFlow(data_path)
    
    # Path for transformed data
    transformed_path = '../Data/Transformed/transformed_scats_data.csv'
    if not os.path.exists(os.path.dirname(transformed_path)):
        os.makedirs(os.path.dirname(transformed_path), exist_ok=True)
        transformed_path = 'Data/Transformed/transformed_scats_data.csv'
        if not os.path.exists(os.path.dirname(transformed_path)):
            os.makedirs(os.path.dirname(transformed_path), exist_ok=True)
    
    # Transform data into time series format with 15-minute intervals
    transformed_df = traffic_flow.transform_data_for_time_series(save_path=transformed_path)
    
    print(f"\nTransformation complete! Time series data has been saved to: {transformed_path}")
    print(f"The dataset contains {len(transformed_df)} rows, with each row representing a 15-minute interval.")
    print(f"Sample of the transformed data:")
    print(transformed_df.head(10))
    
    # Display information about the unique locations and time periods
    locations = transformed_df['SCATS Number'].nunique()
    dates = transformed_df['Date'].str.split(' ').str[0].nunique()
    print(f"\nThe dataset contains traffic flow data for {locations} unique locations over {dates} days.")
    print("Each location has data in 15-minute intervals for each day.")

if __name__ == "__main__":
    main()