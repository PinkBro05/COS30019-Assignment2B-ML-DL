import os
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime

from path_utilities import PathManager


class TrafficDataTransformer:
    def __init__(self, data_path, years=None, output_path=None):
        """Initialize the traffic data transformer.
        
        Args:
            data_path: Path to the Raw/main directory containing year folders
            years: List of years to process (default: all years available)
            output_path: Path to save transformed data
        """
        self.data_path = data_path
        self.years = years if years else self._get_available_years()
        self.output_path = output_path if output_path else os.path.join(os.path.dirname(data_path), '..', 'Transformed')
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
    def _get_available_years(self):
        """Get a list of available years from the data directory."""
        return [d for d in os.listdir(self.data_path) 
                if os.path.isdir(os.path.join(self.data_path, d)) and d.isdigit()]
    
    def read_data_files(self, year, limit=5):
        """Read all VSDATA files for a specific year.
        
        Args:
            year: Year to process (str)
            limit: Limit number of files to read (None for all files)
            
        Returns:
            DataFrame containing combined data from all files
        """
        year_dir = os.path.join(self.data_path, str(year))
        csv_files = glob(os.path.join(year_dir, 'VSDATA_*.csv'))
        
        if not csv_files:
            print(f"No data files found for year {year}")
            return None
        
        # Read and combine all CSV files for the year
        dfs = []
        for file in sorted(csv_files)[:limit] if limit else sorted(csv_files):
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"Processed {os.path.basename(file)}")
            except Exception as e:
                print(f"Error reading {file}: {e}")
        
        if not dfs:
            return None
            
        return pd.concat(dfs, ignore_index=True)
    
    def transform_data(self, df, year=None):
        """Transform the raw data into a format suitable for the Transformer model.
        
        Args:
            df: DataFrame containing raw VSDATA
            year: The year of data being processed (for logging)
            
        Returns:
            DataFrame with transformed data
        """
        if df is None or df.empty:
            return None
            
        year_str = f" for {year}" if year else ""
        print(f"Starting transformation{year_str}. Found columns: {', '.join(df.columns[:5])}...")
        
        # Check if required columns exist
        required_cols = ['NB_SCATS_SITE', 'QT_INTERVAL_COUNT', 'NB_DETECTOR']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return None
            
        # Select only the necessary columns
        # Identify traffic flow columns (Vxx) dynamically
        flow_cols = [col for col in df.columns if col.startswith('V') and col[1:].isdigit()]
        if not flow_cols:
            print(f"Error: No traffic flow columns (Vxx) found in the data")
            return None
            
        print(f"Found {len(flow_cols)} traffic flow columns (V00-V{flow_cols[-1][1:]})")
            
        # Get only the columns we need
        selected_cols = ['NB_SCATS_SITE', 'QT_INTERVAL_COUNT', 'NB_DETECTOR'] + flow_cols
        
        # Make sure all required columns are in the dataframe
        available_cols = [col for col in selected_cols if col in df.columns]
        if len(available_cols) < len(selected_cols):
            missing = set(selected_cols) - set(available_cols)
            print(f"Warning: Some columns are missing: {missing}")
            
        df_selected = df[available_cols].copy()
        
        # Filter out rows with negative or missing values (e.g., -1023 indicates errors/missing data)
        initial_rows = len(df_selected)
        for col in flow_cols:
            df_selected = df_selected[df_selected[col] >= 0]
        filtered_rows = len(df_selected)
        print(f"Filtered out {initial_rows - filtered_rows} rows with negative values")
        
        # Check if we still have data after filtering
        if df_selected.empty:
            print("No data left after filtering")
            return None
        
        # Convert date column to datetime
        try:
            # Try multiple date formats
            date_formats = [
                # Try with format inference first
                lambda x: pd.to_datetime(x, errors='coerce'),
                # Try specific formats if inference fails
                lambda x: pd.to_datetime(x, format='%Y-%m-%d', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%Y-%m-%d %H:%M:%S', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%Y%m%d', errors='coerce'),
                lambda x: pd.to_datetime(x, format='%d/%m/%Y', errors='coerce')
            ]
            
            # Try each format until we get a valid result
            df_selected['QT_INTERVAL_COUNT_datetime'] = None
            for date_format in date_formats:
                # Try to convert using the current format
                converted = date_format(df_selected['QT_INTERVAL_COUNT'])
                if not converted.isna().all():
                    df_selected['QT_INTERVAL_COUNT_datetime'] = converted
                    print(f"Successfully converted dates using format: {date_format.__name__ if hasattr(date_format, '__name__') else 'custom format'}")
                    break
            
            # Check if conversion was successful
            if df_selected['QT_INTERVAL_COUNT_datetime'].isna().all():
                raise ValueError("Could not parse dates with any of the provided formats")
                
            # Extract date and time index
            df_selected['date'] = df_selected['QT_INTERVAL_COUNT_datetime'].dt.date
            df_selected['time_idx'] = df_selected['QT_INTERVAL_COUNT_datetime'].dt.hour * 4 + df_selected['QT_INTERVAL_COUNT_datetime'].dt.minute // 15
                
        except Exception as e:
            print(f"Error converting datetime: {e}")
            # Try alternative datetime format if available
            if 'QT_DATE' in df.columns and 'QT_TIME' in df.columns:
                try:
                    print("Trying alternative date/time columns (QT_DATE and QT_TIME)...")
                    # Try multiple formats for date combination
                    df_selected['date_str'] = df['QT_DATE'].astype(str)
                    if 'QT_TIME' in df.columns:
                        df_selected['time_str'] = df['QT_TIME'].astype(str)
                        df_selected['datetime_str'] = df_selected['date_str'] + ' ' + df_selected['time_str']
                    else:
                        df_selected['datetime_str'] = df_selected['date_str']
                    
                    # Try to parse the combined datetime string
                    df_selected['combined_datetime'] = pd.to_datetime(df_selected['datetime_str'], errors='coerce')
                    
                    # Check if conversion was successful
                    if df_selected['combined_datetime'].isna().all():
                        raise ValueError("Could not parse dates from QT_DATE and QT_TIME columns")
                        
                    df_selected['date'] = df_selected['combined_datetime'].dt.date
                    df_selected['time_idx'] = df_selected['combined_datetime'].dt.hour * 4 + df_selected['combined_datetime'].dt.minute // 15
                    
                except Exception as e2:
                    print(f"Error with alternative datetime approach: {e2}")
                    # Try one more approach - extract date info from filenames if possible
                    try:
                        if 'filename' in df.columns:
                            print("Attempting to extract date from filename...")
                            # Assuming filenames contain date info like YYYYMMDD
                            df_selected['filename_date'] = df['filename'].str.extract(r'(\d{8})')[0]
                            df_selected['date'] = pd.to_datetime(df_selected['filename_date'], format='%Y%m%d').dt.date
                            # Assign arbitrary time indices if time not available
                            df_selected['time_idx'] = range(len(df_selected)) % 96  # 96 time slots in a day (15-min intervals)
                        else:
                            return None
                    except:
                        return None
            else:
                return None
        
        # Create a pivot table with sites as rows, time intervals as columns
        # Sum traffic flow across all detectors for each site
        try:
            df_pivot = df_selected.groupby(['NB_SCATS_SITE', 'date', 'time_idx'])[flow_cols].sum().reset_index()
            
            # Create a sequence for each site and date
            df_transformed = df_pivot.pivot_table(
                index=['NB_SCATS_SITE', 'date'],
                columns='time_idx',
                values=flow_cols
            )
            
            # Flatten the column hierarchy
            df_transformed.columns = [f'{col[0]}_{col[1]}' for col in df_transformed.columns]
            df_transformed = df_transformed.reset_index()
            
            print(f"Successfully transformed data. Result has {len(df_transformed)} rows and {len(df_transformed.columns)} columns")
            return df_transformed
            
        except Exception as e:
            print(f"Error during pivot/transform: {e}")
            return None
    
    def create_sample(self):
        """Create a sample of transformed data from the first few days of a specified year."""
        print("Creating sample data from 2014...")
        
        # Use only the first year (2014) and limit to a few days for the sample
        sample_year = '2014'
        
        # Read a limited set of files
        raw_data = self.read_data_files(sample_year, limit=5)
        
        if raw_data is not None:
            # Transform the sample data
            sample_transformed = self.transform_data(raw_data)
            
            if sample_transformed is not None:
                # Further limit to a reasonable number of rows if needed
                if len(sample_transformed) > 100:
                    sample_transformed = sample_transformed.head(100)
                
                # Save the sample to a CSV file
                sample_file = os.path.join(self.output_path, 'sample_final.csv')
                sample_transformed.to_csv(sample_file, index=False)
                print(f"Saved sample transformed data to {sample_file}")
                
                # Print sample info
                print(f"Sample contains {len(sample_transformed)} records")
                print(f"Sample columns: {sample_transformed.columns.tolist()[:5]}... (total: {len(sample_transformed.columns)})")
                
                return sample_transformed
        
        print("Failed to create sample data")
        return None
    
    def process_all_years(self):
        """Process data for all specified years and save each year to a separate CSV file."""
        processed_years = []
        
        for year in self.years:
            print(f"Processing data for year {year}...")
            raw_data = self.read_data_files(year, limit=None)
            transformed_data = self.transform_data(raw_data, year=year)
            
            if transformed_data is not None:
                # Save each year's data to a separate CSV file
                output_file = os.path.join(self.output_path, f'{year}_transformed_scats_data.csv')
                transformed_data.to_csv(output_file, index=False)
                print(f"Saved transformed data for year {year} to {output_file}")
                
                # Include year information in the DataFrame
                transformed_data['year'] = year
                processed_years.append(year)
                
                print(f"Successfully transformed data for year {year} with {len(transformed_data)} records")
        
        if processed_years:
            print(f"Successfully processed data for years: {', '.join(processed_years)}")
            return processed_years
        else:
            print("No data was processed successfully")
            return None


def load_csv_with_fallback_encoding(file_path, encodings=None):
    """Load a CSV file with fallback encodings if the primary encoding fails.
    
    Args:
        file_path: Path to the CSV file
        encodings: List of encodings to try (defaults to ['utf-8', 'latin-1', 'cp1252'])
        
    Returns:
        Pandas DataFrame of the loaded CSV
        
    Raises:
        FileNotFoundError: If the file does not exist
        ValueError: If the file could not be loaded with any of the provided encodings
    """
    if encodings is None:
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    for encoding in encodings:
        try:
            return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            continue
            
    raise ValueError(f"Could not read {file_path} with any of the provided encodings: {encodings}")