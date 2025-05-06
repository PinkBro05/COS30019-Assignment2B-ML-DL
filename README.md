# COS30019 Assignment 2B - ML-DL
# Travel Time Estimation Using Deep Learning Models

This project focuses on estimating travel time using traffic flow data with various deep learning models. The implementation includes Transformer, LSTM, and GRU architectures for time series prediction.

## Project Structure

```
├── Data/
|   ├── Data_dictionary.docx   # Contain metadata and note about the datasets
│   ├── Raw/                    # Raw traffic data files
│   │   └── main/               # Contains original SCATS traffic flow data by year
│   │       ├── 2014/           # Traffic data for 2014 (VSDATA_*.csv files)
│   │       ├── 2025_public_holiday.csv # Public holiday data
│   │       ├── Scat_Data.csv   # SCATS data information
│   │       ├── scat_type.csv   # Types of SCATS sites
│   │       ├── school.csv      # School location data
│   │       ├── Traffic_Lights.csv # Traffic light location data
│   │       └── Traffic_Lights.geojson # GeoJSON for traffic lights
│   └── Transformed/            # Preprocessed and transformed data ready for models
│       ├── _sample_final_time_series.csv # Sample of time series data
│       ├── _sample_final.csv   # Sample of transformed data
│       ├── 2006_final_scats_data.csv # Unit provided data
│       ├── 2014-2024_final.csv # Final data - in wide table form
│       ├── final_time_series_2014_2019.csv # Time series data for 2014-2019
│       ├── final_time_series_2020_2024.csv # Time series data for 2020-2024
│       └── school_dic.csv      # School count data for each SCATS site
├── Transformer/                # Transformer model implementation
│   ├── models/                 # Model architecture definitions
│   │   ├── model.py           # Main Transformer model implementation
│   │   ├── math.py            # Math utility functions
│   │   ├── vanila_encoder.py  # Encoder implementation
│   │   └── vanila_decoder.py  # Decoder implementation
│   ├── utils/
│   │   └── traffic_data_collector.py  # Data loading and preparation for Transformer
│   ├── supervised_learning.py # Training script for Transformer model
│   └── inference.py           # Script for making predictions with trained models
├── LSTM/                       # LSTM model implementation (placeholder)
├── GRU/                        # GRU model implementation (placeholder)
└── Utils/                      # Utility scripts for data processing
    ├── feature_engineering.py  # Add features like school count, site type, day type
    ├── path_utilities.py       # Utilities for managing file paths
    ├── pre_process.py          # Main preprocessing pipeline script
    ├── time_series_reshaping.py # Reshape data for time series modeling
    └── traffic_data_transformation.py # Transform raw SCATS data
```

## Dataset: [HuggingFace](https://huggingface.co/datasets/PinkBro/vicroads-traffic-signals)

## Data Description

The project uses SCATS (Sydney Coordinated Adaptive Traffic System) traffic flow data from 2014 to 2024 [SCAT](https://opendata.transport.vic.gov.au/dataset/traffic-signal-volume-data). The data includes:
- Traffic volume counts at 15-minute intervals 
- Multiple detector sites across road networks
- Additional contextual data like [public holidays](https://data.gov.au/dataset/ds-dga-b1bc6077-dadd-4f61-9f8c-002ab2cdff10/details?q=Australian%20public%20holidays%20combined%C2%A0(2021%E2%80%91{})), [school locations](https://discover.data.vic.gov.au/dataset/school-locations-2024), [traffic lights](https://discover.data.vic.gov.au/dataset/traffic-lights) etc.

The raw data is preprocessed and transformed into a format suitable for time series prediction models.

## Models

### Transformer Model (Vanila)
- Uses self-attention mechanism to capture temporal dependencies
- Encoder-decoder architecture for sequence-to-sequence prediction
- Implements standard transformer components including multi-head attention

### LSTM & GRU Models
- Placeholder directories for future implementation of recurrent neural network models

## Getting Started

### Data Preprocessing

The project includes a comprehensive data preprocessing pipeline in the `Utils` folder. Follow these steps to preprocess the raw traffic data:

1. **Basic Usage**:

   Run the complete preprocessing pipeline with default settings:
   ```bash
   python Utils/pre_process.py
   ```

   This will:
   - Transform raw SCATS data
   - Apply feature engineering (add school counts, site types, etc.)
   - Reshape data for time series modeling
   - Output final preprocessed files in the `Data/Transformed` directory

2. **Advanced Options**:

   Process specific years and customize output:
   ```bash
   python Utils/pre_process.py --years 2014 2015 2016 --split
   ```

   Available options:
   - `--data-path`: Custom path to the raw data directory
   - `--years`: List of specific years to process (e.g., 2014 2015)
   - `--split`: Split output into separate files for date ranges (2014-2019 and 2020-2024)
   - `--start-date`: Filter data starting from this date (YYYY-MM-DD)
   - `--end-date`: Filter data until this date (YYYY-MM-DD)

3. **Individual Preprocessing Steps**:

   You can also run each preprocessing component separately:

   a. **Data Transformation**:
   ```bash
   python -c "from Utils.traffic_data_transformation import TrafficDataTransformer; transformer = TrafficDataTransformer('Data/Raw/main', years=['2014']); transformer.process_all_years()"
   ```

   b. **Feature Engineering**:
   ```bash
   python -c "from Utils.feature_engineering import feature_engineering; feature_engineering('.')"
   ```

   c. **Time Series Reshaping**:
   ```bash
   python -c "from Utils.time_series_reshaping import reshape_traffic_data; reshape_traffic_data('Data/Transformed/2014_featured_scats_data.csv', 'Data/Transformed/2014_time_series.csv')"
   ```

4. **Data Preprocessing Workflow**:

   The complete preprocessing workflow includes:
   
   a. **Raw Data Transformation**: Converts raw SCATS data to a structured format
   b. **Feature Engineering**: Adds contextual features:
      - School counts near each SCATS site
      - Traffic light and SCATS site types
      - Public holiday information
   c. **Time Series Reshaping**: Formats data for sequence prediction tasks

### Training a Model

To train the Transformer model:

```bash
python Transformer/supervised_learning.py
```

This will:
- Load and prepare the transformed data
- Create and train the Transformer model
- Save the best model checkpoint
- Generate a plot of training history

### Making Predictions

To make predictions with a trained model:

```bash
python Transformer/inference.py
```

## Dependencies

- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Contributors
1. **Hong Anh Nguyen** - Data, Transformer
2. **Phong Tran** - LSTM
3. **James Luong**- GRU
COS30019 - Introduction to Artificial Intelligence - Assignment 2B
