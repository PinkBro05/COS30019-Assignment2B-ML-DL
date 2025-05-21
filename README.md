# COS30019 Assignment 2B - ML-DL
# Travel Time Estimation Using Deep Learning Models

This project focuses on estimating travel time using traffic flow data with various deep learning models. The implementation includes Transformer, LSTM, and GRU architectures for time series prediction.

## Project Structure

```
├── Data/
|   ├── Data_dictionary.docx   # Contain metadata and note about the datasets
│   ├── Raw/                    # Raw traffic data files
│   │   ├── main/               # Contains original SCATS traffic flow data by year
│   │   │   ├── 2014/           # Traffic data for 2014 (VSDATA_*.csv files)
│   │   │   ├── 2025_public_holiday.csv # Public holiday data
│   │   │   ├── Scat_Data.csv   # SCATS data information
│   │   │   ├── scat_type.csv   # Types of SCATS sites
│   │   │   ├── school.csv      # School location data
│   │   │   ├── Traffic_Lights.csv # Traffic light location data
│   │   │   └── Traffic_Lights.geojson # GeoJSON for traffic lights
│   │   └── metadata and excel file/ #Unit Files 
│   └── Transformed/            # Preprocessed and transformed data ready for models
│       ├── _sample_final_time_series.csv # Sample of time series data
│       ├── _sample_final.csv   # Sample of transformed data
│       ├── 2006_final_scats_data.csv # Unit provided data
│       ├── 2024_final_time_series.csv # Final data - in wide table form for 2024
│       ├── test_2025_final_time_series.csv # 2025 test data
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
├── LSTM/                       # LSTM model implementation 
├── GRU/                        # GRU model implementation
│   ├── data/                   # Data processing modules
│   │   ├── data.py             # Main data processing functions
│   │   ├── parse_data.py       # Data parsing utilities
│   │   ├── train.csv           # Training dataset
│   │   └── test.csv            # Testing dataset
│   ├── images/                 # Architecture diagrams
│   │   ├── GRU.png             # GRU model architecture diagram
│   │   └── LSTM.png            # LSTM model architecture diagram
│   ├── model/                  # Model definitions
│   │   ├── model.py            # GRU and LSTM model implementations
│   │   ├── gru.h5              # Saved GRU model
│   │   └── lstm.h5             # Saved LSTM model
│   ├── main.py                 # Main execution script for predictions
│   └── train.py                # Training script for GRU and LSTM models
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
**Note:** This step expects large amount of time and computational resource due to the amount of raw data

The project includes a comprehensive data preprocessing pipeline in the `Utils` folder. Follow these steps to preprocess the raw traffic data:

1. **Basic Usage**:

   Run the complete preprocessing pipeline with default settings:
   ```bash
   python Utils/pre_process.py
   ```

   This will:
   - Transform **raw SCATS data** (This assume you have downloaded all data from [SCAT](https://opendata.transport.vic.gov.au/dataset/traffic-signal-volume-data), create and name folder by year and extract data to the folder. E.g. `Data/Raw/main/2014` - same for the rest) 
   - Apply feature engineering (add school counts, site types, etc.)
   - Reshape data for time series modeling
   - Output final preprocessed files in the `Data/Transformed` directory

2. **Advanced Options**:

   Process specific years and customize output:
   ```bash
   python Utils/pre_process.py --years 2014 2015 2016 --split
   ```

   Available options:
   - `--data-path`: Custom path to the raw data directory (`Data/Raw/main/`)
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
   
   I. **Raw Data Transformation**: Converts raw SCATS data to a structured format

   II. **Feature Engineering**: Adds contextual features:
      - School counts near each SCATS site
      - Traffic light and SCATS site types
      - Public holiday information

   III. **Time Series Reshaping**: Formats data for sequence prediction tasks

### Training a Model

To train the Transformer model:

```bash
python Transformer/supervised_learning.py --data_file Data/Transformed/2024_final_time_series.csv
```

#### Training Parameters

You can customize the training process with various command-line arguments:

```bash
python Transformer/supervised_learning.py --data_file Data/Transformed/2024_final_time_series.csv --batch_size 64 --num_epochs 100 --learning_rate 0.0001
```

Key parameters:
- `--data_file`: Path to the time series data file (required)
- `--batch_size`: Batch size for training (default: 32)
- `--num_epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate for optimizer (default: 0.0001)
- `--embedding_dim`: Dimension for categorical embeddings (default: 16)
- `--d_model`: Transformer model dimension (default: a64)
- `--num_heads`: Number of attention heads (default: 8)
- `--num_layers`: Number of encoder/decoder layers (default: 3)
- `--weight_decay`: Weight decay for regularization (default: 1e-5)

#### Examples

Train on a sample dataset (quick test):
```bash
python Transformer/supervised_learning.py --data_file Data/Transformed/_sample_final_time_series.csv --num_epochs 10
```

Train with a larger model for better performance:
```bash
python Transformer/supervised_learning.py --data_file Data/Transformed/2024_final_time_series.csv --d_model 128 --num_heads 8 --num_layers 4
```

The training script will:
- Load and prepare the transformed time series data
- Create and train the Transformer model with specified parameters
- Save the best model to `Transformer/save_models/transformer_traffic_model.pth`
- Generate training history plots in `Transformer/save_figures/transformer_training_history.png`

### Testing a Trained Model

You can test a trained model's performance on test data:

```bash
python Transformer/supervised_learning.py --data_file Data/Transformed/2024_final_time_series.csv --test --model_path Transformer/save_models/transformer_traffic_model.pth
```

#### Testing Parameters

Key parameters for testing:
- `--test`: Flag to run in test mode instead of training
- `--data_file`: Path to the test data file
- `--model_path`: Path to the saved model (required for testing)
- `--plot_test_results`: Plot visualization of test results (default: True)
- `--test_plot_name`: Name for the test results plot (default: transformer_test_results.png)

The test function will:
- Evaluate the model on the test dataset
- Calculate performance metrics (MSE, MAE, RMSE)
- Generate a scatter plot comparing predicted vs. actual values
- Save the visualization to `Transformer/save_figures/transformer_test_results.png` (or custom name)

#### Examples

Test with a specific model on the 2020-2024 dataset:
```bash
python Transformer/supervised_learning.py --data_file Data/Transformed/final_time_series_2020_2024.csv --test --model_path Transformer/save_models/transformer_traffic_model.pth
```

Test on sample data with a custom plot name:
```bash
python Transformer/supervised_learning.py --data_file Data/Transformed/_sample_final_time_series.csv --test --model_path Transformer/save_models/transformer_traffic_model.pth --test_plot_name my_test_results.png
```

### Making Predictions

To make predictions with a trained model:

```bash
python Transformer/inference.py
```

#### Inference Parameters

The inference script offers several parameters to customize prediction:

```bash
python Transformer/inference.py --input_path Data/Transformed/final_time_series_2020_2024.csv --index 1000 --output_step 4 --output_file predictions.png
```

Key parameters:
- `--input_path`: Path to the test CSV file (default: sample data)
- `--index`: Row index in the CSV file to use as the prediction point (default: random)
- `--output_step`: Number of step to predict (default: 4)
- `--model_path`: Path to the saved model (default: uses the standard saved model)
- `--output_file`: Path to save the prediction plot (optional)

#### Examples

Predict using a specific model at a particular time point:
```bash
python Transformer/inference.py --input_path Data/Transformed/final_time_series_2020_2024.csv --index 5000 --model_path Transformer/save_models/transformer_traffic_model.pth
```

Run inference on different datasets:
```bash
python Transformer/inference.py --input_path Data/Transformed/_sample_final_time_series.csv
```

The inference script will:
- Load the specified time series data
- Prepare the data for the selected index point
- Make predictions for the next 4 time steps (1 hour)
- Display a visualization comparing predicted values with actual values (if available)
- Save the visualization to the specified output file (if provided)

## Dependencies

- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- warnings (for suppressing warnings)

## Contributors
1. **Hong Anh Nguyen** - Data, Transformer
2. **Phong Tran** - LSTM
3. **James Luong**- GRU

COS30019 - Introduction to Artificial Intelligence - Assignment 2B
