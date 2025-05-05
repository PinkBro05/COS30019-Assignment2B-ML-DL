# COS30019 Assignment 2B - ML-DL
# Travel Time Estimation Using Deep Learning Models

This project focuses on estimating travel time using traffic flow data with various deep learning models. The implementation includes Transformer, LSTM, and GRU architectures for time series prediction.

## Project Structure

```
├── Data/
│   ├── Raw/                    # Raw traffic data files
│   │   └── main/               # Contains original SCATS traffic flow data by year
│   │       ├── 2014/           # Traffic data for 2014 (VSDATA_*.csv files)
│   │       ├── 2015/           # Traffic data for 2015
│   │       ├── ...             # Additional years
│   │       └── ...             # Additional data files (holidays, schools, etc.)
│   └── Transformed/            # Preprocessed and transformed data ready for models
│       ├── 2014_transformed_scats_data.csv
│       ├── 2015_transformed_scats_data.csv
│       └── ...
├── Transformer/                # Transformer model implementation
│   ├── models/                 # Model architecture definitions
│   │   ├── model.py           # Main Transformer model implementation
│   │   ├── vanila_encoder.py  # Encoder implementation
│   │   └── vanila_decoder.py  # Decoder implementation
│   ├── utils/
│   │   └── traffic_data_collector.py  # Data loading and preparation for Transformer
│   ├── save_models/           # Saved model checkpoints
│   │   └── transformer_model_test.pth
│   ├── inference.py           # Script for making predictions with trained models
│   └── supervised_learning.py # Training script for Transformer model
├── LSTM/                       # LSTM model implementation (placeholder)
├── GRU/                        # GRU model implementation (placeholder)
└── Utils/                      # Utility scripts for data processing
    ├── pre_processing.py      # General data preprocessing utilities
    └── transform_traffic_data.py  # Script to transform raw data for model input
```

## Data Description

The project uses SCATS (Sydney Coordinated Adaptive Traffic System) traffic flow data from 2014 to 2024. The data includes:
- Traffic volume counts at 15-minute intervals
- Multiple detector sites across road networks
- Additional contextual data like public holidays, school locations, etc.

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
To transform the raw traffic data:

1. Run the transformation script:
```
python Utils/transform_traffic_data.py
```
This will:
- Create a sample of transformed data for review
- Ask if you want to proceed with processing all years
- Generate CSV files with transformed data in the Data/Transformed directory

### Training a Model
To train the Transformer model:

```
python Transformer/supervised_learning.py
```

This will:
- Load and prepare the transformed data
- Create and train the Transformer model
- Save the best model checkpoint
- Generate a plot of training history

### Making Predictions
To make predictions with a trained model:

```
python Transformer/inference.py
```

## Dependencies

- PyTorch
- NumPy
- Pandas
- Matplotlib
- scikit-learn

## Contributors

COS30019 - Introduction to Artificial Intelligence - Assignment 2B
