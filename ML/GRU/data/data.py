"""
Processing the data
Loads training and testing data from CSV files, reshapes them, and splits them into training and testing sets.
Extracts traffic flow data, applies scaling, and creates lagged features for time series analysis.
Normalizes the data using MinMaxScaler and reshapes it for model training.
Converts the reshaped data into NumPy arrays and shuffles the training data.
Return the processed X (input) and y (output) sets for both training and testing.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_data(train, test, lags):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """
    attr = 'Lane 1 Flow (Veh/15 Minutes)'
    # Read CSV into DataFrame
    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8').fillna(0)

    # scaler = StandardScaler().fit(df1[attr].values)
    # Fits a MinMaxScaler on the training data to scale values to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    # ensures neural networks converge faster and more reliably
    # applies scaler to both train and test data and reshape into 1D array
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    print('flow1:', flow1)
    print('flow2:', flow2)

    train, test = [], []
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])

    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])

    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, scaler
