"""
Train the NN model.
Run Command: python3 train.py --model <model_name> --epochs <number_of_epochs>
"""
import sys, os
import warnings
import argparse
import numpy as np
import pandas as pd
from data.data import process_data
from model import model
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
warnings.filterwarnings("ignore")


def train_model(model, X_train, y_train, name, config):
    """train
    train a single model.

    # Arguments
        model: Model, NN model to train.
        X_train: ndarray(number, lags), Input data for train.
        y_train: ndarray(number, ), result data for train.
        name: String, name of model.
        config: Dict, parameter for train.
    """

    model.compile(loss="mse", optimizer="rmsprop", metrics=['mape'])
    # early = EarlyStopping(monitor='val_loss', patience=30, verbose=0, mode='auto')
    hist = model.fit(
        X_train, y_train,
        batch_size=config["batch"],
        epochs=config["epochs"],
        validation_split=0.05)

    model.save('ML/GRU/model/' + name + '.h5')
    df = pd.DataFrame.from_dict(hist.history)
    df.to_csv('ML/GRU/model/' + name + ' loss.csv', encoding='utf-8', index=False)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="gru",
        help="Model to train.")
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs to train the model (Default: 30).")
    args = parser.parse_args()

    lag = 12
    config = {"batch": 256, "epochs": args.epochs}
    file1 = os.path.join(os.path.dirname(__file__), "data/train.csv")
    file2 = os.path.join(os.path.dirname(__file__), "data/test.csv")
    X_train, y_train, _, _, _ = process_data(file1, file2, lag)

    if args.model == 'gru':
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        # Model structure: [input_sequence_length, first_hidden, second_hidden, output_units]
        m = model.get_gru([12, 64, 64, 1]) 
        train_model(m, X_train, y_train, args.model, config)


if __name__ == '__main__':
    main(sys.argv)
