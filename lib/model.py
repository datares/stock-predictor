from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers
from keras.callbacks import CSVLogger

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from lib.utils import build_window

def setup_model(n_features, batch_size, look_back):
    """
    Returns a keras LSTM model. Our architecture will be kept 
    in this method.
    """
    model = Sequential()

    model.add(LSTM(units = 64, return_sequences = True, input_shape = (look_back, n_features)))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 128, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 256, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 512, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 256, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 128, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(units = 64))
    model.add(Dropout(0.2))

    model.add(Dense(units = 1))

    model.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return model

def train_model(model, x_train, y_train, epochs, batch_size, lr):
    """
    Takes a training dataset and a model and returns a trained model 
    after ts timesteps.
    """
    model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)

    return model
