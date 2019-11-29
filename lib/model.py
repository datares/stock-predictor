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

    model.add(LSTM(units = 64, return_sequences = True, input_shape = (look_back, 1)))
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

def validate_model(validation_data, model, look_back=30):
    """
    Takes a validation dataset and a trained model and validates its performance. 
    Should return the accuracy of the model. 

    We are going to be using this method for testing too. 
    """
    scale = MinMaxScaler()
    scaled_df_val = scale.fit_transform(validation_data)

    # Reshaping x_val for the LSTM
    x_val, y_val = build_window(scaled_df_val, look_back)

    x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], 1))
    
    # Predicting data using validation data
    predicted_stock_price = model.predict(x_val)

    # Converting back from normalized data
    predicted_stock_price = scale.inverse_transform(predicted_stock_price)

    # evaluating it
    results = model.evaluate(x_val, y_val)

    # Plotting results for seen data
    plt.figure()
    plt.plot(validation_data[look_back:])
    plt.plot(predicted_stock_price)
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend(['Real Price', 'Predicted Price'])
    plt.show(block=True)

    return results
