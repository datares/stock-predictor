from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers
from keras.callbacks import CSVLogger

from lib.utils import moving_test_window_preds

def setup_model(batch_size, look_back):
    """
    Returns a keras LSTM model. Our architecture will be kept 
    in this method.
    """
    model = Sequential()

    model.add(LSTM(50, input_shape = (look_back, 1), return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(50))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    return model

def train_model(model, x_train, y_train, epochs, batch_size, lr):
    """
    Takes a training dataset and a model and returns a trained model 
    after ts timesteps.
    """
    optimizer = optimizers.RMSprop(lr=lr)
    model.compile(optimizer = optimizer, loss = 'mean_squared_error', metrics = ['accuracy'])

    model.fit(x_train, y_train, epochs = epochs, batch_size = batch_size)

    return model

def validate_model(validation_data, num_predictions, model, look_back=1000):
    """
    Takes a validation dataset and a trained model and validates its performance. 
    Should return the accuracy of the model. 

    We are going to be using this method for testing too. 
    """
    predictions = moving_test_window_preds(validation_data, num_predictions, look_back, model)
    
    return predictions
