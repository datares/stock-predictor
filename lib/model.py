from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers
from keras.callbacks import CSVLogger

from lib.utils import moving_test_window_preds

def setup_model(BATCH_SIZE, TIME_STEPS):
    """
    Returns a keras LSTM model. Our architecture will be kept 
    in this method.
    """
    model = Sequential()

    model.add(LSTM(50, input_shape = (TIME_STEPS, BATCH_SIZE), return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(50))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    return model

def train_model(model, x_train, y_train, EPOCHS, BATCH_SIZE, lr):
    """
    Takes a training dataset and a model and returns a trained model 
    after ts timesteps.
    """
    optimizer = optimizers.RMSprop(lr=lr)
    model.compile(optimizer = optimizer, loss = 'mean_squared_error')

    model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE)

    return model

def validate_model(validation_data, num_predictions, model, TIME_STEPS=1000):
    """
    Takes a validation dataset and a trained model and validates its performance. 
    Should return the accuracy of the model. 

    We are going to be using this method for testing too. 
    """
    predictions = moving_test_window_preds(validation_data, num_predictions, TIME_STEPS, model)
    
    return predictions
