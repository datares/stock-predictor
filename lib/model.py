from tensorflow import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras import optimizers
from keras.callbacks import CSVLogger

from utils import preproc_pipeline

def setup_model(BATCH_SIZE, TIME_STEPS, x_train, lr):
    """
    Returns a keras LSTM model. Our architecture will be kept 
    in this method.
    """
    model = Sequential()

    model.add(LSTM(50, input_shape = (TIME_STEPS, x_train.shape[2]), return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(50, return_sequences = True))
    model.add(Dropout(0.2))

    model.add(LSTM(50))
    model.add(Dropout(0.2))

    model.add(Dense(1))

    optimizer = optimizers.RMSprop(lr=lr)

    return model

def train_model(train_data, model, ts=1000, epochs=10):
    """
    Takes a training dataset and a model and returns a trained model 
    after ts timesteps.
    """
    raise NotImplementedError

def validate_model(validation_data, model, ts=1000):
    """
    Takes a validation dataset and a trained model and validates its performance. 
    Should return the accuracy of the model. 

    We are going to be using this method for testing too. 
    """
    raise NotImplementedError
