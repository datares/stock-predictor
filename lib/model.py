from tensorflow import keras

def setup_model():
    """
    Returns a keras LSTM model. Our architecture will be kept 
    in this method.
    """
    raise NotImplementedError

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
