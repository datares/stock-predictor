import pandas as pd
import numpy as np
from lib.utils import preproc_pipeline, create_data, training_preproc_pipeline, scale_df
from lib.model import setup_model, train_model, validate_model

if __name__ == "__main__":

    # CREATING DATA
    data = create_data(["aapl.us.txt"])

    # DEFINING HYPERPARAMETERS
    TIME_STEPS = 60
    BATCH_SIZE = 100
    lr = 0.00010000 # learning rate
    EPOCHS = 25

    # PREPROCESSING
    train_set, test_set = preproc_pipeline(data, TIME_STEPS, BATCH_SIZE, False) # This returns train and test set
    x_train, y_train = training_preproc_pipeline(train_set, TIME_STEPS, BATCH_SIZE)
    
    # SETTING UP THE MODEL AND TRAINING
    regressor = setup_model(BATCH_SIZE, TIME_STEPS)
    regressor = train_model(regressor, x_train, y_train, EPOCHS, BATCH_SIZE, lr)

