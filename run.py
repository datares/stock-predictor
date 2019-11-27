import pandas as pd
import numpy as np
from lib.utils import preproc_pipeline, create_data, model_preproc_pipeline, scale_df, trim_dataset
from lib.model import setup_model, train_model, validate_model

if __name__ == "__main__":

    # CREATING DATA
    stock = create_data(["data/Stocks/aapl.us.txt"])
    data = stock.iloc[:,1:2]

    # DEFINING HYPERPARAMETERS
    TIME_STEPS = 100
    BATCH_SIZE = 128
    lr = 0.0001 # learning rate
    EPOCHS = 10

    # PREPROCESSING
    train_set, test_set = preproc_pipeline(data, False) # This returns train and test set
    x_train, y_train = model_preproc_pipeline(train_set, TIME_STEPS, BATCH_SIZE)
    
    # SETTING UP THE MODEL AND TRAINING
    regressor = setup_model(BATCH_SIZE, TIME_STEPS)
    regressor = train_model(regressor, x_train, y_train, EPOCHS, BATCH_SIZE, lr)

    # MAKING PREDICTIONS
    df_test = trim_dataset(test_set, BATCH_SIZE)
    df_val, df_testing = np.split(df_test, 2)

    results = validate_model(df_val, regressor, TIME_STEPS)

    print(results)

