import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

from lib.utils import preproc_pipeline, create_data, model_preproc_pipeline, scale_df, trim_dataset
from lib.model import setup_model, train_model, validate_model
from config import Config

if __name__ == "__main__":

    config = Config()

    stock = create_data(["data/Stocks/aapl.us.txt"])
    data = stock.iloc[:,1:6]
    n_features = data.shape[1]
    
    train_set, validation_set, test_set, scaler = preproc_pipeline(data, False) # This returns train and test set
    x_train, y_train = model_preproc_pipeline(train_set, 
                                              config.look_back, 
                                              config.batch_size)

    nn = setup_model(n_features, 
                     config.batch_size, 
                     config.look_back)
    nn = train_model(nn, 
                     x_train, 
                     y_train, 
                     config.epochs, 
                     config.batch_size, 
                     config.lr)

    nn.save("./{}.h5".format(time.time()))

    x_val, y_val = model_preproc_pipeline(validation_set, 
                                              config.look_back, 
                                              config.batch_size)
    val_pred = nn.predict(x_val)
    #val_pred = scaler.inverse_transform([val_pred])
    #y_val = scaler.inverse_transform([y_val])

    plt.plot(val_pred)
    plt.plot(y_val)
    plt.show()
    

