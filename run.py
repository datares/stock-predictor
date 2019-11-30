import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
from keras.models import load_model

from lib.utils import preproc_pipeline, create_data, model_preproc_pipeline, generate_dataset
from lib.model import setup_model, train_model
from config import Config

if __name__ == "__main__":


    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True,help='choose between train and test.')
    parser.add_argument('-n', '--name', type=str, default="my_name_is_not_important" ,help='Your model name.')

    args = parser.parse_args()

    stock = create_data(["data/Stocks/aapl.us.txt"])
    data = stock.iloc[:,1:6]
    n_features = data.shape[1]
    train_set, validation_set, test_set, scaler = preproc_pipeline(data, args.name) 

    if args.mode == "train":
        nn = setup_model(n_features, 
                         config.batch_size, 
                         config.look_back)
        x_train, y_train = model_preproc_pipeline(train_set, 
                                                  config.look_back, 
                                                  config.batch_size,
                                                  n_features)
        nn = train_model(nn, 
                         x_train, 
                         y_train, 
                         config.epochs, 
                         config.batch_size, 
                         config.lr)

        nn.save("./saved_models/{}.h5".format(args.name))

    if args.mode == "validation":
        nn = load_model("./saved_models/{}.h5".format(args.name))
        x_val, y_val = model_preproc_pipeline(validation_set, 
                                              config.look_back, 
                                              config.batch_size,
                                              n_features)
        val_pred = nn.predict(x_val)
    
        #val_pred = scaler.inverse_transform([val_pred])
        #y_val = scaler.inverse_transform([y_val])

        plt.plot(val_pred)
        plt.plot(y_val)
        plt.show()
    if args.mode == "dataset":
        #generate_dataset()
        raise NotImplementedError
    
 


    

    

