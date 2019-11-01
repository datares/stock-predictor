import numpy as np
import pandas as pd
import os

from sklearn.preprocessing import StandardScaler

# Function definition
def create_data(file_list):
    # create data from text files
    counter = 1
    df_list = pd.DataFrame()
    for file in file_list:
        if (os.stat(file).st_size != 0):
            df = pd.read_csv(file, sep = ",")
            df['symbol'] = file
            df_list = df_list.append(df)
            print (counter, " out of ", len(file_list))
            counter += 1
    return pd.DataFrame(df_list)

def preprocess(data):
    # scaling data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data.reshape(-1, 1))
    return scaled

def window_data(data, window_size):
    # windowing the data
    X, Y = [], []
    i = 0
    while (i + window_size < len(data)):
        # data points
        X.append(data[i:i+window_size])
        # next number in sequence we are trying to predict
        Y.append(data[i+window_size])

        i += 1
    assert len(X) == len(Y)
    return X, Y
