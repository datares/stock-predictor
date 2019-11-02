import numpy as np
import pandas as pd
import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

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
    """
    takes in a dataframe and returns a normalized train test split
    """
    col_names = ['Open', 'High', 'Low', 'Close', 'Volume']

    # splits the data into training and testing set without shuffling (time series)
    df_train, df_test = train_test_split(data, train_size=0.8, test_size=0.2, shuffle=False)

    # subsets the data and get values
    X = df_train.loc[:,col_names].values

    # scales the data
    min_max_scaler = MinMaxScaler()
    df_train = min_max_scaler.fit_transform(X)
    df_test = min_max_scaler.transform(df_test.loc[:,col_names])

    return (df_train, df_test)

def build_timeseries(mat, y_col_index, time_steps):
    # y_col_index is the index of column that would act as output column
    # total number of time-series samples would be len(mat) - TIME_STEPS
    dim_0 = mat.shape[0] - time_steps
    dim_1 = mat.shape[1]
    # input is in shape [batch_size (rows), timesteps, features (cols)]
    x = np.zeros((dim_0, time_steps, dim_1))
    y = np.zeros((dim_0,))
    
    for i in range(dim_0):
        x[i] = mat[i:time_steps+i]
        y[i] = mat[time_steps+i, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y

def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by the batch size
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat
