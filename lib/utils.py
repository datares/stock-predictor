import numpy as np
import pandas as pd

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def create_data(file_list):
    """
    create data from text files
    """
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

def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by the batch size
    """
    no_of_rows_drop = mat.shape[0]%batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat

def preprocess(data):
    """
    This class takes in a pandas dataframe and cleans it
    """
    col_names = ['Open', 'High', 'Low', 'Close', 'Volume']

    X = data.loc[:,col_names]

    # scales the data
    min_max_scaler = MinMaxScaler()
    df = min_max_scaler.fit_transform(X)

    return df

def build_timeseries(mat, y_col_index, time_steps):
    """
    y_col_index is the index of column that would act as output column
    total number of time-series samples would be len(mat) - time steps
    """
    dim_0 = mat.shape[0] - time_steps
    dim_1 = mat.shape[1]
    # input is in shape [batch_size (rows), timesteps, features (cols)]
    x = np.zeros((dim_0, time_steps, dim_1))
    y = np.zeros((dim_0,))
    
    for i in range(dim_0):
        x[i] = mat[i:time_steps+i]
        y[i] = mat[time_steps+i, y_col_index]
    return x, y

def shape_for_keras(data):
    """
    This class takes in a pre_processed pandas dataframe and cleans it
    """
    raise NotImplementedError

def train_val_test_split(data):
    """
    Returns three arrays. 3/4 data (train_set), 1/8 (validation_set),
    1/8 (test_set)
    """
    # splits the data into training and testing set without shuffling (time series)
    df_train, df_test = train_test_split(data, train_size=0.75, test_size=0.25, shuffle=False)
    df_val, df_test = np.split(df_test, 2)

    return df_train, df_val, df_test

def generate_ta(data):
    """
    Runs ta on a dataset
    """
    raise NotImplementedError

def preproc_pipeline(data):
    """
    Preprocesses a dataset to be used for training. 
    """
    # Preprocess
    data = preprocess(data)

    # Optional --> run a technical analysis on it and add more features
    data = generate_ta(data)
    
    # Split
    train_set, validation_set, test_set = train_val_test_split(data)
    
    # Set up for Keras
    train_set = shape_for_keras(train_set)
    validation_set = shape_for_keras(validation_set)
    test_set = shape_for_keras(test_set)

    # We could save this to csv.
    return train_set, validation_set, test_set
