import pandas as pd
import os
import numpy as np
import matplotlib as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib 
import time
from ta import *

#### DATA CREATION FUNCTIONS ####
def create_data(file_list):
    """
    Utility function to create a dataset from a filelist.
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
def fetch_data():
    """
    Get the files from the data folder. 
    """
    main_dir = os.getcwd()
    # STOCKS
    os.chdir(main_dir)
    os.chdir("./data/Stocks")
    stock_list = os.listdir()
    stocks = create_data(stock_list)
    #ETFs
    os.chdir(main_dir)
    os.chdir("./data/ETFs")
    etf_list = os.listdir()
    etf = create_data(etf_list)

    return stocks, etf



#### DATA PROCESSING FUNCTIONS ####
def scale_df(data):
    """
    This class takes in a pandas dataframe and generates 
    the normalized version of it
    """
    col_names = ['Open', 'High', 'Low', 'Close', 'Volume']

    X = data.loc[:,col_names]

    # scales the data
    scaler = MinMaxScaler()
    df = scaler.fit_transform(X)
    
    # saves the scaler to file
    joblib.dump(scaler, "{}.pkl".format(time.time()))
    return df
def generate_ta(data):
    """
    Runs ta on a dataset and saves to csv.
    """
    # converts data into ta dataframe
    df = add_all_ta_features(data, "Open", "High", "Low", "Close", "Volume", fillna=True)
    df.to_csv("../data/df_ta.csv")
def build_timeseries(df, y_col_index, look_back, type):
    """
    WRONG --> FIX THIS!
    y_col_index is the index of column that would act as output column
    total number of time-series samples would be len(mat) - time steps
    type represents whether it is train or test data
    """
    dim_0 = df.shape[0] - look_back
    dim_1 = df.shape[1]

    if (type == 'train'):
        # input is in shape [batch_size (rows), timesteps, features (cols)]
        x = np.zeros((dim_0, look_back, dim_1))
        for i in range(dim_0):
            x[i] = df[i: i + look_back]
        return x
    elif (type == 'test'):
        y = np.zeros((dim_0,))
        for i in range(dim_0):
            y[i] = df[i + look_back, y_col_index]
        return y
    else:
        return False
def trim_dataset(mat, batch_size):
    """
    trims dataset to a size that's divisible by the batch size
    """

    no_of_rows_drop = mat.shape[0] % batch_size
    if(no_of_rows_drop > 0):
        return mat[:-no_of_rows_drop]
    else:
        return mat
def reshape_data(data, predicted_col, look_back, batch_size, df_type):
    """
    This class takes in a pre_processed pandas dataframe and cleans it
    """
    x = build_timeseries(data, predicted_col, look_back, df_type)
    # Trimming the data to make sure that it will fit the batch size
    x = trim_dataset(x, batch_size)

    return x 

#### FINAL PIPELINE FUNCTION ####
def preproc_pipeline(data, look_back, batch_size, needs_processing=False):
    """
    The preprocessing pipeline takes in a csv of processed data and creates
    the training, validation, and test sets
    """
    # save ta to csv
    if needs_processing:
        stocks, etf = fetch_data()
        data = pd.concat([stocks, etf])
        generate_ta(data)
        # we have to read file
        data = pd.read_csv("./df_ta.csv")

    # Scale values
    data = scale_df(data)
    # Split
    train_set, test_set = train_test_split(data, train_size=0.75, test_size=0.25, shuffle=False)
    
    # # Set up for Keras
    # train_set = reshape_data(train_set, 3, look_back, batch_size, 'train')
    # test_set = reshape_data(test_set, 3, look_back, batch_size, 'test')

    # # We have to reshape the data first before we can split it to a validation set
    # # This is in order to have equally sized test and val
    # validation_set, test_set = np.split(test_set, 2)

    # We could save this to csv.
    return train_set, test_set
