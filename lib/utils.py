import numpy as np
import pandas as pd
import os


def preprocess(data):
    """
    This class takes in a pandas dataframe and cleans it
    """
    raise NotImplementedError

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
    raise NotImplementedError

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

def merge_df(file_list):
    counter = 1
    df_list = pd.DataFrame()
    for file in file_list:
        if (os.stat(file).st_size != 0):
            df = pd.read_csv(file, sep = ",")
            df['symbol'] = file.split(".")[0]
            df_list = df_list.append(df)
            print (counter, " out of ", len(file_list))
            counter += 1
    return pd.DataFrame(df_list)

def fetch_data():
    main_dir = os.getcwd()
    # STOCKS
    os.chdir(main_dir)
    os.chdir("./data/Stocks")
    stock_list = os.listdir()
    stocks = merge_df(stock_list)
    #ETFs
    os.chdir(main_dir)
    os.chdir("./data/ETFs")
    etf_list = os.listdir()
    etf = merge_df(etf_list)

    return stocks, etf
