import numpy as np
import pandas as pd
import os

# Function definition
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