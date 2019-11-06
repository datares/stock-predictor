from lib.utils import preproc_pipeline
import pandas as pd

if __name__ == "__main__":
    data = pd.read_csv("./data/00.csv")
    train_set, validation_set, test_set = preproc_pipeline(data)
