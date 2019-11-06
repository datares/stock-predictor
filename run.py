from lib.utils import preproc_pipeline
import pandas as pd

if __name__ == "__main__":
    data = []
    train_set, test_set = preproc_pipeline(data, 50, 25, True)

