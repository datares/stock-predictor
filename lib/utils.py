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
    Runs ta on a dataset
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

    return train_set, validation_set, test_set
