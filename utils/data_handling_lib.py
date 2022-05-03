"""
Basic essential tools for handling data prior to training.
"""

import os
import pandas as pd
import numpy as np
import joblib as jbl

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

RANDOM_STATE = int(19)

def load_data(path, sep=","):
    """
    Loads data using Panda's read_csv module.
    
    Parameters:
        String - Filepath to use
        
        String - Separator used in source. Defaults to comma (",")
        
    Returns:
        DataFrame - raw_data
    """
    raw_data = pd.read_csv(path, sep=sep)
    return raw_data

def split_dataset(train, labels):
    """
    Basic train-test split using sklearn train_test_split module.
    
    Parameters:
        DataFrame/Array/Series/List - train
        DataFrame/Array/Series/List - labels
    
    Returns:
        DataFrame : train_set
        DataFrame : train_set_labels
        DataFrame : test_set
        DataFrame : test_set_labels
    """
    train_set, train_set_labels, test_set, test_set_labels = train_test_split(train, labels, random_state=RANDOM_STATE)
    
    return train_set, train_set_labels, test_set, test_set_labels

def strat_split_dataset(data, label_id, n_splits, test_size=0.15):
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=RANDOM_STATE)
    
    for train_index, test_index in sss.split(data, data[label_id]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    

    # split the labels off of both datasets
    train_set_labels = strat_train_set[label_id].copy()
    train_set = strat_train_set.drop(label_id, axis=1)

    test_set_labels = strat_test_set[label_id].copy()
    test_set = strat_test_set.drop(label_id, axis=1)

    return train_set, train_set_labels, test_set, test_set_labels

def save_model(obj, filename):
    savedir = "exports/models/" + filename + ".pkl"
    jbl.dump(obj, savedir)

def load_object(category, filename, filetype= ".pkl"):
    savedir = "exports/" + category + "/" + filename + filetype
    return jbl.load(savedir)
