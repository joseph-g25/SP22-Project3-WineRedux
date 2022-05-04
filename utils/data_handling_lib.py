# -*- coding: utf-8 -*-

"""
Basic essential tools for handling data prior to training.
"""

import pandas as pd
import joblib as jbl

from sklearn.model_selection import train_test_split, \
    StratifiedShuffleSplit

RANDOM_STATE = int(17)
PATH = 'datasets/winequality-combined.csv'


def load_data(path=PATH, sep=','):
    """
    Loads the original wine project dataset.

    Args:
        path (str, optional): Filepath to the wine dataset. Defaults to PATH.
        sep (str, optional): Character separating individual elements of the wine dataset. Defaults to ",".

    Returns:
        DataFrame: Pandas DataFrame object of the wine dataset, with all features and labels.
    """

    raw_data = pd.read_csv(path, sep=sep)
    return raw_data


def split_dataset(train, labels):
    """
    Basic train-test split using sklearn train_test_split module.

    Parameters:
        train (DataFrame/Array/Series/List) : Dataset features
        labels (DataFrame/Array/Series/List) : Dataset labels

    Returns:
        DataFrame : train_set
        DataFrame : train_set_labels
        DataFrame : test_set
        DataFrame : test_set_labels
    """

    (train_set, train_set_labels, test_set, test_set_labels) = \
        train_test_split(train, labels, random_state=RANDOM_STATE)

    return (train_set, train_set_labels, test_set, test_set_labels)


def strat_split_dataset(
    X,
    label_id,
    n_splits=1,
    test_size=0.15,
    ):
    """AI is creating summary for strat_split_dataset

    Args:
        X ([type]): [description]
        label_id ([type]): [description]
        n_splits (int, optional): [description]. Defaults to 1.
        test_size (float, optional): [description]. Defaults to 0.15.

    Returns:
        ndarray : train_set
        ndarray : train_set_labels
        ndarray : test_set
        ndarray : test_set_labels
    """

    sss = StratifiedShuffleSplit(n_splits=n_splits,
                                 test_size=test_size,
                                 random_state=RANDOM_STATE)

    for (train_index, test_index) in sss.split(X, X[label_id]):
        strat_train_set = X.loc[train_index]
        strat_test_set = X.loc[test_index]

    # split the labels off of both datasets

    train_set_labels = strat_train_set[label_id].copy()
    train_set = strat_train_set.drop(label_id, axis=1)

    test_set_labels = strat_test_set[label_id].copy()
    test_set = strat_test_set.drop(label_id, axis=1)

    return train_set, train_set_labels, test_set, test_set_labels


def save_model(model_obj, filename):
    """Saves a fitted model as a .pkl file.

    Args:
        model_obj (Any): Fitted model object to be exported.
        filename (str): Filename of exported model.
    """

    savedir = 'exports/models/' + filename + '.pkl'
    jbl.dump(model_obj, savedir)


def save_object(obj, category, filename):
    """Saves a given object as a .pkl file.

    Args:
        obj (Any): Object to be exported.
        category (str): Subfolder/category identifier.
        filename (str): Filename of exported object.
    """

    savedir = 'exports/' + category + '/' + filename + '.pkl'
    jbl.dump(obj, savedir)


def load_model(filename):
    """Load an exported model from a .pkl file.

    Args:
        filename (str): Filename of model to be retrieved.

    Returns:
        [type]: [description]
    """

    savedir = 'exports/models/' + filename + '.pkl'
    return jbl.load(savedir)


def load_object(category, filename):
    """Load an exported Python object from a .pkl file.

    Args:
        category (str): Subfolder/category of object to retrieve.
        filename (str): Filename to be retrieved.

    Returns:
        Any: Any Python object.
    """

    savedir = 'exports/' + category + '/' + filename + '.pkl'
    return jbl.load(savedir)
